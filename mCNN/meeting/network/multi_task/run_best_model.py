from hyperopt import Trials, STATUS_OK, tpe, rand, atpe
from hyperas import optim
from hyperas.distributions import choice, uniform, loguniform

import os
import numpy as np
from sklearn.utils import class_weight
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras import Input, models, layers, optimizers, callbacks
from mCNN.Network.metrics import test_report
from keras.utils import to_categorical

'suppose that we have neighbor 120'

def data(neighbor_obj):
    kneighbor = neighbor_obj[:-1]
    obj_flag = neighbor_obj[-1]
    if obj_flag == 't':
        obj = 'test_report'
    elif obj_flag == 'v':
        obj = 'val'

    random_seed = 10
    # data = np.load('E:\projects\mCNN\yanglab\mCNN-master\dataset\S2648\mCNN\wild\center_CA_PCA_False_neighbor_%s.npz'%kneighbor)
    data = np.load('/dl/sry/mCNN/dataset/S2648/feature/mCNN/wild/npz/center_CA_PCA_False_neighbor_%s.npz'%kneighbor)
    x = data['x']
    y = data['y']
    ddg = data['ddg'].reshape(-1)

    train_num = x.shape[0]
    indices = [i for i in range(train_num)]
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]
    ddg = ddg[indices]

    positive_indices, negative_indices = ddg >= 0, ddg < 0
    x_positive, x_negative = x[positive_indices], x[negative_indices]
    y_positive, y_negative = y[positive_indices], y[negative_indices]
    ddg_positive, ddg_negative = ddg[positive_indices], ddg[negative_indices]

    left_positive, left_negative = round(0.8 * x_positive.shape[0]), round(0.8 * x_negative.shape[0])
    x_train = np.vstack((x_positive[:left_positive], x_negative[:left_negative]))
    x_test  = np.vstack((x_positive[left_positive:], x_negative[left_negative:]))
    y_train = np.vstack((y_positive[:left_positive], y_negative[:left_negative]))
    y_test  = np.vstack((y_positive[left_positive:], y_negative[left_negative:]))
    ddg_train = np.hstack((ddg_positive[:left_positive], ddg_negative[:left_negative]))
    ddg_test  = np.hstack((ddg_positive[left_positive:], ddg_negative[left_negative:]))

    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train.reshape(-1))
    class_weights_dict = dict(enumerate(class_weights))

    # sort row default is chain
    # reshape and one-hot
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    # normalization
    train_shape = x_train.shape
    test_shape = x_test.shape
    col_train = train_shape[-1]
    col_test = test_shape[-1]
    x_train = x_train.reshape((-1, col_train))
    x_test = x_test.reshape((-1, col_test))
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[np.argwhere(std == 0)] = 0.01
    x_train -= mean
    x_train /= std
    x_test -= mean
    x_test /= std
    x_train = x_train.reshape(train_shape)
    x_test = x_test.reshape(test_shape)

    # reshape
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    return x_train, y_train, ddg_train, x_test, y_test, ddg_test, class_weights_dict,obj


def Conv2DMultiTaskIn1(x_train, y_train, ddg_train, x_test, y_test, ddg_test, class_weights_dict,obj):
        K.clear_session()
        summary = False
        verbose = 0
        # setHyperParams------------------------------------------------------------------------------------------------
        batch_size = 64

        activator = 'elu'
        basic_conv2D_filter_num = 32
        y_dense2_num = 32
        basic_conv2D_layers = 1
        ddg_dense1_num = 128
        ddg_dense2_num = 64
        ddg_reduce_layers = 3  # conv 5 times: 120 => 60 => 30 => 15 => 8 => 4
        y_reduce_layers = 5  # conv 5 times: 120 => 60 => 30 => 15 => 8 => 4
        drop_num = 0.16
        epochs = 150
        loop_dilation2D_dropout_rate = 0.25
        loop_dilation2D_filter_num = 64
        ddg_reduce_conv2D_filter_num = 32
        loop_dilation2D_layers = 2
        lr = 0.009

        optimizer = 'sgd'

        reduce_conv2D_dropout_rate = 0.01

        y_dense1_num = 128

        y_reduce_conv2D_filter_num = 32  # used for reduce dimention



        dilation_lower = 2
        dilation_upper = 16

        ddg_residual_stride = 2
        y_residual_stride = 2

        kernel_size=(3,3)
        pool_size=(2,2)
        initializer='random_uniform'
        padding_style='same'
        loss_type=['mse','binary_crossentropy']
        loss_weights=[0.5,10]
        metrics = (['mae'], ['accuracy'])


        my_callbacks = [
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.8,
                patience=10,
                )
            ]

        if lr > 0:
            if optimizer == 'adam':
                chosed_optimizer = optimizers.Adam(lr=lr)
            elif optimizer == 'sgd':
                chosed_optimizer = optimizers.SGD(lr=lr)
            elif optimizer == 'rmsprop':
                chosed_optimizer = optimizers.RMSprop(lr=lr)


        # build --------------------------------------------------------------------------------------------------------
        ## basic Conv2D
        input_layer = Input(shape=x_train.shape[1:])
        y = layers.Conv2D(basic_conv2D_filter_num,kernel_size,padding=padding_style,kernel_initializer=initializer,activation=activator)(input_layer)
        y = layers.BatchNormalization(axis=-1)(y)
        if basic_conv2D_layers == 2:
            y = layers.Conv2D(basic_conv2D_filter_num,kernel_size,padding=padding_style,kernel_initializer=initializer,activation=activator)(y)
            y = layers.BatchNormalization(axis=-1)(y)
        ## loop with Conv2D with dilation (padding='same')
        for _ in range(loop_dilation2D_layers):
            y = layers.Conv2D(loop_dilation2D_filter_num, kernel_size, padding=padding_style,dilation_rate=dilation_lower, kernel_initializer=initializer, activation=activator)(y)
            y = layers.BatchNormalization(axis=-1)(y)
            y = layers.Dropout(loop_dilation2D_dropout_rate)(y)
            dilation_lower*=2
            if dilation_lower>dilation_upper:
                dilation_lower=2

        ## Conv2D with dilation (padding='valaid') and residual block to reduce dimention.
        ## for regressor branch
        y_ddg = layers.Conv2D(ddg_reduce_conv2D_filter_num, kernel_size, padding=padding_style,
                              kernel_initializer=initializer, activation=activator)(y)
        y_ddg = layers.BatchNormalization(axis=-1)(y_ddg)
        y_ddg = layers.Dropout(reduce_conv2D_dropout_rate)(y_ddg)
        y_ddg = layers.MaxPooling2D(pool_size, padding=padding_style)(y_ddg)
        residual_ddg = layers.Conv2D(ddg_reduce_conv2D_filter_num, 1, strides=ddg_residual_stride, padding='same')(
            input_layer)
        y_ddg = layers.add([y_ddg, residual_ddg])
        ddg_residual_stride *= 2
        for _ in range(ddg_reduce_layers-1):
            y_ddg = layers.Conv2D(ddg_reduce_conv2D_filter_num,kernel_size,padding=padding_style,kernel_initializer=initializer,activation=activator)(y_ddg)
            y_ddg = layers.BatchNormalization(axis=-1)(y_ddg)
            y_ddg = layers.Dropout(reduce_conv2D_dropout_rate)(y_ddg)
            y_ddg = layers.MaxPooling2D(pool_size,padding=padding_style)(y_ddg)
            residual_ddg = layers.Conv2D(ddg_reduce_conv2D_filter_num, 1, strides=ddg_residual_stride, padding='same')(input_layer)
            y_ddg = layers.add([y_ddg, residual_ddg])
            ddg_residual_stride*=2
        ## flat & dense
        y_ddg = layers.Flatten()(y_ddg)
        y_ddg = layers.Dense(ddg_dense1_num, activation=activator)(y_ddg)
        y_ddg = layers.BatchNormalization(axis=-1)(y_ddg)
        y_ddg  = layers.Dropout(drop_num)(y_ddg)
        y_ddg = layers.Dense(ddg_dense2_num, activation=activator)(y_ddg)
        y_ddg = layers.BatchNormalization(axis=-1)(y_ddg)
        y_ddg = layers.Dropout(drop_num)(y_ddg)
        ddg_prediction = layers.Dense(1, name='ddg')(y_ddg)
        # class_prediction = layers.Dense(len(np.unique(y_train)), activation='softmax', name='class')(y_ddg)

        ## for classifier branch
        y_y = layers.Conv2D(y_reduce_conv2D_filter_num, kernel_size, padding=padding_style,
                            kernel_initializer=initializer, activation=activator)(y)
        y_y = layers.BatchNormalization(axis=-1)(y_y)
        y_y = layers.Dropout(reduce_conv2D_dropout_rate)(y_y)
        y_y = layers.MaxPooling2D(pool_size, padding=padding_style)(y_y)
        residual_y = layers.Conv2D(y_reduce_conv2D_filter_num, 1, strides=y_residual_stride, padding='same')(
            input_layer)
        y_y = layers.add([y_y, residual_y])
        y_residual_stride *= 2
        for _ in range(y_reduce_layers-1):
            y_y = layers.Conv2D(y_reduce_conv2D_filter_num,kernel_size,padding=padding_style,kernel_initializer=initializer,activation=activator)(y_y)
            y_y = layers.BatchNormalization(axis=-1)(y_y)
            y_y = layers.Dropout(reduce_conv2D_dropout_rate)(y_y)
            y_y = layers.MaxPooling2D(pool_size,padding=padding_style)(y_y)
            residual_y = layers.Conv2D(y_reduce_conv2D_filter_num, 1, strides=y_residual_stride, padding='same')(input_layer)
            y_y = layers.add([y_y, residual_y])
            y_residual_stride*=2
        ## flat & dense
        y_y = layers.Flatten()(y_y)
        y_y = layers.Dense(y_dense1_num, activation=activator)(y_y)
        y_y = layers.BatchNormalization(axis=-1)(y_y)
        y_y = layers.Dropout(drop_num)(y_y)
        y_y = layers.Dense(y_dense2_num, activation=activator)(y_y)
        y_y = layers.BatchNormalization(axis=-1)(y_y)
        y_y = layers.Dropout(drop_num)(y_y)
        class_prediction = layers.Dense(len(np.unique(y_train)),activation='softmax',name='class')(y_y)


        model = models.Model(inputs=input_layer, outputs=[ddg_prediction,class_prediction])


        if summary:
            model.summary()

        model.compile(optimizer=chosed_optimizer,
                           loss={'ddg':loss_type[0],
                                 'class':loss_type[1]
                                 },
                           loss_weights={'ddg':loss_weights[0],
                                         'class':loss_weights[1]
                                         },
                           metrics={'ddg':metrics[0],
                                    'class':metrics[1]
                                    }
                           )

        # K.set_session(tf.Session(graph=model.output.graph))
        # init = K.tf.global_variables_initializer()
        # K.get_session().run(init)

        result=model.fit(x=x_train,
                  y={'ddg':ddg_train,
                     'class':y_train
                     },
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=verbose,
                  callbacks=my_callbacks,
                  validation_data=(x_test,
                                   {'ddg':ddg_test,
                                    'class':y_test}
                                   ),
                  shuffle=True,
                  class_weight={'ddg':None,
                                'class':class_weights_dict},
                  )


        # print('\n----------History:\n%s'%result.history)

        return model

if __name__ == '__main__':
    import sys
    neighbor_obj,CUDA,CUDA_rate = '120v','3','full'
    ## config TF
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    if CUDA_rate != 'full':
        config = tf.ConfigProto()
        if float(CUDA_rate)<0.1:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = float(CUDA_rate)
        set_session(tf.Session(config=config))

    x_train, y_train, ddg_train, x_test, y_test, ddg_test, class_weights_dict, obj = data(neighbor_obj=neighbor_obj)
    model = Conv2DMultiTaskIn1(x_train, y_train, ddg_train, x_test, y_test, ddg_test, class_weights_dict, obj)
    pearson_coeff, std, acc, mcc, recall_p, recall_n, precision_p, precision_n = test_report(model, x_test, y_test,
                                                                                             ddg_test)
    print('\n----------Predict:'
          '\npearson_coeff: %s, std: %s'
          '\nacc: %s, mcc: %s, recall_p: %s, recall_n: %s, precision_p: %s, precision_n: %s'
          % (pearson_coeff, std, acc, mcc, recall_p, recall_n, precision_p, precision_n))
