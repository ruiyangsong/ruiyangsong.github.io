
# 网络超参数搜索
## 1. 评估近邻数的影响（110最好）
+ 基于所有特征
+ 只基于 wild 数据 '/dl/sry/mCNN/dataset/deepddg/npz/wild/train_data_neighbor_140.npz'
+ 选定seed随机 70训练 30验证，random_seed = 527
+ 简单网络
+ 50-140 with stepsize 10 angstrog
note, the least neighbor in deepddg is 148,   
pdbid信息： 148 /public/home/sry/mCNN/dataset/SSD/feature/mCNN/wild/csv/2JOF_D_A_9_E/center_CA.csv
## 2. 筛选特征（只去掉冗余的，其他全部保留为最优）
+ 选上诉**简单网络**，其不一定是最优，评估去掉某个特征之后预测结果的变化
+ 选定seed随机 70训练 30验证，random_seed = 527
+ 注意使用和上述一样的训练和测试集
+ 首先测试在全部特征上的效果
```
idx_all = [
    [x for x in range(158) if x not in [24, 25]], #去除无用特征
    [x for x in range(158) if x not in [24, 25] + [x for x in range(1, 6)] + [x for x in range(16, 22)] + [40, 41]], #去除无用特征+冗余特征
    [x for x in range(158) if x not in [24, 25] + [x for x in range(0, 22)]], #去除无用特征+方位特征
    [x for x in range(158) if x not in [24, 25] + [22, 23, 26, 37, 38]], #去除无用特征+深度特征
    [x for x in range(158) if x not in [24, 25] + [x for x in range(27, 37)] + [x for x in range(40, 46)]], #去除无用特征+二级结构信息
    [x for x in range(158) if x not in [24, 25] + [x for x in range(27, 34)] + [x for x in range(40, 46)]], #去除无用特征+二级结构信息1
    [x for x in range(158) if x not in [24, 25] + [x for x in range(34, 37)] + [x for x in range(40, 46)]], #去除无用特征+二级结构信息2
    [x for x in range(158) if x not in [24, 25] + [46, 47]], #去除无用特征+实验条件
    [x for x in range(158) if x not in [24, 25] + [39] + [x for x in range(57, 61)] + [x for x in range(48, 57)] + [x for x in range(61, 81)] + [x for x in range(140, 155)]], #去除无用特征+所有原子编码
    [x for x in range(158) if x not in [24, 25] + [39] + [x for x in range(57, 61)] + [x for x in range(48, 57)] + [x for x in range(140, 145)]], #去除无用特征+原子编码1
    [x for x in range(158) if x not in [24, 25] + [39] + [x for x in range(57, 61)] + [x for x in range(61, 77)] + [x for x in range(145, 153)]], #去除无用特征+原子编码2
    [x for x in range(158) if x not in [24, 25] + [39] + [x for x in range(57, 61)] + [x for x in range(77, 81)] + [x for x in range(153, 155)]], #去除无用特征+原子编码3
    [x for x in range(158) if x not in [24, 25] + [x for x in range(81, 98)]], #去除无用特征+rosetta_energy
    [x for x in range(158) if x not in [24, 25] + [x for x in range(98, 140)] + [x for x in range(155, 158)]] #去除无用特征+msa
]

```
0. 无用特征(所有特征都需要排除此类特征)
```
[24,25]
['occupancy', 'b_factor']
```
1. 冗余特征
[x for x in range(1, 6)] + [x for x in range(16, 22)] + [40, 41]
['omega_Orient', 'theta12', 'theta21', 'phi12', 'phi21',
 'x', 'y', 'z', 'x2CA', 'y2CA', 'z2CA',
 'phi', 'psi']
```
[24,25]
['occupancy', 'b_factor']
```
2. 方位信息
```
[x for x in range(0,22)]
['dist',
 'omega_Orient', 'theta12', 'theta21', 'phi12', 'phi21',
 'sin_omega', 'cos_omega',
 'sin_theta12', 'cos_theta12', 'sin_theta21', 'cos_theta21',
 'sin_phi12', 'cos_phi12', 'sin_phi21', 'cos_phi21',
 'x', 'y', 'z', 'x2CA', 'y2CA', 'z2CA']
```
3. 深度信息
```
idx = [22, 23, 26, 37, 38]
['hse_up', 'hse_down',
 'depth',
 'sa', 'rsa']
```
4. 二级结构信息
```
全部二级结构信息
[x for x in range(27,37)] + [x for x in range(40,46)]

二级结构信息1
[x for x in range(27,34)] + [x for x in range(40,46)]
['s_H', 's_G', 's_I', 's_E', 's_B', 's_T', 's_C',
 'phi', 'psi',
 'sin_phi', 'cos_phi', 
 'sin_psi', 'cos_psi']

二级结构信息2
idx = [x for x in range(34,37)] + [x for x in range(40,46)]
['s_Helix', 's_Strand', 's_Coil',
 'phi', 'psi',
 'sin_phi', 'cos_phi',
 'sin_psi', 'cos_psi']
```
5. 实验条件
```
idx = [46, 47]
['ph', 'temperature']
```
6. 原子编码
```
全部原子编码
[39] + [x for x in range(57,61)] + [x for x in range(48,57)] + [x for x in range(61,81)] + [x for x in range(140,155)]

原子编码1
[39] + [x for x in range(57,61)] + [x for x in range(48,57)] + [x for x in range(140,145)]
['asa', 'C_mass', 'O_mass', 'N_mass', 'S_mass',
 'C', 'O', 'N', 'Other',
 'res_C', 'res_H', 'res_O', 'res_N', 'res_Other',
 'dC', 'dH', 'dO', 'dN', 'dOther']

原子编码2
idx = [39] + [x for x in range(57,61)] + [x for x in range(61,77)] + [x for x in range(145,153)]
['asa', 'C_mass', 'O_mass', 'N_mass', 'S_mass',
 'hydrophobic', 'positive', 'negative', 'neutral', 'acceptor', 'donor', 'aromatic', 'sulphur', 
 'res_hydrophobic', 'res_positive', 'res_negative', 'res_neutral', 'res_acceptor', 'res_donor', 'res_aromatic', 'res_sulphur',
 'dhydrophobic', 'dpositive', 'dnegative', 'dneutral', 'dacceptor', 'ddonor', 'daromatic', 'dsulphur']

原子编码3
idx = [39] + [x for x in range(57,61)] + [x for x in range(77,81)] + [x for x in range(153,155)]
['asa', 'C_mass', 'O_mass', 'N_mass', 'S_mass',
 'hydrophobic_bak', 'polar', 
 'res_hydrophobic_bak', 'res_polar',
 'dhydrophobic_bak', 'dpolar']
```
7. rosetta energy
```
idx = [x for x in range(81,98)]
['fa_atr', 'fa_rep', 'fa_sol', 'fa_intra_rep', 'fa_intra_sol_xover4', 'lk_ball_wtd', 'fa_elec', 'pro_close',
'hbond_bb_sc', 'hbond_sc', 'omega', 'fa_dun', 'p_aa_pp', 'yhh_planarity', 'ref', 'rama_prepro', 'total']
```
8. msa
```
idx = [x for x in range(98,140)] + [x for x in range(155,158)]
['WT_A', 'WT_R', 'WT_N', 'WT_D', 'WT_C', 'WT_Q', 'WT_E', 'WT_G', 'WT_H', 'WT_I',
 'WT_L', 'WT_K', 'WT_M', 'WT_F', 'WT_P', 'WT_S', 'WT_T', 'WT_W', 'WT_Y', 'WT_V', 'WT_-',
 'MT_A', 'MT_R', 'MT_N', 'MT_D', 'MT_C', 'MT_Q', 'MT_E', 'MT_G', 'MT_H', 'MT_I',
 'MT_L', 'MT_K', 'MT_M', 'MT_F', 'MT_P', 'MT_S', 'MT_T', 'MT_W', 'MT_Y', 'MT_V', 'MT_-',
 'dEntropy', 'entWT', 'entMT']
```
## 3. 评估数据增强（先放一放）
+ 基于筛选得到的特征，评估mutant是否有用，两种形式（stack or isolate）
+ stack 需要事先找到突变前后的原子对应关系（但是在原子层面很难对应，在残基层面尚可）
+ 选定seed随机 70训练 30验证，random_seed = 527
+ 仍然基于上述的简单网络
## 4. hyperas确定网络层数和滤波器个数
+ 只优化网络层数，其他的都为固定值(可以选择大卷积核，减少计算量，padding=valid)
+ 选定seed随机 70训练 30验证，random_seed = 527
+ 固定epoch为200
+ hyperas优化的损失为 200 epoch 后最终的 val_loss
+ hyper空间所构成的的最大参数量需要控制不要超过20w
## 5. hyperas确定 dilation 和 reduce 层 的filter 大小
## 6. lr
## 7. 评估batch_size
不仅要看val_loss 也看训练结果是否稳定
[1,8,16,32,64,128]

## feature keys
```text
        self.keys = [		
0           'dist',
1-5         'omega_Orient', 'theta12', 'theta21', 'phi12', 'phi21',
6-15        'sin_omega', 'cos_omega','sin_theta12', 'cos_theta12', 'sin_theta21', 'cos_theta21', 'sin_phi12', 'cos_phi12', 'sin_phi21', 'cos_phi21',
16-21       'x', 'y', 'z', 'x2CA', 'y2CA', 'z2CA',
22-23       'hse_up', 'hse_down',
24-25       'occupancy', 'b_factor',
26          'depth',
27-33       's_H', 's_G', 's_I', 's_E', 's_B', 's_T', 's_C',
34-36       's_Helix', 's_Strand', 's_Coil',
37-39       'sa', 'rsa', 'asa',
40-41       'phi', 'psi',
42-45       'sin_phi', 'cos_phi', 'sin_psi', 'cos_psi',
46-47       'ph', 'temperature',
48-51       'C', 'O', 'N', 'Other',
52-56       'res_C', 'res_H', 'res_O', 'res_N', 'res_Other',
57-60       'C_mass', 'O_mass', 'N_mass', 'S_mass',
61-68       'hydrophobic', 'positive', 'negative', 'neutral', 'acceptor', 'donor', 'aromatic', 'sulphur',
69-76       'res_hydrophobic', 'res_positive', 'res_negative', 'res_neutral', 'res_acceptor', 'res_donor', 'res_aromatic', 'res_sulphur',
77-78       'hydrophobic_bak', 'polar',
79-80       'res_hydrophobic_bak', 'res_polar',
81-97       'fa_atr', 'fa_rep', 'fa_sol', 'fa_intra_rep', 'fa_intra_sol_xover4', 'lk_ball_wtd', 'fa_elec', 'pro_close',
            'hbond_bb_sc', 'hbond_sc', 'omega', 'fa_dun', 'p_aa_pp', 'yhh_planarity', 'ref', 'rama_prepro', 'total',
98-139      'WT_A', 'WT_R', 'WT_N', 'WT_D', 'WT_C', 'WT_Q', 'WT_E', 'WT_G', 'WT_H', 'WT_I',
            'WT_L', 'WT_K', 'WT_M', 'WT_F', 'WT_P', 'WT_S', 'WT_T', 'WT_W', 'WT_Y', 'WT_V', 'WT_-',
			'MT_A', 'MT_R', 'MT_N', 'MT_D', 'MT_C', 'MT_Q', 'MT_E', 'MT_G', 'MT_H', 'MT_I',
			'MT_L', 'MT_K', 'MT_M', 'MT_F', 'MT_P', 'MT_S', 'MT_T', 'MT_W', 'MT_Y', 'MT_V', 'MT_-',
140-144     'dC', 'dH', 'dO', 'dN', 'dOther',
145-152     'dhydrophobic', 'dpositive', 'dnegative', 'dneutral', 'dacceptor', 'ddonor', 'daromatic', 'dsulphur',
153-154     'dhydrophobic_bak', 'dpolar',
155-157     'dEntropy', 'entWT', 'entMT']
```
# dataset
deepddg_all_train = (5425, 120, 158)
deepddg_all_train = (275, 120, 158)