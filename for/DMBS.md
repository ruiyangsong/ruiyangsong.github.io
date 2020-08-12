# 2020-08-12, 需要下列文件的路径

## 1.HumVar
1. 原始数据集（uniprotID, 突变描述信息，标签）
2. 此数据集上的 psiblast, hhblits, ssite, NucBind 结果数据
3. 此数据集上的 10折中每一折包含的数据 (**每一种分类器，如果它们训练时所采用的 “折” 不一样**)
4. 此数据集上的 10折训练好的模型文件 及 训练、测试代码 （**每一种分类器, DMBS的全版本和reduce版本**）
5. 此数据集上的 10折交叉验证的预测结果 (**每一种分类器, DMBS的全版本和reduce版本**)
6. 此数据集上的 之前现有方法的预测结果
**考虑的分类器为**
1. 朴素贝叶斯
2. 逻辑回归
3. 多层感知机
4. Adaboost
5. XGboost
6. RF
**比较的之前方法为**
1. SIFT 
2. PolyPhen2
3. SNPdryad

## 2.HumDiv
**比较的之前方法为**
1. SNPdryad
2. SNAP
3. PolyPhen2
其他同 HumVar

## 3.SNPdbe
**比较的之前方法为**
1. SNPdryad
2. SNAP
3. PolyPhen2
其他同 HumVar

## 4.ExoVar
**比较的之前方法为**
1. SIFT
2. MutationTaster
3. Logit
4. SNPdryad
其他同 HumVar