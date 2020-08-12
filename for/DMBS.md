# 2020-08-12
需要下列文件的**路径**

## A.HumVar
1. 原始数据集（**包含uniprotID, 突变描述信息，标签）**    
[path_of_origin_dataset]()   

2. 此数据集上的 **psiblast, hhblits, ssite, NucBind 结果数据**    
[path_of_psiblast]()  
[path_of_hhblits]()  
[path_of_ssite]()  
[path_of_NucBind]()  

3. 此数据集上的 10折中**每一折包含的数据 (每一种分类器，如果它们训练时所采用的 “折” 不一样**)    
[path_of_cv_file_clf_MLP]()   
[path_of_cv_file_clf_NB]()    
[path_of_cv_file_clf_LOG]()   
[path_of_cv_file_clf_ADA]()   
[path_of_cv_file_clf_XGB]()   
[path_of_cv_file_clf_RF]()   

4. 此数据集上的 10折**训练好的模型文件 （每一种分类器, DMBS的全版本和reduce版本**）    
[path_of_MLP_model]()   
[path_of_NB_model]()   
[path_of_LOG_model]()  
[path_of_ADA_model]()  
[path_of_XGB_model]()  
[path_of_DMBS_model]()      
[path_of_DMBS_BR_model]()    
[path_of_DMBS_MSA_model]()    

5. 此数据集上的 10折 的**训练、测试代码 （每一种分类器, DMBS的全版本和reduce版本**）    
[path_of_MLP_code]()  
[path_of_NB_code]()  
[path_of_LOG_code]()  
[path_of_ADA_code]()  
[path_of_XGB_code]()  
[path_of_DMBS_code]()      
[path_of_DMBS_BR_code]()      
[path_of_DMBS_MSA_code]()     

6. 此数据集上的 10折 的**预测结果 (每一种分类器, DMBS的全版本和reduce版本**)    
[path_of_MLP_result]()  
[path_of_NB_result]()  
[path_of_LOG_result]()  
[path_of_ADA_result]()  
[path_of_XGB_result]()   
[path_of_DMBS_result]()      
[path_of_DMBS_BR_result]()     
[path_of_DMBS_MSA_result]()    

7. 此数据集上的 **其他比较方法的预测结果**    
[path_of_SIFT]()   
[path_of_PolyPhen2]()   
[path_of_SNPdryad]()   

## B.HumDiv
**比较的之前方法为**
1. SNPdryad
2. SIFT
3. PolyPhen2  

**其他同 HumVar**

## C.SNPdbe
**比较的之前方法为**
1. SNPdryad
2. SNAP
3. PolyPhen2

**其他同 HumVar**

## D.ExoVar
**比较的之前方法为**
1. SIFT
2. MutationTaster
3. Logit
4. SNPdryad

**其他同 HumVar**