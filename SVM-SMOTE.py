import warnings
import numpy as np
warnings.filterwarnings('ignore')
# path= "test-cache/"
# dataset="ProSE_test_result_end.txt"
# idx_features_labels = np.genfromtxt("{}{}".format(path, dataset),dtype=np.dtype(str))
# path1= "test-cache/"
# dataset1="ProSE_train_result_end.txt"
# idx_features_labels1 = np.genfromtxt("{}{}".format(path1, dataset1),
#                                dtype=np.dtype(str))
path= "./M983_ProSE/Inner membrane/"
dataset="M983Inner membrane_ProSE.txt"
idx_features_labels = np.genfromtxt("{}{}".format(path, dataset),dtype=np.dtype(str))
# path1= "test-cache/"
# dataset1="train_result_end.txt"
# idx_features_labels1 = np.genfromtxt("{}{}".format(path1, dataset1),
#                                dtype=np.dtype(str))
X_test=idx_features_labels[:,0:-1].tolist()
Y_test=idx_features_labels[:,-1].tolist()
# X_train=idx_features_labels1[:,0:-1].tolist()
# Y_train=idx_features_labels1[:,-1].tolist()
print(len(X_test))
# print(len(X_train))
from collections import Counter
# 查看所生成的样本类别分布，0和1样本比例9比1，属于类别不平衡数据
print(Counter(Y_test))
# Counter({0: 900, 1: 100})
# 使用imlbearn库中上采样方法中的SMOTE接口
from imblearn.over_sampling import SMOTE, SVMSMOTE
# 定义SMOTE模型，random_state相当于随机数种子的作用
smo = SMOTE(random_state=42)
X_test_smo, Y_test_smo = SVMSMOTE(sampling_strategy='minority').fit_resample(X_test,Y_test)
# X_train_smo, Y_train_smo = SVMSMOTE(sampling_strategy='minority').fit_resample(X_train,Y_train)
f = open("./M983_ProSE/Inner membrane/M983Inner membrane_SVMSMOTE.txt", 'w', encoding="utf-8")
for i in range(len(X_test_smo)):
   f.writelines(str(X_test_smo[i]).replace('[','').replace(']','').replace(',','')+' '+str(Y_test_smo[i])+'\n')
f.close()
# f1 = open("./test-cache/train_SVM-SMOTE.txt", 'w', encoding="utf-8")
# for i in range(len(X_train_smo)):
#    f1.writelines(str(X_train_smo[i]).replace('[','').replace(']','').replace(',','')+' '+str(Y_train_smo[i])+'\n')
# f1.close()
print(len(X_test_smo))
# print(len(Y_train_smo))
# y_train_smo=[]
# y_test_smo=[]
# for i in Y_train_smo:
#     y_train_smo.append(float(i))
# for i in Y_test_smo:
#     y_test_smo.append(float(i))