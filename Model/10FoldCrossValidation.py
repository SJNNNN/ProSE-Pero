import numpy as np
import lightgbm as lgbm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import plot_importance
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score, recall_score, matthews_corrcoef
from sklearn.metrics import f1_score, accuracy_score, precision_score
from sklearn.naive_bayes import GaussianNB
import warnings
import numpy as np
warnings.filterwarnings('ignore')
path= "./IPVP_Datasets/ProSE/"
dataset="iPVP data_ProSE_ANOVA606.txt"
idx_features_labels = np.genfromtxt("{}{}".format(path, dataset),
                               dtype=np.dtype(str))
X_train=idx_features_labels[0:400, 0:-1]
Y_train=idx_features_labels[0:400, -1]
X_test=idx_features_labels[400:548,0:-1]
Y_test=idx_features_labels[400:548,-1]
X_train_smo=[]
X_test_smo=[]
for i in range(len(X_train)):
       x_train1 = []
       m=X_train[i]
       for j in range(len(m)):
              x_train1.append(float(m[j]))
       X_train_smo.append(x_train1)
for i in range(len(X_test)):
       x_test1 = []
       m=X_test[i]
       for j in range(len(m)):
              x_test1.append(float(m[j]))
       X_test_smo.append(x_test1)
y_train_smo=[]
y_test_smo=[]
for i in Y_train:
    y_train_smo.append(float(i))
for i in Y_test:
    y_test_smo.append(float(i))
KF = KFold(n_splits=5,shuffle=True,random_state=200)
print("---GaussianNB---")
lr = GaussianNB()
acc=[]
prec=[]
recall=[]
f1=[]
auc=[]
spe=[]
mcc=[]
for train_index,test_index in KF.split(X_train):
    # 朴素贝叶斯模型预测
    # rf = RandomForestClassifier()
    lr.fit(X_train[train_index], Y_train[train_index])
    pre_test = lr.predict(X_train[test_index])
    lr_score = lr.predict_proba(X_train[test_index])
    test_acc = accuracy_score( Y_train[test_index] ,pre_test)
    test_prec = precision_score(Y_train[test_index], pre_test, pos_label='1')
    test_recall = recall_score(Y_train[test_index] ,pre_test, pos_label='1')
    test_f1 = f1_score(Y_train[test_index], pre_test, pos_label='1')
    test_auc = roc_auc_score(Y_train[test_index], lr_score[:,1])
    auc.append(test_auc)
    # test_auc = roc_auc_score(Y_train[test_index], lr_score[:,1])
    con_matrix = confusion_matrix(Y_train[test_index] ,pre_test)
    test_spec = con_matrix[0][0] / (con_matrix[0][0] + con_matrix[0][1])
    test_mcc = (con_matrix[0][0] * con_matrix[1][1] - con_matrix[0][1] * con_matrix[1][0]) / (((con_matrix[1][1] + con_matrix[0][1]) *(con_matrix[1][1]+con_matrix[1][0])*(con_matrix[0][0]+con_matrix[0][1])*(con_matrix[0][0]+con_matrix[1][0]))**0.5)
    acc.append(test_acc)
    prec.append(test_prec)
    recall.append(test_recall)
    f1.append(test_f1)
    # auc.append(test_auc)
    spe.append(test_spec)
    mcc.append(test_mcc)
    f = open("./IPVP_Datasets/ProSE/fold/GaussianNB_elasticNet.txt", 'w', encoding="utf-8")
    p = lr_score[:, 1].tolist()
    l = Y_train[test_index]
    for j in range(len(p)):
        f.writelines(str(p[j]) + " " + str(int(l[j])) + "\n")
    f.close()
print("acc: ", np.mean(acc), " ; prec: ", np.mean(prec), " ; recall: ", np.mean(recall), " ; f1: ", np.mean(f1), " ; auc: ",
          np.mean(auc), " ; spec:", np.mean(spe), " ; mcc: ", np.mean(mcc))
print("---LR---")
lr = LogisticRegression()
acc=[]
prec=[]
recall=[]
f1=[]
auc=[]
spe=[]
mcc=[]
for train_index,test_index in KF.split(X_train):
    # RF模型预测
    # rf = RandomForestClassifier()
    lr.fit(X_train[train_index], Y_train[train_index])
    pre_test = lr.predict(X_train[test_index])
    lr_score = lr.predict_proba(X_train[test_index])
    test_acc = accuracy_score( Y_train[test_index] ,pre_test)
    test_prec = precision_score(Y_train[test_index], pre_test, pos_label='1')
    test_recall = recall_score(Y_train[test_index] ,pre_test, pos_label='1')
    test_f1 = f1_score(Y_train[test_index], pre_test, pos_label='1')
    try:
        test_auc = roc_auc_score(Y_train[test_index], lr_score[:, 1])
        auc.append(test_auc)
    except ValueError:
        pass
    con_matrix = confusion_matrix(Y_train[test_index] ,pre_test)
    test_spec = con_matrix[0][0] / (con_matrix[0][0] + con_matrix[0][1])
    test_mcc = (con_matrix[0][0] * con_matrix[1][1] - con_matrix[0][1] * con_matrix[1][0]) / (((con_matrix[1][1] + con_matrix[0][1]) *(con_matrix[1][1]+con_matrix[1][0])*(con_matrix[0][0]+con_matrix[0][1])*(con_matrix[0][0]+con_matrix[1][0]))**0.5)
    acc.append(test_acc)
    prec.append(test_prec)
    recall.append(test_recall)
    f1.append(test_f1)
    # auc.append(test_auc)
    spe.append(test_spec)
    mcc.append(test_mcc)
    f = open("./IPVP_Datasets/ProSE/fold/LR_elasticNet.txt", 'w', encoding="utf-8")
    p = lr_score[:, 1].tolist()
    l = Y_train[test_index]
    for j in range(len(p)):
        f.writelines(str(p[j]) + " " + str(int(l[j])) + "\n")
    f.close()
print("acc: ", np.mean(acc), " ; prec: ", np.mean(prec), " ; recall: ", np.mean(recall), " ; f1: ", np.mean(f1), " ; auc: ",
          np.mean(auc), " ; spec:", np.mean(spe), " ; mcc: ", np.mean(mcc))
print("----RF----")
model = RandomForestClassifier()
rf = RandomForestClassifier()
acc=[]
prec=[]
recall=[]
f1=[]
auc=[]
spe=[]
mcc=[]
# rf.fit(X_train_smo, y_train_smo)
# pre_test = rf.predict_proba(X_test_smo)
# pres = rf.predict(X_test_smo)  # 使用训练好的模型lr对X_test进行预测
# test_acc = accuracy_score( y_test_smo, pres)
for train_index,test_index in KF.split(X_train):
    # RF模型预测
    # rf = RandomForestClassifier()
    rf.fit(X_train[train_index], Y_train[train_index])
    pre_test = rf.predict(X_train[test_index])
    lr_score = rf.predict_proba(X_train[test_index])
    test_acc = accuracy_score(Y_train[test_index] ,pre_test)
    test_prec = precision_score(Y_train[test_index], pre_test, pos_label='1')
    test_recall = recall_score(Y_train[test_index] ,pre_test, pos_label='1')
    test_f1 = f1_score(Y_train[test_index], pre_test, pos_label='1')
    test_auc = roc_auc_score(Y_train[test_index], lr_score[:,1])
    con_matrix = confusion_matrix(Y_train[test_index] ,pre_test)
    test_spec = con_matrix[0][0] / (con_matrix[0][0] + con_matrix[0][1])
    test_mcc = (con_matrix[0][0] * con_matrix[1][1] - con_matrix[0][1] * con_matrix[1][0]) / (((con_matrix[1][1] + con_matrix[0][1]) *(con_matrix[1][1]+con_matrix[1][0])*(con_matrix[0][0]+con_matrix[0][1])*(con_matrix[0][0]+con_matrix[1][0]))**0.5)
    acc.append(test_acc)
    prec.append(test_prec)
    recall.append(test_recall)
    f1.append(test_f1)
    auc.append(test_auc)
    spe.append(test_spec)
    mcc.append(test_mcc)
    f = open("./IPVP_Datasets/ProSE/fold/RF_elasticNet.txt", 'w', encoding="utf-8")
    p = lr_score[:, 1].tolist()
    l = Y_train[test_index]
    for j in range(len(p)):
        f.writelines(str(p[j]) + " " + str(int(l[j])) + "\n")
    f.close()

print("acc: ", np.mean(acc), " ; prec: ", np.mean(prec), " ; recall: ", np.mean(recall), " ; f1: ", np.mean(f1), " ; auc: ",
          np.mean(auc), " ; spec:", np.mean(spe), " ; mcc: ", np.mean(mcc))
print("----SVM----")
model = svm.SVC(C=2, kernel='linear', gamma=10, decision_function_shape='ovr',probability=True)
for train_index,test_index in KF.split(X_train):
    # model = svm.SVC(C=2, kernel='linear', gamma=10, decision_function_shape='ovr')
    model.fit(X_train[train_index], Y_train[train_index])
    lr_pres = model.predict(X_train[test_index])
    lr_score = model.predict_proba(X_train[test_index])# 使用训练好的模型lr对X_test进行预测
    test_acc = accuracy_score(Y_train[test_index], lr_pres)
    test_prec = precision_score(Y_train[test_index], lr_pres, pos_label='1')
    test_recall = recall_score(Y_train[test_index], lr_pres, pos_label='1')
    test_f1 = f1_score(Y_train[test_index], lr_pres, pos_label='1')
    test_auc = roc_auc_score(Y_train[test_index],  lr_score[:,1])
    con_matrix = confusion_matrix(Y_train[test_index], lr_pres)
    test_spec = con_matrix[0][0] / (con_matrix[0][0] + con_matrix[0][1])
    test_mcc = (con_matrix[0][0] * con_matrix[1][1] - con_matrix[0][1] * con_matrix[1][0]) / (
    ((con_matrix[1][1] + con_matrix[0][1])*(con_matrix[1][1]+con_matrix[1][0])*(con_matrix[0][0]+con_matrix[0][1])*(con_matrix[0][0]+con_matrix[1][0]))**0.5)
    acc.append(test_acc)
    prec.append(test_prec)
    recall.append(test_recall)
    f1.append(test_f1)
    auc.append(test_auc)
    spe.append(test_spec)
    mcc.append(test_mcc)
    f = open("./IPVP_Datasets/ProSE/fold/SVM_elasticNet.txt", 'w', encoding="utf-8")
    p = lr_score[:, 1].tolist()
    l = Y_train[test_index]
    for j in range(len(p)):
        f.writelines(str(p[j]) + " " + str(int(l[j])) + "\n")
    f.close()
print("acc: ", np.mean(acc), " ; prec: ", np.mean(prec), " ; recall: ", np.mean(recall), " ; f1: ", np.mean(f1),
          " ; auc: ",np.mean(auc), " ; spec:", np.mean(spe), " ; mcc: ", np.mean(mcc))
print("----LightGBM----")
lgbm = lgbm.LGBMClassifier(num_leaves=60, learning_rate=0.5, n_estimators=40)
for train_index,test_index in KF.split(X_train):
    lgbm.fit(X_train[train_index], Y_train[train_index])
    y_pre = lgbm.predict(X_train[test_index])
    lr_score = lgbm.predict_proba(X_train[test_index])
    test_acc = accuracy_score(Y_train[test_index], y_pre)
    test_prec = precision_score(Y_train[test_index], y_pre, pos_label='1')
    test_recall = recall_score(Y_train[test_index], y_pre, pos_label='1')
    test_f1 = f1_score(Y_train[test_index], y_pre, pos_label='1')
    test_auc = roc_auc_score(Y_train[test_index], lr_score[:,1])
    con_matrix = confusion_matrix(Y_train[test_index], y_pre)
    test_spec = con_matrix[0][0] / (con_matrix[0][0] + con_matrix[0][1])
    test_mcc = (con_matrix[0][0] * con_matrix[1][1] - con_matrix[0][1] * con_matrix[1][0]) / (((con_matrix[1][1] + con_matrix[0][1]) * (con_matrix[1][1] + con_matrix[1][0]) * (con_matrix[0][0] + con_matrix[0][1]) * (con_matrix[0][0] + con_matrix[1][0])) ** 0.5)
    acc.append(test_acc)
    prec.append(test_prec)
    recall.append(test_recall)
    f1.append(test_f1)
    auc.append(test_auc)
    spe.append(test_spec)
    mcc.append(test_mcc)
    f = open("./IPVP_Datasets/ProSE/fold/LightGBM_elasticNet.txt", 'w', encoding="utf-8")
    p = lr_score[:, 1].tolist()
    l = Y_train[test_index]
    for j in range(len(p)):
        f.writelines(str(p[j]) + " " + str(int(l[j])) + "\n")
    f.close()
print("acc: ", np.mean(acc), " ; prec: ", np.mean(prec), " ; recall: ", np.mean(recall), " ; f1: ", np.mean(f1),
          " ; auc: ", np.mean(auc), " ; spec:", np.mean(spe), " ; mcc: ", np.mean(mcc))

print("----GBDT----")
for train_index,test_index in KF.split(X_train):
    params = {'n_estimators': 500,  # 弱分类器的个数
              'max_depth': 3,  # 弱分类器（CART回归树）的最大深度
              'min_samples_split': 5,  # 分裂内部节点所需的最小样本数
              'learning_rate': 0.5,  # 学习率
              'loss': 'exponential'}
    GBDTreg = GradientBoostingClassifier(**params)
    GBDTreg.fit(X_train[train_index], Y_train[train_index])
    y_pre = GBDTreg.predict(X_train[test_index])
    r_score = GBDTreg.predict_proba(X_train[test_index])
    test_acc = accuracy_score(Y_train[test_index], y_pre)
    test_prec = precision_score(Y_train[test_index], y_pre, pos_label='1')
    test_recall = recall_score(Y_train[test_index], y_pre, pos_label='1')
    test_f1 = f1_score(Y_train[test_index], y_pre, pos_label='1')
    test_auc = roc_auc_score(Y_train[test_index], r_score[:,1])
    con_matrix = confusion_matrix(Y_train[test_index], y_pre)
    test_spec = con_matrix[0][0] / (con_matrix[0][0] + con_matrix[0][1])
    test_mcc = (con_matrix[0][0] * con_matrix[1][1] - con_matrix[0][1] * con_matrix[1][0]) / (((con_matrix[1][1] + con_matrix[0][1]) * (con_matrix[1][1] + con_matrix[1][0]) * (con_matrix[0][0] + con_matrix[0][1]) * (con_matrix[0][0] + con_matrix[1][0])) ** 0.5)
    acc.append(test_acc)
    prec.append(test_prec)
    recall.append(test_recall)
    f1.append(test_f1)
    auc.append(test_auc)
    spe.append(test_spec)
    mcc.append(test_mcc)
    f = open("./IPVP_Datasets/ProSE/fold/GBDT_elasticNet.txt", 'w', encoding="utf-8")
    p = r_score[:, 1].tolist()
    l = Y_train[test_index]
    for j in range(len(p)):
        f.writelines(str(p[j]) + " " + str(int(l[j])) + "\n")
    f.close()
print("acc: ", np.mean(acc), " ; prec: ", np.mean(prec), " ; recall: ", np.mean(recall), " ; f1: ", np.mean(f1),
          " ; auc: ", np.mean(auc), " ; spec:", np.mean(spe), " ; mcc: ", np.mean(mcc))
print("----MLP----")
clf = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(30,20,10),random_state=1)
acc=[]
prec=[]
recall=[]
f1=[]
auc=[]
spe=[]
mcc=[]
for train_index,test_index in KF.split(X_train):
    # rf = RandomForestClassifier()
    clf.fit(X_train[train_index], Y_train[train_index])
    pre_test = clf.predict(X_train[test_index])
    lr_score = clf.predict_proba(X_train[test_index])
    test_acc = accuracy_score(Y_train[test_index] ,pre_test)
    test_prec = precision_score( Y_train[test_index], pre_test, pos_label='1')
    test_recall = recall_score( Y_train[test_index] ,pre_test, pos_label='1')
    test_f1 = f1_score( Y_train[test_index], pre_test, pos_label='1')
    test_auc = roc_auc_score( Y_train[test_index], lr_score[:,1])
    con_matrix = confusion_matrix( Y_train[test_index] ,pre_test)
    test_spec = con_matrix[0][0] / (con_matrix[0][0] + con_matrix[0][1])
    test_mcc = (con_matrix[0][0] * con_matrix[1][1] - con_matrix[0][1] * con_matrix[1][0]) / (((con_matrix[1][1] + con_matrix[0][1]) *(con_matrix[1][1]+con_matrix[1][0])*(con_matrix[0][0]+con_matrix[0][1])*(con_matrix[0][0]+con_matrix[1][0]))**0.5)
    acc.append(test_acc)
    prec.append(test_prec)
    recall.append(test_recall)
    f1.append(test_f1)
    auc.append(test_auc)
    spe.append(test_spec)
    mcc.append(test_mcc)
    f = open("./IPVP_Datasets/ProSE/fold/MLP_elasticNet.txt", 'w', encoding="utf-8")
    p = lr_score[:, 1].tolist()
    l = Y_train[test_index]
    for j in range(len(p)):
        f.writelines(str(p[j]) + " " + str(int(l[j])) + "\n")
    f.close()
print("acc: ", np.mean(acc), " ; prec: ", np.mean(prec), " ; recall: ", np.mean(recall), " ; f1: ", np.mean(f1), " ; auc: ",
          np.mean(auc), " ; spec:", np.mean(spe), " ; mcc: ", np.mean(mcc))
print("----KNN----")
for train_index,test_index in KF.split(X_train):
    # print(train_index)
    # print(test_index)
    knn = KNeighborsClassifier()  # 实例化KNN模型
    knn.fit(X_train[train_index], Y_train[train_index])
    Y_pre = knn.predict(X_train[test_index])
    lr_score = knn.predict_proba(X_train[test_index])
    test_acc = accuracy_score(Y_train[test_index], Y_pre)
    test_prec = precision_score(Y_train[test_index], Y_pre, pos_label='1')
    test_recall = recall_score(Y_train[test_index], Y_pre, pos_label='1')
    test_f1 = f1_score(Y_train[test_index], Y_pre, pos_label='1')
    test_auc = roc_auc_score(Y_train[test_index], lr_score[:,1])
    con_matrix = confusion_matrix(Y_train[test_index], Y_pre)
    test_spec = con_matrix[0][0] / (con_matrix[0][0] + con_matrix[0][1])
    test_mcc = (con_matrix[0][0] * con_matrix[1][1] - con_matrix[0][1] * con_matrix[1][0]) / (((con_matrix[1][1] + con_matrix[0][1]) * (con_matrix[1][1] + con_matrix[1][0]) * (con_matrix[0][0] + con_matrix[0][1]) * (con_matrix[0][0] + con_matrix[1][0])) ** 0.5)
    acc.append(test_acc)
    prec.append(test_prec)
    recall.append(test_recall)
    f1.append(test_f1)
    auc.append(test_auc)
    spe.append(test_spec)
    mcc.append(test_mcc)
    f = open("./IPVP_Datasets/ProSE/fold/KNN_elasticNet.txt", 'w', encoding="utf-8")
    p = lr_score[:, 1].tolist()
    l = Y_train[test_index]
    for j in range(len(p)):
        f.writelines(str(p[j]) + " " + str(int(l[j])) + "\n")
    f.close()
print("acc: ", np.mean(acc), " ; prec: ", np.mean(prec), " ; recall: ", np.mean(recall), " ; f1: ", np.mean(f1),
          " ; auc: ", np.mean(auc), " ; spec:", np.mean(spe), " ; mcc: ", np.mean(mcc))

#XGboost
print("----XGboost独立测试----")
model = XGBClassifier(learning_rate=0.5,
                        n_estimators=200,         # 树的个数--1000棵树建立xgboost
                        max_depth=6,               # 树的深度
                        min_child_weight = 1,      # 叶子节点最小权重
                        gamma=0.,                  # 惩罚项中叶子结点个数前的参数
                        subsample=0.8,             # 随机选择80%样本建立决策树
                        colsample_btree=0.8,       # 随机选择80%特征建立决策树
                        objective='binary:logitraw', # 指定损失函数
                        scale_pos_weight=1,        # 解决样本个数不平衡的问题
                        random_state=27,   # 随机数
                       )
model.fit(np.array(X_train_smo),np.array(y_train_smo),eval_set = [(np.array(X_test_smo), np.array(y_test_smo))],eval_metric = "logloss",early_stopping_rounds = 10,verbose = True)


fig,ax = plt.subplots(figsize=(15,15))
plot_importance(model,height=0.5,ax=ax,max_num_features=64)
# plt.show()

### make prediction for test data
Y_pred = model.predict(np.array(X_test_smo))
Y_score = model.predict_proba(np.array(X_test_smo))
test_acc = accuracy_score(np.array(y_test_smo), Y_pred)
test_prec = precision_score(np.array(y_test_smo),Y_pred,pos_label=1)
test_recall = recall_score(np.array(y_test_smo), Y_pred,pos_label=1)
test_f1 = f1_score(np.array(y_test_smo), Y_pred,pos_label=1)
test_auc = roc_auc_score(np.array(y_test_smo), Y_score[:,1])
con_matrix = confusion_matrix(np.array(y_test_smo), Y_pred)
test_spec = con_matrix[0][0]/(con_matrix[0][0]+con_matrix[0][1])
test_mcc = (con_matrix[0][0]*con_matrix[1][1]-con_matrix[0][1]*con_matrix[1][0])/(((con_matrix[1][1]+con_matrix[0][1])*(con_matrix[1][1]+con_matrix[1][0])*(con_matrix[0][0]+con_matrix[0][1])*(con_matrix[0][0]+con_matrix[1][0]))**0.5)
f = open("./IPVP_Datasets/ProSE/fold/XGBoost_result.txt", 'w', encoding="utf-8")
p = Y_score[:,1].tolist()
l = y_test_smo
for j in range(len(p)):
  f.writelines(str(p[j]) + " " + str(int(l[j])) + "\n")
f.close()
print("acc: ",test_acc," ; prec: ",test_prec," ; recall: ",test_recall," ; f1: ",test_f1," ; auc: ",test_auc," ; spec:",test_spec," ; mcc: ",test_mcc)
print("----XGboost10折交叉验证----")
model = XGBClassifier(learning_rate=0.1,
                        n_estimators=1000,         # 树的个数--1000棵树建立xgboost
                        max_depth=6,               # 树的深度
                        min_child_weight = 1,      # 叶子节点最小权重
                        gamma=0.,                  # 惩罚项中叶子结点个数前的参数
                        subsample=0.8,             # 随机选择80%样本建立决策树
                        colsample_btree=0.8,       # 随机选择80%特征建立决策树
                        objective='binary:logitraw', # 指定损失函数
                        scale_pos_weight=1,        # 解决样本个数不平衡的问题
                        random_state=27            # 随机数
                       )
for train_index,test_index in KF.split(X_train_smo):
    print(np.array(y_train_smo)[test_index])
    model.fit(np.array(X_train_smo)[train_index],np.array(y_train_smo)[train_index],eval_set = [(np.array(X_train_smo)[test_index ], np.array(y_train_smo)[test_index ])],eval_metric = "logloss",early_stopping_rounds = 10,verbose = True)
    fig,ax = plt.subplots(figsize=(15,15))
    plot_importance(model,height=0.5,ax=ax,max_num_features=64)
    # plt.show()
### make prediction for test data
    Y_pred = model.predict(np.array(X_train_smo)[test_index ])
    lr_score = lr.predict_proba(np.array(X_train_smo)[test_index])
    test_acc = accuracy_score(np.array(y_train_smo)[test_index], Y_pred)
    test_prec = precision_score(np.array(y_train_smo)[test_index ],Y_pred,pos_label=1)
    test_recall = recall_score(np.array(y_train_smo)[test_index ], Y_pred,pos_label=1)
    test_f1 = f1_score(np.array(y_train_smo)[test_index ], Y_pred,pos_label=1)
    test_auc = roc_auc_score(np.array(y_train_smo)[test_index ], lr_score[:,1])
    con_matrix = confusion_matrix(np.array(y_train_smo)[test_index ], Y_pred)
    test_spec = con_matrix[0][0]/(con_matrix[0][0]+con_matrix[0][1])
    test_mcc = (con_matrix[0][0] * con_matrix[1][1] - con_matrix[0][1] * con_matrix[1][0]) /(((con_matrix[1][1]+con_matrix[0][1])*(con_matrix[1][1]+con_matrix[1][0])*(con_matrix[0][0]+con_matrix[0][1])*(con_matrix[0][0]+con_matrix[1][0]))**0.5)
    acc.append(test_acc)
    prec.append(test_prec)
    recall.append(test_recall)
    f1.append(test_f1)
    auc.append(test_auc)
    spe.append(test_spec)
    mcc.append(test_mcc)
    f = open("./IPVP_Datasets/ProSE/fold/XGBoost_elasticNet.txt", 'w', encoding="utf-8")
    p = lr_score[:, 1].tolist()
    l = np.array(y_train_smo)[test_index]
    for j in range(len(p)):
        f.writelines(str(p[j]) + " " + str(int(l[j])) + "\n")
    f.close()
print("acc: ", np.mean(acc), " ; prec: ", np.mean(prec), " ; recall: ", np.mean(recall), " ; f1: ", np.mean(f1),
          " ; auc: ", np.mean(auc), " ; spec:", np.mean(spe), " ; mcc: ", np.mean(mcc))


