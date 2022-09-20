# from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.feature_selection import chi2
import numpy as np
np.set_printoptions(threshold=np.inf)
import warnings
warnings.filterwarnings('ignore')
path= "./IPVP_Datasets/ProSE/"
dataset="iPVP data_TAPE.txt"
idx_features_labels = np.genfromtxt("{}{}".format(path, dataset),
                               dtype=np.dtype(str))
data=idx_features_labels[:,0:-1]
target=idx_features_labels[:,-1]
# print(model1.scores_ )
from sklearn.feature_selection import f_classif, SelectKBest  ## 导入 f_classif 检验
X_new = SelectKBest(f_classif, k=186).fit_transform(data, y=target)
#print(X_new)
f = open("./IPVP_Datasets/ProSE/iPVP data_TAPE_ANOVA186new.txt", 'w', encoding="utf-8")
for k,i in enumerate(X_new):
    for j in range(len(i)):
       if j!=len(i)-1:
         f.writelines(str(i[j]).replace('''''', '')+' ')
       else:
         f.writelines(str(i[j]).replace('''''', '')+'\t'+target[k]+'\n')
f.close()
