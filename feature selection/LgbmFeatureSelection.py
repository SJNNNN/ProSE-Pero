from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
import warnings
import numpy as np
np.set_printoptions(threshold=np.inf)
import torch.nn.functional as F
warnings.filterwarnings('ignore')
path= "./Vesicle/dataset/ProSE/"
dataset="D2_ProSE.txt"
idx_features_labels = np.genfromtxt("{}{}".format(path, dataset),
                               dtype=np.dtype(str))
data=idx_features_labels[:,0:-1]
target=idx_features_labels[:,-1]
model = LGBMClassifier(num_leaves=28,n_estimators=1024,max_depth=8,learning_rate=0.16,min_child_samples=28,random_state=2008,n_jobs=8)
model.fit(data, target)
importantFeatures = model.feature_importances_
Values = np.sort(importantFeatures)[::-1]*0.618
CriticalValue=np.mean(Values)
K=importantFeatures.argsort()[::-1][:5251]
f = open("./Vesicle/dataset/ProSE/D2_ProSE_LGBM.txt", 'w', encoding="utf-8")
for k,i in enumerate(data[:,K]):
    for j in range(len(i)):
       if j!=len(i)-1:
         f.writelines(str(i[j]).replace('[','').replace(']','').replace('''''', '')+' ')
       else:
         f.writelines(str(i[j]).replace('[','').replace(']','').replace('''''', '')+' '+target[k]+'\n')
         # f.writelines(str(i[j]).replace('''''', '') + '\n')
f.close()
