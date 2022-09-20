import shap
import xgboost
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score,precision_score,f1_score
import lightgbm as lgbm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')
# path= "./data/"f
# dataset="data_end.txt"
# idx_features_labels = np.genfromtxt("{}{}".format(path, dataset),
#                                dtype=np.dtype(str))
# X_train=idx_features_labels[0:434, 0:-1].tolist()
# Y_train=idx_features_labels[0:434, -1].tolist()
# X_test=idx_features_labels[434:536,0:-1].tolist()
# Y_test=idx_features_labels[434:536,-1].tolist()

import numpy as np
np.set_printoptions(threshold=np.inf)
import warnings
warnings.filterwarnings('ignore')
path= "./M983_ProSE/Inner membrane/"
dataset="M983Inner membrane_SVMSMOTE.txt"
idx_features_labels = np.genfromtxt("{}{}".format(path, dataset),
                               dtype=np.dtype(str))
data=idx_features_labels[:,0:-1]
target=idx_features_labels[:,-1]
# load JS visualization code to notebook
shap.initjs()

model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(data, label=target), 100)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(data)
print(data.shape)
# print(shap_values)
# shap.force_plot(explainer.expected_value, shap_values[0,:], X_train[0,:])
shap.force_plot(explainer.expected_value, shap_values[0,:], data[0,:], matplotlib = True, show = True)
# shap.force_plot(explainer.expected_value, shap_values[536:,:], data[536:,:],matplotlib = True, show = True)
shap.summary_plot(shap_values, data)
shap.summary_plot(shap_values, data, plot_type="bar")



