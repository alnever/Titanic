# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 14:53:00 2018

@author: al_neverov
"""

from features_1 import test, train, y, test_ids
from sklearn.cross_validation import KFold
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier 
from sklearn.svm import SVC
import xgboost as xgb

import seaborn as sea
import matplotlib.pyplot as plt


ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)
        

def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)   

# Put in our parameters for said classifiers
# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }

#### First-level prediction
# Create 5 objects that represent our 4 models
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)  

y_train = y.values
x_train = train.values # Creates an array of the train data
x_test = test.values # Creats an array of the test data   

et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier

rf_feature = rf.feature_importances(x_train,y_train)
et_feature = et.feature_importances(x_train, y_train)
ada_feature = ada.feature_importances(x_train, y_train)
gb_feature = gb.feature_importances(x_train,y_train)

rf_features = [ 0.1274174,   0.24951729,  0.21425103,  0.02226315,  0.04728697,  0.07083099,  0.26843317]
et_features = [ 0.13694681,  0.39941705,  0.07393727,  0.03643434,  0.09528626,  0.03664779,  0.22133049]
ada_features = [ 0.026,  0.03,   0.888,  0.004,  0.002,  0.012,  0.038]
gb_features =  [ 0.02050457,  0.04843825,  0.70195184,  0.05733918,  0.00962386,  0.05281442,  0.10932789]

"""
rf_features = [ 0.09360277,  0.22881467 , 0.07273949 , 0.02712059 , 0.02013902 , 0.12424187,
  0.02082018,  0.01444257,  0.05550353, 0.00782066, 0.02341974 , 0.04649718,
  0.05415284,  0.20154009,  0.0091448 ]
et_features =[ 0.10925439,  0.36487617,  0.02904868 , 0.02538309,  0.0163131 ,  0.03469433,
  0.02688964 , 0.02405262 , 0.0348318,   0.00998148, 0.02395721,  0.07778783,
  0.02999117,  0.18172572 , 0.01121276]
ada_features = [ 0.006,  0.032 , 0.184,  0.01 ,  0.03,   0.598 , 0.016 , 0.008,  0.048,0.004,
  0.002,  0. ,    0.02 ,  0.034 , 0.008]
gb_features = [ 0.03511192,  0.02994557 , 0.3176324 ,  0.01554506,  0.00844086,  0.40736473,
  0.02677235,  0.00823322 , 0.02045711 , 0.03613806 , 0.01660617,  0.01291805,
  0.029943,    0.0304607 ,  0.00443082]
"""

cols = train.columns.values
# Create a dataframe with features
feature_dataframe = pd.DataFrame( {'features': cols,
     'Random Forest feature importances': rf_features,
     'Extra Trees  feature importances': et_features,
      'AdaBoost feature importances': ada_features,
    'Gradient Boost feature importances': gb_features
    })

# Scatter plot 
plt.figure()
sea.stripplot(feature_dataframe['features'].values,feature_dataframe['Random Forest feature importances'].values, size=20)
plt.figure()
sea.stripplot(feature_dataframe['features'].values,feature_dataframe['Extra Trees  feature importances'].values, size=20)
plt.figure()
sea.stripplot(feature_dataframe['features'].values,feature_dataframe['AdaBoost feature importances'].values, size=20)
plt.figure()
sea.stripplot(feature_dataframe['features'].values,feature_dataframe['Gradient Boost feature importances'].values, size=20)

feature_dataframe['mean'] = feature_dataframe.mean(axis= 1)

plt.figure()
sea.barplot(feature_dataframe['features'].values,feature_dataframe['mean'].values)


### Second-level prediction

base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),
     'ExtraTrees': et_oof_train.ravel(),
     'AdaBoost': ada_oof_train.ravel(),
      'GradientBoost': gb_oof_train.ravel()
    })
base_predictions_train.head()

plt.figure()
colormap = plt.cm.RdBu
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sea.heatmap(base_predictions_train.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)

### 
x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)

gbm = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(x_train, y_train)
predictions = gbm.predict(x_test)

submission = pd.DataFrame({ 'PassengerId': test_ids,
                            'Survived': predictions })
submission.to_csv("submission_stacking_all.csv", index=False)