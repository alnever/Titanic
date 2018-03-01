# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 16:56:19 2018

@author: al_neverov
"""

from stacking_1 import x_train, x_test, y_train, test_ids
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd

def model_XGBoost(train_x, train_y, stop_rounds = 5, depth = 6, rate = 0.02, n_est = 500):
    model = XGBClassifier(max_depth=depth, n_estimators=n, learning_rate=rate)
    model.fit(train_x, train_y, early_stopping_rounds = stop_rounds, eval_set=[(train_x, train_y)], verbose=False)
    sc = model.score(train_x, train_y)    
    prediction = model.predict(train_x)
    mae = mean_absolute_error(train_y, prediction)
    return (sc, mae, prediction, model)

def make_prediction(model, x):
    return (model.predict(x))


max_acc = 0
stops = 0
depth = 0
rate = 0
best_mae  = 0

rows = []
print("Model tuning and selection...")
for n in range(1000,2500,500):
    for i in range(10,110,10):
        for j in range(3,15,3):
            for r in np.arange(.01, .16, .05):
                (acc, mae, prediction, model) = model_XGBoost(x_train, y_train, i,j,r,n)
                print("XGBoost Est=%d\t Stop =%d\t Depth=%d\t Rate=%f\t Acc = %f\t MAE = %f"%(n, i, j, r, acc, mae))
                dict = {}
                dict.update({'Stops':i, 'Depth':j, 'Rate':r, 'Acc': acc, 'MAE': mae})
                rows.append(dict)
                if (acc > max_acc):
                    max_acc = acc
                    xgb_model = model

                
predictions = make_prediction(xgb_model, x_test)

print("Submition preparation...")
submission = pd.DataFrame({ 'PassengerId': test_ids,
                            'Survived': predictions })
submission.to_csv("submission_stack_tuned.csv", index=False)                