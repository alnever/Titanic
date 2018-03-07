# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 18:05:26 2018

@author: al_neverov
"""

from stacking_1 import x_train, x_test, y_train, test_ids
from sklearn.neural_network import MLPClassifier
import pandas as pd

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1)
clf.fit(x_train, y_train) 
predictions = clf.predict(x_test)

print("Submition preparation...")
submission = pd.DataFrame({ 'PassengerId': test_ids,
                            'Survived': predictions })
submission.to_csv("submission_stack_nnet.csv", index=False)                