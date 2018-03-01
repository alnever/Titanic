# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 10:02:23 2018

@author: al_neverov
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sea

data = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# select Y (target) and X (predictors)
y = data.Survived
x = data.drop(['Survived','PassengerId','Ticket'], axis = 1)
x.is_copy = False
test_ids = test.PassengerId 
test = test.drop(['PassengerId','Ticket'], axis = 1)

## Has Family
x.insert(x.shape[1], 'HasFamily', [1 if x.SibSp[i] + x.Parch[i] > 0 else 0 for i in range(x.shape[0])])
test.insert(test.shape[1], 'HasFamily', [1 if test.SibSp[i] + test.Parch[i] > 0 else 0 for i in range(test.shape[0])])

## FamilySize
x.insert(x.shape[1], 'FamilySize',[x.SibSp[i] + x.Parch[i] + 1 for i in range(x.shape[0])])
test.insert(test.shape[1], 'FamilySize',[test.SibSp[i] + test.Parch[i] + 1 for i in range(test.shape[0])])

## Age Set
x.insert(x.shape[1], 'AgeSet', [1 if not math.isnan(x.Age[i]) else 0 for i in range(x.shape[0]) ])
test.insert(test.shape[1], 'AgeSet', [1 if not math.isnan(test.Age[i]) else 0 for i in range(test.shape[0]) ])

## Age null imputing
x.is_copy = False
test.is_copy = False
x.loc[x.Age.isnull(),'Age'] = np.mean(x.Age)
test.loc[test.Age.isnull(),'Age'] = np.mean(test.Age)

## AgeGroups

def CatMapping(intervals, value):
    idx = 0
    for i in range(len(intervals)):
        if (intervals[i][0] < value) & (intervals[i][1] >= value):
            idx = i + 1
    return(idx)

age_groups = pd.cut(x.Age, 5).cat.categories   
age_groups = [[a.left, a.right] for a in age_groups]   
age_groups[0][0] = 0
age_groups[len(age_groups)-1][1] = np.inf 

x.insert(x.shape[1], 'AgeGroup', [CatMapping(age_groups, a) for a in x.Age])
test.insert(test.shape[1], 'AgeGroup', [CatMapping(age_groups, a) for a in test.Age])

## Has a cabin
x.insert(x.shape[1], 'HasCabin', [1 if str(r) != 'nan' else 0 for r in x.Cabin])
test.insert(test.shape[1], 'HasCabin', [1 if str(r) != 'nan' else 0 for r in test.Cabin])

## Deck selection
x.insert(x.shape[1], 'Deck', [str(r)[0] if str(r) != 'nan' else 'X' for r in x.Cabin])
test.insert(test.shape[1], 'Deck', [str(r)[0] if str(r) != 'nan' else 'X' for r in test.Cabin])

## Titel
x.is_copy = False
test.is_copy = False
x['Titel'] = x.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test['Titel'] = test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

## Embarked null imputing
x.is_copy = False
test.is_copy = False
x.loc[x.Embarked.isnull(), 'Embarked'] = 'X'
test.loc[test.Embarked.isnull(), 'Embarked'] = 'X'

## Fare null imputing
x.is_copy = False
test.is_copy = False
x.loc[x.Fare.isnull(),'Fare'] = np.mean(x.Fare)
test.loc[test.Fare.isnull(),'Fare'] = np.mean(test.Fare)

## FareGroups
fare_groups = pd.cut(x.Fare, 5).cat.categories   
fare_groups = [[a.left, a.right] for a in fare_groups]   
fare_groups[0][0] = 0
fare_groups[len(fare_groups)-1][1] = np.inf 

x.insert(x.shape[1], 'FareGroup', [CatMapping(fare_groups, a) for a in x.Fare])
test.insert(test.shape[1], 'FareGroup', [CatMapping(fare_groups, a) for a in test.Fare])

## Categorial vars coding
x = x.drop(['Name','Cabin'], axis = 1)
test = test.drop(['Name','Cabin'], axis = 1)

"""
If we have thruely categorical valiables

cat_vars = [col for col in x.columns if str(x[col].dtype) == 'category']
for col in cat_vars:
    x[col] = pd.factorize(x[col])[0] + 1
    test[col] = pd.factorize(x[col])[0] + 1
"""

"""
### Using dumies variables 

cat_vars = [col for col in x.columns if x[col].dtype == 'object']    
dummies = pd.get_dummies(x[cat_vars])
x = x.drop(cat_vars, axis = 1)
x = pd.concat([x, dummies], axis = 1)
"""

### Using mapping
cat_vars = [col for col in x.columns if x[col].dtype == 'object']    
for col in cat_vars:
    vals = list(x[col].unique())
    test_vals = list(test[col].unique())
    vals = list(set(vals + test_vals))
    dict = {}
    for v in vals:
        dict[v] = vals.index(v) + 1
    x[col] = x[col].map(dict).astype(int)
    test[col] = test[col].map(dict).astype(int)
    
test = test[x.columns]    

## Correlation analysis

z = pd.concat([y,x], axis = 1)

colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sea.heatmap(z.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)

cor = z.astype(float).corr().Survived
cor_vars = [col for col in cor.index if (math.fabs(cor[col]) > .2)]

g = sea.pairplot(z[cor_vars], 
                 hue='Survived', 
                 palette = 'seismic',
                 size=1.2,
                 diag_kind = 'kde')
g.set(xticklabels=[])

### Remove bad cor variables

cor_vars.remove('Survived')
train = x[cor_vars]
test  = test[cor_vars]

"""

train = x
test = test[x.columns]

"""






