
import pandas as pd
import re
import math
import numpy as np
from sklearn.preprocessing import Imputer, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier

def model_XGBoost(train_x, train_y, test_x, test_y, n_est = 100):

    model = XGBClassifier(n_estimators=n_est, learning_rate=0.1)
    model.fit(train_x, train_y, early_stopping_rounds = 5, eval_set=[(test_x, test_y)], verbose=False)
    sc = model.score(test_x, test_y)    
    prediction = model.predict(test_x)
    mae = mean_absolute_error(test_y, prediction)
    return (sc, mae, prediction, model)


def model_GBC(train_x, train_y, test_x, test_y, n_est = 100):
    model = GradientBoostingClassifier(n_estimators = n_est, learning_rate = .05)
    model.fit(train_x, train_y)
    sc = model.score(test_x, test_y)
    prediction = model.predict(test_x)
    mae = mean_absolute_error(test_y, prediction)
    return (sc, mae, prediction, model)

def model_RandomForest(train_x, train_y, test_x, test_y, n_est = 100):
    model = RandomForestClassifier(n_estimators = n_est)
    model.fit(train_x, train_y)
    sc = model.score(test_x, test_y)
    prediction = model.predict(test_x)
    mae = mean_absolute_error(test_y, prediction)
    return (sc, mae, prediction, model)

def model_CVS(train_x, train_y, test_x, test_y, n_est = 100):
    model = svm.SVC()
    model.fit(train_x, train_y)
    sc = model.score(test_x, test_y)
    prediction = model.predict(test_x)
    mae = mean_absolute_error(test_y, prediction)
    return (sc, mae, prediction, model)

def model_SGD(train_x, train_y, test_x, test_y, n_est = 100):
    model = SGDClassifier()
    model.fit(train_x, train_y)
    sc = model.score(test_x, test_y)
    prediction = model.predict(test_x)
    mae = mean_absolute_error(test_y, prediction)
    return (sc, mae, prediction, model)

def make_prediction(model, x, ids):
    return (model.predict(x))


np.random.seed(0)

training = pd.read_csv("train.csv")
testing = pd.read_csv("test.csv")

# Create target and predictors variable
y = training.Survived
predictors = training.drop(['PassengerId', 'Name', 'Ticket', 'Survived'], axis = 1)
test_ids = testing.PassengerId
test_predictors = testing.drop(['PassengerId', 'Name', 'Ticket'], axis = 1)

# Caclulate deck number & replace NaN values with 'X' value
tmp = predictors.Cabin
tmp[tmp.isnull()] = "X"
tmp = tmp.str[0]
predictors['Deck'] = tmp

tmp = test_predictors.Cabin
tmp[tmp.isnull()] = "X"
tmp = tmp.str[0]
test_predictors['Deck'] = tmp


# Check for cabin
tmp = predictors.Cabin
tmp[tmp.isnull()] = ""
predictors['HasCabin'] = [len(re.findall(r"[0-9]+",s)) > 0 for s in tmp]


tmp = test_predictors.Cabin
tmp[tmp.isnull()] = ""
test_predictors['HasCabin'] = [len(re.findall(r"[0-9]+",s)) > 0 for s in tmp]

# Drop a cabin variable
predictors = predictors.drop(['Cabin'], axis = 1)
test_predictors = test_predictors.drop(['Cabin'], axis = 1)

# Replace NaN values of Embarked with Y value
tmp = test_predictors.Embarked
tmp[tmp.isnull()] = "Y"
predictors.Embarked = tmp

tmp = test_predictors.Embarked
tmp[tmp.isnull()] = "Y"
test_predictors.Embarked = tmp

# Create new variable HasFamily
predictors['HasFamily'] = (predictors.SibSp + predictors.Parch) > 0
test_predictors['HasFamily'] = (test_predictors.SibSp + test_predictors.Parch) > 0

# Imput missing values into numeric variables
num_columns = [col for col in predictors.columns if predictors[col].dtype in ['float64', 'int64'] ]
imputer = Imputer()
predictors[num_columns] = imputer.fit_transform(predictors[num_columns])
test_predictors[num_columns] = imputer.fit_transform(test_predictors[num_columns])

# Create new variable Young
predictors['Young'] = predictors.Age < 20
test_predictors['Young'] = test_predictors.Age < 20

#Create variable Rich
mean_fair = np.mean(predictors.Fare)
predictors['Rich'] = predictors.Fare > mean_fair
test_predictors['Rich'] = test_predictors.Fare > mean_fair

# Encode categorical variables
cat_columns = [col for col in predictors.columns if predictors[col].dtype == "object"] 
dummies = pd.get_dummies(predictors[cat_columns])
predictors = predictors.drop(cat_columns, axis = 1)
predictors = pd.concat([predictors, dummies], axis = 1)

dummies = pd.get_dummies(test_predictors[cat_columns])
test_predictors = test_predictors.drop(cat_columns, axis = 1)
test_predictors = pd.concat([test_predictors, dummies], axis = 1)

test_predictors["Deck_T"] = 0

print(test_predictors)

#encoder = OneHotEncoder()
#predictors = encoder.fit_transform(predictors)
#test_predictors = encoder.transform(test_predictors)

print(predictors.columns)
print(test_predictors.columns)


## Split predictors and target variables into training and validation datasets

train_x, test_x, train_y, test_y = train_test_split(predictors, y, random_state = 0)

## MODELING
print("\n\nGBC")
max_acc = 0
for i in range(300,460,20):
    (acc, mae, prediction, model) = model_GBC(train_x, train_y, test_x, test_y, i)
    if acc > max_acc: 
        gbs_model = model
        max_acc = acc
    print("GBC Est =%d\t Acc = %f\t MAE = %f"%(i, acc, mae))

print("GBC selected model with NEst = %d"%(gbs_model.n_estimators))
x = make_prediction(gbs_model, test_predictors, test_ids)
print(x)


# (acc, mae, prediction, model) = model_CVS(train_x, train_y, test_x, test_y, i)
# print("CVS Acc = %f\t MAE = %f"%(acc, mae))

#(acc, mae, prediction, model) = model_SGD(train_x, train_y, test_x, test_y, i)
#print("CGD Acc = %f\t MAE = %f"%(acc, mae))


#print("\n\nRandom Forest")
#for i in range(50,500,50):
#    (acc, mae, prediction, model) = model_RandomForest(train_x, train_y, test_x, test_y, i)
#    print("RF Est =%d\t Acc = %f\t MAE = %f"%(i, acc, mae))

#feature_selector = SelectFromModel(gbs_model, prefit=True)
#var_predictors = feature_selector.transform(predictors)
#print(var_predictors.head())

bad_cols = [ col for col in predictors.columns if np.var(predictors[col]) < .2 ]
print(bad_cols)

new_train = train_x.drop(bad_cols, axis = 1)
new_valid = test_x.drop(bad_cols, axis = 1)
new_test = test_predictors.drop(bad_cols, axis = 1)


print("\n\nGBC after removing")
max_acc = 0
for i in range(100,500,20):
    (acc, mae, prediction, model) = model_GBC(new_train, train_y, new_valid, test_y, i)
    if acc > max_acc: 
        gbs_model = model
        max_acc = acc
    print("GBC Est =%d\t Acc = %f\t MAE = %f"%(i, acc, mae))

print("GBC selected model with NEst = %d"%(gbs_model.n_estimators))
y = make_prediction(gbs_model, new_test, test_ids)
print(y)

print(all(x == y))

print("\n\nXGBoost")
maes = []
for i in range(100,500,20):
    (acc, mae, prediction, model) = model_XGBoost(train_x, train_y, test_x, test_y, i)
    print("XGBoost Est =%d\t Acc = %f\t MAE = %f"%(i, acc, mae))
    maes.append([i, mae, acc])
    
maes = np.array(maes)
maes = maes[ maes[:,1] == min(maes[:,1]) ]
maes = maes[ maes[:,0] == min(maes[:,0]) ]

(mae, predictions) = model_XGBoost(train_x, train_y, test_x, test_y, maes[0,0])

print("Using Gradient Boost with %d steps:\t MAE =\t %d" %(maes[0,0],mae))