import pandas as pd
import re
import math
import numpy as np
from sklearn.preprocessing import Imputer, OneHotEncoder
from sklearn.metrics import mean_absolute_error
from xgboost import XGBClassifier


def model_XGBoost(train_x, train_y, stop_rounds = 5, depth = 6, rate = 0.02):
    model = XGBClassifier(max_depth=depth, n_estimators=500, learning_rate=rate)
    model.fit(train_x, train_y, early_stopping_rounds = stop_rounds, eval_set=[(train_x, train_y)], verbose=False)
    sc = model.score(train_x, train_y)    
    prediction = model.predict(train_x)
    mae = mean_absolute_error(train_y, prediction)
    return (sc, mae, prediction, model)

def make_prediction(model, x):
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

print("\n\nXGBoost")
max_acc = 0

training_log = pd.DataFrame(data={'Stops':[0], 'Depth':[0], 'Rate':[0], 'Acc':[0], 'MAE':[0]})

rows = []
print("Model tuning and selection...")
for i in range(100,510,10):
    for j in range(20,31):
        for r in np.arange(.01, .1, .01):
            (acc, mae, prediction, model) = model_XGBoost(predictors, y, i,j,r)
            # print("XGBoost Stop =%d\t Depth=%d\t Rate=%f\t Acc = %f\t MAE = %f"%(i, j, r, acc, mae))
            dict = {}
            dict.update({'Stops':i, 'Depth':j, 'Rate':r, 'Acc': acc, 'MAE': mae})
            rows.append(dict)
            if (acc > max_acc):
                max_acc = acc
                xgb_model = model

training_log = pd.DataFrame(rows)
training_log.to_csv("training_table.csv")
    
print("Prediction process...")
test_predictors = test_predictors[predictors.columns]
predictions = make_prediction(xgb_model, test_predictors)

print("Submition preparation...")
submission = pd.DataFrame({ 'PassengerId': test_ids,
                            'Survived': predictions })
submission.to_csv("submission2.csv", index=False)

print("That's all, folks!")


