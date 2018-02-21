import pandas as pd
import re
import math
import numpy as np
from sklearn.preprocessing import Imputer, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier


def predict_GBC(train_x, train_y, test_x, n_est = 340):
    model = GradientBoostingClassifier(n_estimators = n_est, learning_rate = .05)
    model.fit(train_x, train_y)
    prediction = model.predict(test_x)
    return prediction

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

predictions = predict_GBC(predictors, y, test_predictors, 340)

submission = pd.DataFrame({ 'PassengerId': test_ids,
                            'Survived': predictions })
submission.to_csv("submission.csv", index=False)

