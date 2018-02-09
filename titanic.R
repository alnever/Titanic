library(dplyr)
library(caret)
library(randomForest)
library(e1071)
library(rpart.plot)
library(rattle)
library(gbm)

#Set seed
set.seed(753)


# Load training and test datasets
training <- read.csv("train.csv", header = TRUE)
testing  <- read.csv("test.csv", header = TRUE)

summary(training)
summary(testing)

# Remove some variables
# training <- training %>% 
#   select(-Name, -Ticket, -PassengerId) %>%
#   mutate(Board = as.factor(substr(as.character(Cabin), 0, 1)),
#          Survived = as.factor(Survived),
#          Pclass = as.factor(Pclass),
#          SibSp = as.factor(SibSp),
#          Parch = as.factor(Parch),
#          AgeGroup = cut(Age, breaks = 5)
#          ) %>%
#   select(-Cabin, -Age)
# 
# 
# levels(training$AgeGroup) <- c( levels(training$AgeGroup), "Missing")
# training$AgeGroup[is.na(training$AgeGroup)] <- "Missing"
# 

# Remove some variables
training <- training %>% 
  select(-Name, -Ticket, -PassengerId) %>%
  mutate(Board = as.factor(substr(as.character(Cabin), 0, 1)),
         Survived = as.factor(Survived),
         AgeGroup = cut(Age, breaks = 5)
  ) %>%
  select(-Cabin, -Age)

levels(training$AgeGroup) <- c( levels(training$AgeGroup), "Missing")
training$AgeGroup[is.na(training$AgeGroup)] <- "Missing"

x <- dummyVars( ~ Sex + Embarked + Board + AgeGroup, training)

g <- glm(Survived ~ ., training, family = "binomial")
summary(g)


summary(training)

# Split training dataset into training and validation datasets
inTrain <- createDataPartition(training$Survived, p = .6, list = FALSE)

validation <- training[-inTrain,]
training   <- training[inTrain,]

# RPart model
rpartModel <- train(Survived ~ .,method="rpart", data=training)

predictValidate1 <- predict(rpartModel,newdata = validation)
cm1 <- confusionMatrix(predictValidate1, validation$Survived)

fancyRpartPlot(rpartModel$finalModel, sub = "Prediction tree")


# Try Random Forest Model
rfModel <- train(Survived ~ .,method="rf", data=training)

predicted <- predict(rfModel,newdata = validation)
confMatrix <- confusionMatrix(predicted, validation$Survived)
confMatrix

# Random search

metric <- "Accuracy"
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="random")
mtry <- sqrt(ncol(training))
tunegrid <- expand.grid(.mtry=mtry)
rf_random <- train(Survived~., data=training, method="rf", metric=metric, tuneLength=15, trControl=control)
print(rf_random)
plot(rf_random)

# Grid search

control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
tunegrid <- expand.grid(.mtry=c(1:15))
rf_gridsearch <- train(Survived~., data=training, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_gridsearch)
plot(rf_gridsearch)

# TuneRF
bestmtry <- tuneRF(training[,-1], training[,1], stepFactor=1.5, improve=1e-5, ntree=500)
print(bestmtry)




# provide additionnal parameters to GBM

fitControl <- trainControl(method = "cv", number = 4)
gbmGrid <-  expand.grid(interaction.depth = 1:3,
                        n.trees = (1:4)*100,
                        shrinkage = 0.1,
                        n.minobsinnode = 20)

cvModel <- train(Survived ~. ,data = training ,method="gbm" ,trControl = fitControl ,verbose = FALSE,tuneGrid = gbmGrid)
print(cvModel)
plot(cvModel)
predictValidate4 <- predict(cvModel,newdata = validation)
cm4 <- confusionMatrix(predictValidate4, validation$Survived)
cm4
