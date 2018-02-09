training <- read.csv("train.csv", header = TRUE)

plot(training)

head(training$Ticket)


# Remove some variables
training <- training %>% 
  select(-Name, -Ticket, -PassengerId) %>%
  mutate(Board = as.factor(substr(as.character(Cabin), 0, 1)),
         Survived = as.factor(Survived),
         Pclass = as.factor(Pclass),
         SibSp = as.factor(SibSp),
         Parch = as.factor(Parch),
         AgeGroup = cut(Age, breaks = 5)
  ) %>%
  select(-Cabin, -Age)


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

g <- glm(Survived ~ ., training, family = "binomial")
summary(g)


# training <- training %>% select(Survived, Pclass, SibSp, Board, AgeGroup)


plot(training)


inTrain <- createDataPartition(training$Survived, p = .6, list = FALSE)

validation <- training[-inTrain,]
training   <- training[inTrain,]


# Try Random Forest Model
rfModel <- train(Survived ~ .,method="rf", data=training)

predicted <- predict(rfModel,newdata = validation)
confMatrix <- confusionMatrix(predicted, validation$Survived)
confMatrix
