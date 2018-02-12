library(dplyr)
library(ggplot2)

data <- read.csv("train.csv", header = TRUE)

# Number of survived and non-survived
g <- ggplot(data, aes(as.factor(Survived)))
g <- g + geom_bar(stat = "count")
g


# Dependence of the surviving on the sex of passengers
g <- ggplot(data, aes(as.factor(Survived)))
g <- g + geom_bar(stat = "count", aes(fill = Sex), position = position_dodge())
g
# --> The most of survived are women, the most of non-survived are men

# Dependence of the surviving on the board's class

data <- data %>%
  mutate(Deck = as.factor(substr(as.character(Cabin), 0, 1)))
head(data)


g <- ggplot(data, aes(as.factor(Deck)))
g <- g + geom_bar(stat = "count", aes(fill = as.factor(Survived)), position = position_dodge())
g <- g + ylim(0,100)
g
# --> The majority of survived had tickets on decks B, C, D, E 

g <- ggplot(data, aes(as.factor(Deck)))
g <- g + geom_bar(stat = "count", aes(fill = as.factor(Survived)), position = position_dodge())
g
# --> The majority of non-survived had tickets on decks A, G, and T, or the deck is unknown

# Proportions of survived to non-survived for each deck
temp <- data %>%
  mutate(NonSurvived = Survived == 0) %>%
  mutate(Survived = Survived == 1) %>%
  group_by(Deck) %>%
  summarise(TotalSurvived = sum(Survived), TotalNonSurvived = sum(NonSurvived) ) %>%
  mutate(Ratio = TotalSurvived / TotalNonSurvived)

g <- ggplot(temp, aes(Deck, Ratio))
g <- g + geom_bar( stat = "identity", aes(fill = Ratio > 1) )
g
# --> Passengers of decks B, C, D, E, and F had the higher chances to survive, than passengers of other decks

# What about cabins?
data <- data %>%
  mutate(HasCabin = grepl("[0-9]+", Cabin))

g <- ggplot(data, aes(as.factor(HasCabin)))
g <- g + geom_bar(stat = "count", aes(fill = as.factor(Survived)), position = position_dodge())
g
# --> The majority of survived had a cabin

# Age
g <- ggplot(data, aes(as.factor(Survived), Age))
g <- g + geom_boxplot(aes(colour = as.factor(Survived)))
g  
# --> There isn't a significant differene in ages of survived and non-sirvived

fit <- lm(Survived ~ Age - 1, data)
summary(fit)
plot(fit)


fit <- glm(Survived ~ Age - 1, data, family = "binomial")
summary(fit)
plot(fit)

# And what about age's groups
data <- data %>%
  mutate(AgeGroup = cut(Age, breaks = 5))

summary(data$AgeGroup)
levels(data$AgeGroup) <- c( levels(data$AgeGroup), "Missing")
data$AgeGroup[is.na(data$AgeGroup)] <- "Missing"

temp <- data %>%
  mutate(NonSurvived = Survived == 0) %>%
  mutate(Survived = Survived == 1) %>%
  group_by(AgeGroup) %>%
  summarise(TotalSurvived = sum(Survived), TotalNonSurvived = sum(NonSurvived) ) %>%
  mutate(Ratio = TotalSurvived / TotalNonSurvived)

g <- ggplot(temp, aes(AgeGroup, Ratio))
g <- g + geom_bar( stat = "identity", aes(fill = Ratio > 1) )
g
# --> The group of  youngest passengers has the highest surviving ratio

# Dependence of the surviving on the passengers class
g <- ggplot(data, aes(as.factor(Pclass)))
g <- g + geom_bar(stat = "count", aes(fill = as.factor(Survived)), position = position_dodge())
g
# --> 1st class - good, 2nd and 3rd - bad chances, especialy the 3rd

# Two strange parameters - SibSp, Parch. Both of them are connected to the family condition of passengers
g <- ggplot(data, aes(SibSp, Parch))
g <- g + geom_jitter(aes(color = as.factor(Survived)))
g
# --> High values of these parameters are rare, but if they're observed then surviving chances are low

# The 0-0 combination is unobvious
temp <- data %>%
  filter(SibSp == 0 & Parch == 0)

g <- ggplot(data, aes(as.factor(Survived)))
g <- g + geom_bar(stat = "count")
g
# --> The most didn't survived

# What are other charecterics of this group of passengers. May be an age?
g <- ggplot(data, aes(Age))
g <- g + geom_density()
g

data <- data %>%
  mutate(Family = SibSp + Parch, IsFamily = Family > 0)

g <- ggplot(data, aes(AgeGroup))
g <- g + geom_bar(stat = "count", aes(fill = IsFamily))
g
# --> The chances to survive are higher in every age group, when the passenger wasn't alone on the board
g <- ggplot(data, aes(Sex))
g <- g + geom_bar(stat = "count", aes(fill = IsFamily))
g
# --> It has matter for different sex groups too