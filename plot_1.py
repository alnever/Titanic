# -*- coding: utf-8 -*-
"""
Редактор Spyder

Это временный скриптовый файл.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sea

default_colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

"""
Barplot function
    
"""
def plot_bar(df, x, y, group, 
             xlabs = [], grouplabs = [],
             groupcolors = [],
             title = "", xlab = "", ylab = ""):
    fig, ax = plt.subplots()
    bar_count    = df[group].unique().size
    labels       = df[group].unique().tolist()
    width = 1/bar_count - .01
    for g in df[group].unique():
        if (bar_count % 2 != 0):
            bar_offset = labels.index(g) - bar_count // 2
        else:
            bar_offset = labels.index(g) - bar_count + 3/2
        ax.bar(list(df.loc[df[group] == g][x] + bar_offset * width) if (str(df[x].dtype) != "category") else 
                   list(pd.factorize(df.loc[df[group] == g][x])[0] + bar_offset * width),
               df.loc[df[group] == g][y].tolist(),
               width = width,
               label = str(g) if grouplabs == [] else str(grouplabs[labels.index(g)]),
               color = groupcolors[labels.index(g)] if groupcolors != [] else default_colors[labels.index(g)] 
              )  
    ax.set_xticks(df[x].unique() if str(df[x].dtype) != 'category' else pd.factorize(df[x].unique())[0])
    if (xlabs != []):
        ax.set_xticklabels(xlabs)
    else:        
        ax.set_xticklabels(df[x].unique())
    ax.legend(loc = "best")
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    return

data = pd.read_csv("train.csv")

print(data.head())
print(data.shape)

# Barplot Survived 0/1

tmp = {"Survived" : data.loc[data.Survived == 1].Survived.count(),
       "Dead"     : data.loc[data.Survived == 0].Survived.count()}

print(tmp)

var  = list(tmp.keys())
var1  = [1,0] 
val = list(tmp.values())


fig, ax = plt.subplots()
ax.barh(var1, val)
ax.set_yticks(var1)
ax.set_yticklabels(var)

# Barplot Sex male/female

tmp = {"Male"       : data.loc[data.Sex == "male"].Sex.count(),
       "Female"     : data.loc[data.Sex == "female"].Sex.count()}

print(tmp)

var  = list(tmp.keys())
var1  = [1,0] 
val = list(tmp.values())

fig, ax = plt.subplots()
ax.barh(var1, val)
ax.set_yticks(var1)
ax.set_yticklabels(var)

# Barplot: Survived by Sex
males = {"Survived": data.loc[(data.Survived == 1) & (data.Sex == "male")].Survived.count(),
         "Dead": data.loc[(data.Survived == 0) & (data.Sex == "male")].Survived.count()
         }

females = {"Survived": data.loc[(data.Survived == 1) & (data.Sex == "female")].Survived.count(),
           "Dead": data.loc[(data.Survived == 0) & (data.Sex == "female")].Survived.count()
         }

labels = list(males.keys())
labels_keys = [1,0]

fig, ax = plt.subplots()
ax.bar(list(np.array(labels_keys) - .15), males.values(), width = .3, color = "b", label = "males")
ax.bar(list(np.array(labels_keys) + .15), females.values(), width = .3, color = "r", label = "females")
ax.set_xticks(labels_keys)
ax.set_xticklabels(labels)
ax.legend(loc = "best")
plt.title("Survived by Sex")
plt.ylabel("Number of People")

# Barplot: Survived by Pclass

## Create a temporary data frame 
tmp = data.groupby(['Survived','Pclass']).PassengerId.count().reset_index()

### var 1
plot_bar(tmp, 'Survived', 'PassengerId', 'Pclass', 
             xlabs = ['Dead','Survived'], 
             title = "Survived by Passenger Class",
             ylab = "Number of People")


### var 2
plot_bar(tmp, 'Pclass', 'PassengerId', 'Survived', 
         grouplabs = ['Dead','Survived'],
         groupcolors = ['r','g'],
         title = "Survived by Passenger Class",
         ylab = "Number of People")

# Age analysis
## Age of survived and dead passengers
tmp = data[['Survived','Age']]
tmp = tmp.loc[~ tmp.Age.isnull()]

### var 1
tmp.boxplot(by = "Survived")
tmp = [tmp.loc[tmp.Survived == 0].Age.tolist(), tmp.loc[tmp.Survived == 1].Age.tolist()]

### var 2
fig, ax = plt.subplots()
boxes = plt.boxplot(tmp)
ax.set_xticklabels(['Dead','Survived'])
plt.title("Age of survived and dead passengers")
plt.ylabel("Age, years")

## Survived: age defined
tmp = data[['Survived', 'Age', 'PassengerId']]
tmp.insert(tmp.columns.size,'AgeSet',[0 if math.isnan(age) else 1 for age in tmp.Age], True)
tmp = tmp[['Survived','AgeSet', 'PassengerId']]

tmp = tmp.groupby(['Survived','AgeSet']).PassengerId.count().reset_index()

plot_bar(tmp, 'AgeSet', 'PassengerId', 'Survived', 
         xlabs = ['Age dont set','Age set'],
         grouplabs = ['Dead','Survived'],
         groupcolors = ['r','g'],
         title = "Survived with age set or not",
         ylab = "Number of People")

## Analysis of age's groups
tmp = data[['Survived', 'Age', 'PassengerId']]
tmp.is_copy = False
tmp.loc[tmp.Age.isnull(),'Age'] = np.mean(data.Age)
tmp.insert(tmp.shape[1],'AgeGroup',pd.qcut(tmp.Age, 5), True)
tmp = tmp.groupby(['Survived','AgeGroup']).PassengerId.count().reset_index()

### By groups
plot_bar(tmp, 'AgeGroup', 'PassengerId', 'Survived', 
         grouplabs = ['Dead','Survived'],
         groupcolors = ['r','g'],
         title = "Survived by age's groups",
         ylab = "Number of People")


### Survived-dead by age's groups Ratio calculation
ratio = [ [ tmp.AgeGroup.unique()[group], 
            list(tmp[(tmp.Survived == 1) & (tmp.AgeGroup == tmp.AgeGroup.unique()[group])].PassengerId)[0] / 
            list(tmp[(tmp.Survived == 0) & (tmp.AgeGroup == tmp.AgeGroup.unique()[group])].PassengerId)[0],
            list(tmp[(tmp.Survived == 1) & (tmp.AgeGroup == tmp.AgeGroup.unique()[group])].PassengerId)[0] / 
            (list(tmp[(tmp.Survived == 0) & (tmp.AgeGroup == tmp.AgeGroup.unique()[group])].PassengerId)[0] +
             list(tmp[(tmp.Survived == 1) & (tmp.AgeGroup == tmp.AgeGroup.unique()[group])].PassengerId)[0])
          ]
               for group in pd.factorize(tmp.AgeGroup.unique())[0]
        ]

ratio = np.array(ratio)    
    
fig, ax = plt.subplots()
ticks = [list(ratio[:,0]).index(x) for x in list(ratio[:,0])]
labels = [str(x) for x in ratio[:,0]]
plt.bar(np.array(ticks) - .15,ratio[:,1], color = ["g" if r > .7 else "r" for r in ratio[:,1]], width = .3, label = "Survived/Dead")    
plt.bar(np.array(ticks) + .15,ratio[:,2], color = ["b" if r > .5 else "m" for r in ratio[:,2]], width = .3, label = "Survived/Total")    
ax.set_xticks(ticks)
ax.set_xticklabels(labels)
ax.legend(loc = "best")
plt.title("Survived-dead ratio within age's groups")
plt.ylabel("Ratio")  

#Family size analysis
## Survived by SibSp
tmp = data.groupby(['Survived','SibSp']).PassengerId.count().reset_index()

plot_bar(tmp,'SibSp','PassengerId','Survived',
         grouplabs = ['Dead','Survived'],
         groupcolors = ['C1','C2'],
         title = "Survived by SibSp",
         ylab = "Number of People")

## Survived by Parch
tmp = data.groupby(['Survived','Parch']).PassengerId.count().reset_index()

plot_bar(tmp,'Parch','PassengerId','Survived',
         grouplabs = ['Dead','Survived'],
         groupcolors = ['C1','C2'],
         title = "Survived by Parch",
         ylab = "Number of People")

## Is the a covariation
tmp = data[['Survived', 'SibSp', 'Parch']]
plt.figure()
sea.stripplot(tmp.SibSp, tmp.Parch, hue = tmp.Survived, palette = 'Set1', jitter = True)


## Survived by FamilySize = SibSp + Parch + 1
tmp = data[['Survived', 'SibSp', 'Parch', 'PassengerId']]
tmp.insert(tmp.columns.size, 'FamilySize', [ tmp.SibSp[i] + tmp.Parch[i] + 1 for i in range(tmp.shape[0]) ])
tmp = tmp.groupby(['Survived','FamilySize']).PassengerId.count().reset_index()

plot_bar(tmp,'FamilySize','PassengerId','Survived',
         grouplabs = ['Dead','Survived'],
         groupcolors = ['C1','C2'],
         title = "Survived by FamilySize",
         ylab = "Number of People")

sea.barplot(x = tmp.FamilySize, y = tmp.PassengerId, hue = tmp.Survived, palette = 'Set1')

## FamilySize and Age
tmp = data[['Survived', 'SibSp', 'Parch', 'PassengerId', 'Age']]
tmp.is_copy = False
tmp.insert(tmp.columns.size, 'FamilySize', [ tmp.SibSp[i] + tmp.Parch[i] + 1 for i in range(tmp.shape[0]) ])
tmp.loc[tmp.Age.isnull(),'Age'] = np.mean(data.Age)

plt.figure()
sea.stripplot(tmp.FamilySize, tmp.Age, hue = tmp.Survived, jitter = True, palette = 'Set1')

## Ticket
tmp = data[['Ticket','Survived','PassengerId']]
tmp.is_copy = False
tmp.insert(tmp.columns.size,'HasTicket',[s != "" for s in tmp.Ticket])
tmp = tmp.groupby(['Survived','HasTicket']).PassengerId.count().reset_index()

## Cabin
tmp = data[['Cabin','Survived','PassengerId']]
tmp.is_copy = False
tmp.insert(tmp.columns.size,'HasCabin',[str(s) != 'nan' for s in tmp.Cabin])
tmp = tmp.groupby(['Survived','HasCabin']).PassengerId.count().reset_index()

plt.figure()
sea.barplot(x = tmp.HasCabin, y = tmp.PassengerId, hue = tmp.Survived, palette = 'Set1')

## Cabin
tmp = data[['Cabin','Survived','PassengerId']]
tmp.is_copy = False
tmp.insert(tmp.columns.size,'Deck',[str(s)[0] if str(s) != 'nan' else 'X' for s in tmp.Cabin])
tmp = tmp.groupby(['Survived','Deck']).PassengerId.count().reset_index()

plt.figure()
sea.barplot(x = tmp.Deck, y = tmp.PassengerId, hue = tmp.Survived, palette = 'Set1')

## Cabin
tmp = data[['Pclass','Cabin','Survived']]
tmp.is_copy = False
tmp.insert(tmp.columns.size,'Deck',[str(s)[0] if str(s) != 'nan' else 'X' for s in tmp.Cabin])
tmp = tmp.groupby(['Pclass','Deck']).apply(lambda tmp: tmp[tmp.Survived == 1].Survived.count()).reset_index()
tmp = tmp.rename(index = str, columns =  {0: 'Survived'})

plt.figure()
sea.set_style("whitegrid")
sea.stripplot(x = tmp.Deck, y = tmp.Pclass, hue = tmp.Survived, size = tmp.Survived)


tmp = data[['Pclass','Cabin','Survived']]
tmp.is_copy = False
tmp.insert(tmp.columns.size,'Deck',[str(s)[0] if str(s) != 'nan' else 'X' for s in tmp.Cabin])
tmp = tmp.groupby(['Pclass','Deck']).apply(lambda tmp: tmp[tmp.Survived == 1].Survived.count() / tmp[tmp.Survived == 0].Survived.count()).reset_index()
tmp = tmp.rename(index = str, columns =  {0: 'Ratio'})

plt.figure()
sea.set_style("whitegrid")
ax = sea.stripplot(x = tmp.Deck, y = tmp.Pclass, hue = tmp.Ratio, size = tmp.Ratio * 10)
ax.legend_.remove()
ax.set_yticks(tmp.Pclass.unique())
plt.show()


tmp = data[['Pclass','Cabin','Survived']]
tmp.is_copy = False
tmp.insert(tmp.columns.size,'Deck',[str(s)[0] if str(s) != 'nan' else 'X' for s in tmp.Cabin])
tmp = tmp.groupby(['Pclass','Deck']).apply(lambda tmp: tmp[tmp.Survived == 1].Survived.count() / tmp.Survived.count() ).reset_index()
tmp = tmp.rename(index = str, columns =  {0: 'Ratio'})

sea.set_style("whitegrid")
ax = sea.stripplot(x = tmp.Deck, y = tmp.Pclass, hue = tmp.Ratio, size = tmp.Ratio * 50)
ax.legend_.remove()
ax.set_yticks(tmp.Pclass.unique())
plt.show()







