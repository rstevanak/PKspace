import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import nacitanie
from sklearn.model_selection import KFold

dataset, answers = nacitanie.loadfrom("meh")
folds=5
rightstat=0
allstat=0
for train,test in KFold(folds).split(dataset):
    knn = KNeighborsClassifier().fit(dataset[train], answers[train])
    answers_predicted = knn.predict(dataset[test])
    for i in range(len(answers_predicted)):
        allstat+=1
        if answers[test][i] == answers_predicted[i]:
            rightstat += 1
print("There is {}% accuracy with {} fold cross-validation from dataset {} samples big".
              format(rightstat / allstat * 100, folds, len(dataset)))



