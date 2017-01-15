import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import loading
from sklearn.model_selection import KFold

for folder in ("UFPR04","UFPR05","PUCPRmini"):
    dataset, answers = loading.loadfrom(folder)
    folds=3
    for neighbours in range(1,11):
        rightstat = 0
        allstat = 0
        for test,train in KFold(folds).split(dataset):
            knn = KNeighborsClassifier(n_neighbors=neighbours).fit(dataset[train], answers[train])
            answers_predicted = knn.predict(dataset[test])
            for i in range(len(answers_predicted)):
                allstat+=1
                if answers[test][i] == answers_predicted[i]:
                    rightstat += 1
        print("{} - There is {}% accuracy with {} fold cross-validation from dataset {} samples big with {} neighbours".
                      format(folder,rightstat / allstat * 100, folds, len(dataset),neighbours))



