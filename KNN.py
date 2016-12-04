import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import nacitanie
import visual
from sklearn.model_selection import KFold


# Split iris data in train and test data
# A random permutation, to split the data randomly
dataset, answers = nacitanie.loadfrommeh()
np.random.seed(42)
# testsize = 130
# indices = np.random.permutation(len(dataset))
# dataset_train = dataset[indices[:-testsize]]
# answers_train = answers[indices[:-testsize]]
# dataset_test = dataset[indices[-testsize:]]
# answers_test = answers[indices[-testsize:]]
folds=5
rightstat=0
allstat=0
for train,test in KFold(3).split(dataset):
    knn = KNeighborsClassifier().fit(dataset[train], answers[train])
    answers_predicted = knn.predict(dataset[test])
    for i in range(len(answers_predicted)):
        allstat+=1
        if answers[test][i] == answers_predicted[i]:
            rightstat += 1
print("There is {}% accuracy with {} fold cross-validation from dataset {} samples big".
              format(rightstat / allstat * 100, folds, len(dataset)))


# Create and fit a nearest-neighbor classifier
# print("Trenujem")
# knn = KNeighborsClassifier().fit(dataset_train, answers_train)
# answers_predicted = knn.predict(dataset_test)
# print("spravne " + str(answers_test))
# print("hadane  " + str(answers_predicted))
# rightans = 0
# for i in range(testsize):
#     if answers_test[i] == answers_predicted[i]:
#         rightans += 1
# print("There is {}% accuracy with {} samples from dataset {} samples big".
#       format(rightans / testsize * 100, testsize, len(indices) - testsize))
# allpredicted=dict(zip(indices[-testsize:], answers_predicted))
#visual.visualizefrommeh(allpredicted)
