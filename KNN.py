import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import nacitanie
import visual

dataset, answers = nacitanie.loadfrommeh()
testsize = 130
indices = np.random.permutation(len(dataset))
dataset_train = dataset[indices[:-testsize]]
answers_train = answers[indices[:-testsize]]
dataset_test = dataset[indices[-testsize:]]
answers_test = answers[indices[-testsize:]]

print("Trenujem")
knn = KNeighborsClassifier().fit(dataset_train, answers_train)
answers_predicted = knn.predict(dataset_test)
print("spravne " + str(answers_test))
print("hadane  " + str(answers_predicted))
rightans = 0
for i in range(testsize):
    if answers_test[i] == answers_predicted[i]:
        rightans += 1
print("There is {}% accuracy with {} samples from dataset {} samples big".
      format(rightans / testsize * 100, testsize, len(indices) - testsize))
allpredicted=dict(zip(indices[-testsize:], answers_predicted))
visual.visualizefrommeh(allpredicted)