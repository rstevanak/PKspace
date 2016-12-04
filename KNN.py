import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import nacitanie
import visual

testsize=10
# Split iris data in train and test data
# A random permutation, to split the data randomly
dataset, answers = nacitanie.loadfrommeh()
np.random.seed(42)
indices = np.random.permutation(len(dataset))
dataset_train = dataset[indices[:-testsize]]
answers_train = answers[indices[:-testsize]]
dataset_test = dataset[indices[-testsize:]]
answers_test = answers[indices[-testsize:]]
# Create and fit a nearest-neighbor classifier
print("Trenujem")
knn = KNeighborsClassifier()
knn.fit(dataset_train, answers_train)
answers_predicted = knn.predict(dataset_test)
print("spravne "+str(answers_test))
print("hadane  "+str(answers_predicted))
allpredicted = {}
for i in range(testsize):
    allpredicted[indices[-i-1]] = answers_predicted[-i-1]
visual.visualizefrommeh(allpredicted)
