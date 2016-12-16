import numpy as np
from sklearn.neural_network import MLPClassifier
import nacitanie
import visual
folder="dataset"
trainsize = 300
dataset, answers = nacitanie.loadfrom(folder)
np.random.seed(0)
indices = np.random.permutation(len(dataset))
dataset_train = dataset[indices[:trainsize]]
answers_train = answers[indices[:trainsize]]
dataset_test = dataset[indices[:-trainsize]]
answers_test = answers[indices[:-trainsize]]

print("Training")
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15,), random_state=1)
clf.fit(dataset_train, answers_train)
answers_predicted = clf.predict(dataset_test)
rightans = 0
for i in range(len(answers_test)):
    if answers_test[i] == answers_predicted[i]:
        rightans += 1
print("There is {0:.3f}% accuracy with {1} samples trained on {2} samples".
      format(rightans / len(answers_test) * 100, len(answers_test), trainsize))
allpredicted=dict(zip(indices[:-trainsize], answers_predicted))
visual.visualizefrom(folder,allpredicted)
