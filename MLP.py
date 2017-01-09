import numpy as np
from sklearn.neural_network import MLPClassifier
import loading
import visual

folder = "UFPR04"

#np.random.seed(0)
dataset, answers = loading.loadfrom(folder)
trainsize = int(np.ceil(len(dataset)/10))

indices = np.random.permutation(len(dataset))
dataset_train = dataset[indices[:trainsize]]
answers_train = answers[indices[:trainsize]]
dataset_test = dataset[indices[:-trainsize]]
answers_test = answers[indices[:-trainsize]]

print("Training")
clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(15,), random_state=1)
clf.fit(dataset_train, answers_train)
print("Predicting")
answers_predicted = clf.predict(dataset_test)

truepositive = 0
falsepositive,falsenegative=0,0
for i in range(len(answers_test)):
    if answers_test[i] == answers_predicted[i]:
        truepositive += 1
    elif answers_predicted[i]=='1':
        falsepositive+=1
    else:
        falsenegative+=1
precision=truepositive/(truepositive+falsepositive)
recall=truepositive/(truepositive+falsenegative)

print("{3} MLP There is {0:.3f}% accuracy with {1} samples trained on {2} samples".
      format(truepositive / len(answers_test) * 100, len(answers_test), trainsize,folder))
print("Precision: {} Recall: {}".format(precision,recall))
print("F1 is {}".format(2*(precision*recall)/(precision+recall)))
allpredicted = dict(zip(indices[:-trainsize], answers_predicted))
visual.visualizefrom(folder, allpredicted)
