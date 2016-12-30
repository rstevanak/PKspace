import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import loading
import visual

folder="UFPR04"
dataset, answers = loading.loadfrom(folder)
trainsize = int(np.ceil(len(dataset)/10))
#np.random.seed(0)
indices = np.random.permutation(len(dataset))
dataset_train = dataset[indices[:trainsize]]
answers_train = answers[indices[:trainsize]]
dataset_test = dataset[indices[:-trainsize]]
answers_test = answers[indices[:-trainsize]]

print("Training")
knn = KNeighborsClassifier().fit(dataset_train, answers_train)
print("Predicting")
answers_predicted = knn.predict(dataset_test)

truepositive = 0
for i in range(len(answers_test)):
    if answers_test[i] == answers_predicted[i]:
        truepositive += 1
falsepositive=np.count_nonzero(answers_predicted)-truepositive
falsenegative=np.count_nonzero(answers_test)-truepositive
precision=truepositive/(truepositive+falsepositive)
recall=truepositive/(truepositive+falsenegative)

print("{3} KNN There is {0:.3f}% accuracy with {1} samples trained on {2} samples".
      format(truepositive / len(answers_test) * 100, len(answers_test), trainsize,folder))
print("Precision: {} Recall: {}".format(precision,recall))
print("F1 is {}".format(2*(precision*recall)/(precision+recall)))

allpredicted = dict(zip(indices[:-trainsize], answers_predicted))
visual.visualizefrom(folder, allpredicted)
