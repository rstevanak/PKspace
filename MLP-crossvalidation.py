import numpy as np
from sklearn.neural_network import MLPClassifier
import loading
from sklearn.model_selection import KFold

for folder in ("UFPR04","UFPR05","PUCPRmini"):
    dataset, answers = loading.loadfrom(folder)
    folds=3
    rightstat=0
    allstat=0
    for hiddenlay1 in range(5,25,5):
        for test,train in KFold(folds).split(dataset):
            mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(hiddenlay1,)).fit(dataset[train], answers[train])
            answers_predicted = mlp.predict(dataset[test])
            for i in range(len(answers_predicted)):
                allstat+=1
                if answers[test][i] == answers_predicted[i]:
                    rightstat += 1
        print("{} - solver='lbfgs', hidden layer=({},), accuracy {}".
              format(folder, hiddenlay1, rightstat / allstat, ))
    print()
    print("**************************************************************************")



