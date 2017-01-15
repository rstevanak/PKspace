import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import loading
import visual

def knn(folder,neighbours=1):
    dataset, answers = loading.loadfrom(folder)
    trainsize = int(np.ceil(len(dataset)/10))
    #np.random.seed(0)
    indices = np.random.permutation(len(dataset))
    dataset_train = dataset[indices[:trainsize]]
    answers_train = answers[indices[:trainsize]]
    dataset_test = dataset[indices[:-trainsize]]
    answers_test = answers[indices[:-trainsize]]

    if (__name__ == '__main__'):print("Training")
    knn = KNeighborsClassifier(n_neighbors=neighbours).fit(dataset_train, answers_train)
    if (__name__ == '__main__'):print("Predicting")
    answers_predicted = knn.predict(dataset_test)

    truepositive = 0
    falsepositive,falsenegative=0,0
    for i in range(len(answers_test)):
        if answers_test[i] == answers_predicted[i]:
            truepositive += 1
        elif answers_predicted[i]=='1':
            falsepositive+=1
        else:
            falsenegative+=1
    accuracy=round(truepositive / len(answers_test),5)
    precision=round(truepositive/(truepositive+falsepositive),5)
    recall=round(truepositive/(truepositive+falsenegative),5)
    F1=round(2*(precision*recall)/(precision+recall),5)
    if __name__== '__main__':
        print("{3} KNN There is {0:.3f}% accuracy with {1} samples trained on {2} samples".
              format(truepositive / len(answers_test) * 100, len(answers_test), trainsize,folder))
        print("Precision: {} Recall: {}".format(precision,recall))
        print("F1 is {}".format(2*(precision*recall)/(precision+recall)))

        allpredicted = dict(zip(indices[:-trainsize], answers_predicted))
        visual.visualizefrom(folder, allpredicted)
    return accuracy,precision,recall,F1
if __name__== '__main__':
    knn("UFPR05")

