import loading
from sklearn.neural_network import MLPClassifier
import pickle
import os


def makemodel(folder,hiddenlayer=(15,10)):
    outputname = "MLP" + str(hiddenlayer) + folder
    trainingdata,traininganswers=loading.loadfrom(folder)
    clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=hiddenlayer, random_state=1)
    print("Training")
    clf.fit(trainingdata,traininganswers)
    print("Dumping")
    pickle.dump(clf,open(os.path.join(os.path.curdir,folder,outputname+".pkl"),"wb"))
if __name__== '__main__':
    makemodel("UFPR05")