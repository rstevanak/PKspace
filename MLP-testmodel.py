from sklearn.neural_network import MLPClassifier
import pickle
import os
import loading
import visual

def testmodel(folder):
    clf=pickle.load(open(os.path.join(os.path.curdir,folder,"MLP.pkl"),"rb"))
    testdata,real_answers=loading.loadfrom(folder)
    print("Predicting")
    predicted_answers=clf.predict(testdata)
    if real_answers!=None:
        truepositive,correct = 0,0
        falsepositive, falsenegative = 0, 0
        for i in range(len(real_answers)):
            if real_answers[i] == predicted_answers[i]:
                correct += 1
                if real_answers[i]=='1':
                    truepositive+=1
            elif real_answers[i] == '1':
                falsepositive += 1
            else:
                falsenegative += 1
        accuracy = round(correct / len(real_answers), 5)
        precision = round(truepositive / (truepositive + falsepositive), 5)
        recall = round(truepositive / (truepositive + falsenegative), 5)
        F1 = round(2 * (precision * recall) / (precision + recall), 5)
        print("{1} MLP There is {0:.3f}% accuracy on {2} samples".
              format(accuracy * 100, folder,len(real_answers)))
        print("Precision: {} Recall: {}".format(precision, recall))
        print("F1 is {}".format(F1))
        print("There is {} correctly classified, {} false positives and {} false negatives".format(correct,falsepositive,falsenegative))
    allpredicted = dict(zip(list(range(len(predicted_answers))), predicted_answers))
    visual.visualizefrom(folder,allpredicted)
if __name__== '__main__':
    testmodel("UFPR04")