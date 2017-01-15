import KNN
import operator
values={}
for folder in ("UFPR04","UFPR05","PUCPRmini"):
    print("{} - solver='lbfgs',random state=1 , hidden layer,accuracy,precision,recall,f1".format(folder))
    for i in range(1,11):
        result=KNN.knn(folder,i)
        print(i,result)
        values[i]=values.get(i,0)+result[3]
sorted_values = sorted(values.items(), key=operator.itemgetter(1))
print(sorted_values)

