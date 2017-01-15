import MLP
import operator
values={}
for folder in ("UFPR04","UFPR05","PUCPRmini"):
    print("{} - solver='lbfgs',random state=1 , hidden layer,accuracy,precision,recall,f1".format(folder))
    for i in range(5,26,5):
        layer=(i,)
        result=MLP.mlp(folder,layer)
        print(layer,result)
        values[layer]=values.get(layer,0)+result[3]
    print()
    for i in range(5,26,5):
        for j in range(5,26,5):
            layer=(i,j)
            result = MLP.mlp(folder, layer)
            print(layer, result)
            values[layer] = values.get(layer, 0) + result[3]
        print()
    print("**************************************************************************")
sorted_values = sorted(values.items(), key=operator.itemgetter(1))
print(sorted_values)

