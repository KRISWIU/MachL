from asyncio import start_unix_server
import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from nn import nn
from kfoldCrossV import kfold,evaluate

structure = [1,2,1]
nnetwork = nn(structure,0)
nnetwork.benchmark1()

structure = [2,4,3,2]
nnetwork = nn(structure,0.25)
nnetwork.benchmark2()

structure1 = [16,10,6,3,2]
structure2 = [16,10,8,4,2]
structure3 = [16,10,8,2]
structure4 = [16,8,4,2]
structure5 = [16,10,2]
structure6 = [16,8,2]

datasets = np.array(pandas.read_csv('datasets/hw3_house_votes_84.csv'))
x = datasets[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]
y = datasets[:,16]
#network = nn(structure,constant,lamda)

ynew = []
for i in range(len(y)):
    if y[i] == 0:
        ynew.append([0,1])
    elif y[i] == 1:
        ynew.append([1,0])

#x_trains, y_trains, x_test, y_test = kfold(x,ynew,10)

def kfoldTrain(X,Y,k,structure,lamda):
    x_trains, y_trains, x_test, y_test = kfold(X,Y,10)
    yvoting = []
    costJ = 0
    for i in range(k):
        x = x_trains[i]
        y = y_trains[i]
        nn1 = nn(structure,lamda)
        nn1.train(x,y)
        ypre = nn1.predict(X)
        ytrue = []
        for yp in ypre:
            if yp[0]>yp[1]:
                ytrue.append(1)
            else:
                ytrue.append(0)
        yvoting.append(ytrue)
        cost = nn1.costFunc(X,Y)
        costJ+=cost
    f1 = 0
    yresult = []
    i = 0
    while i<len(Y):
        ones = 0
        zeros = 0
        for j in range(k):
            if yvoting[j][i] == 1:
                ones+=1
            else:
                zeros+=1
        i+=1
        if ones>zeros:
            yresult.append(1)
        elif ones==zeros:
            n = nn1.rand()
            if n>0:
                yresult.append(1)
            else:
                yresult.append(0)
        else:
            yresult.append(0)
    acc,f1 = evaluate(yresult,Y)
    J = costJ/k
    return acc,f1, J

js = []

lamda = 0.1
acc, f1, J = kfoldTrain(x,ynew, 10, structure1, lamda)
print('acc: ',acc, ' f1: ',f1)
print(J)
js.append(J)
lamda = 0.2
acc, f1, J = kfoldTrain(x,ynew, 10, structure2, lamda)
print('acc: ',acc, ' f1: ',f1)
js.append(J)
lamda = 0.25
acc, f1, J = kfoldTrain(x,ynew, 10, structure3, lamda)
print('acc: ',acc, ' f1: ',f1)
js.append(J)
lamda = 0.25
acc, f1, J = kfoldTrain(x,ynew, 10, structure4, lamda)
print('acc: ',acc, ' f1: ',f1)
js.append(J)
lamda = 0.2
acc, f1, J = kfoldTrain(x,ynew, 10, structure5, lamda)
print('acc: ',acc, ' f1: ',f1)
js.append(J)
lamda = 0.25
acc, f1, J = kfoldTrain(x,ynew, 10, structure6, lamda)
print('acc: ',acc, ' f1: ',f1)
js.append(J)
print(js)



structure1 = [13,10,6,3,2]
structure2 = [13,10,8,4,2]
structure3 = [13,10,8,2]
structure4 = [13,8,4,2]
structure5 = [13,8,2]
structure6 = [13,6,2]
datasets = np.array(pandas.read_csv('datasets/hw3_wine.csv'))
x = datasets[:,[1,2,3,4,5,6,7,8,9,10,11,12,13]]
y = datasets[:,0]

ynew = []
for i in range(len(y)):
    if y[i] == 0:
        ynew.append([1,0,0])
    elif y[i] == 1:
        ynew.append([0,1,0])
    elif y[i] == 2:
        ynew.append([0,0,1])

#x_trains, y_trains, x_test, y_test = kfold(x,ynew,10)

js = []

lamda = 0.1
acc, f1, J = kfoldTrain(x,ynew, 10, structure1, lamda)
print('acc: ',acc, ' f1: ',f1)
js.append(J)
lamda = 0.2
acc, f1, J = kfoldTrain(x,ynew, 10, structure2, lamda)
print('acc: ',acc, ' f1: ',f1)
js.append(J)
lamda = 0.25
acc, f1, J = kfoldTrain(x,ynew, 10, structure3, lamda)
print('acc: ',acc, ' f1: ',f1)
js.append(J)
lamda = 0.25
acc, f1, J = kfoldTrain(x,ynew, 10, structure4, lamda)
print('acc: ',acc, ' f1: ',f1)
js.append(J)
lamda = 0.2
acc, f1, J = kfoldTrain(x,ynew, 10, structure5, lamda)
print('acc: ',acc, ' f1: ',f1)
js.append(J)
lamda = 0.25
acc, f1, J = kfoldTrain(x,ynew, 10, structure6, lamda)
print('acc: ',acc, ' f1: ',f1)
js.append(J)
print(js)