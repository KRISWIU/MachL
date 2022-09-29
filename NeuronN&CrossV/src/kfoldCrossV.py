from collections import Counter
import numpy as np
import pandas
from sklearn.model_selection import KFold
from nn import nn

def kfold(x: np, y: np, k):
    xks = [[] for i in range(k)]
    yks = [[] for i in range(k)]
    i = -1
    for j in range(len(y)):
        i+=1
        if i>=k:
            i = 0
        xks[i].append(x[j])
        yks[i].append(y[j])
    
    x_trains = []
    x_test = []
    y_trains = []
    y_test = []
    i = 0
    while i<k:
        tempx = []
        tempy = []
        x_test.append(xks[i])
        y_test.append(yks[i])
        j = 0
        while j<k:
            if i==j:
                j+=1
            else:
                tx = xks[j]
                ty = yks[j]
                for x in tx:
                    tempx.append(x)
                for y in ty:
                    tempy.append(y)
                j+=1
        x_trains.append(tempx)
        y_trains.append(tempy)
        i+=1
    return x_trains, y_trains, x_test, y_test
    
def evaluate(ytest, ytrue):
    tpfp = 1
    tp = 1
    tpfn = 1
    i = 0
    while i<len(ytrue):
        if ytest[i]==ytrue[i]:
            tpfn+=1
            if ytest[i] == 1:
                tp+=1
        if ytest[i] == 1:
            tpfp+=1
        i+=1
    print('acc:')
    acc = tpfn/len(ytrue)
    print(acc)
    #print('recall:')
    re = tp/tpfn*1.0
    #print(re)
    #print('precision')
    pre = tp/tpfp*1.0
    #print(pre)
    print('f1')
    f1 = (re+pre)/2
    print(f1)
    return acc, f1
