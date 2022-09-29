from collections import defaultdict
from knn import knn

def kfold(x, y, k):
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

def kfoldTrain(trainX, trainY,testX,k,knnvalue):
    x_trains, y_trains, x_test, y_test = kfold(trainX,trainY,k)
    yvoting = []
    for i in range(k):
        x = x_trains[i]
        y = y_trains[i]
        model = knn(knnvalue,x,y)
        ypre = model.predict(testX)
        yvoting.append(ypre)
    yresult = []
    i = 0
    while i<len(ypre):
        ndict = defaultdict()
        for j in range(k):
            if yvoting[j][i] not in ndict:
                ndict[yvoting[j][i]] = 1
            else:
                ndict[yvoting[j][i]] += 1
        max = 0
        result = -1
        for n in ndict:
            if ndict[n]>max:
                max = ndict[n]
                result = n
        yresult.append(result)
        i+=1
    return yresult
