import numpy as np
import pandas
from sklearn import datasets
import matplotlib . pyplot as plt
from sklearn.preprocessing import maxabs_scale
from knn import knn
from sklearn.model_selection import train_test_split
from kfold import kfoldTrain

#4. Parkinsons
datasets = np.array(pandas.read_csv('parkinsons.csv'))
x = datasets[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
y = datasets[:,22]
#for first 3 column in x, x[i][0,1,2] = x[i][0,1,2]/max and HNR also to be processed like this and spread1 should be spread1/minspread1 since it is all negative
max1 = 0
max2 = 0
max3 = 0
maxhnr = 0
minspread1 = 100
for i in range(len(y)):
    if x[i][0]>max1:
        max1 = x[i][0]
    if x[i][1]>max2:
        max2 = x[i][1]
    if x[i][2]>max3:
        max3 = x[i][2]
    if x[i][15]>maxhnr:
        maxhnr = x[i][15]
    if x[i][18]<maxhnr:
        minspread1 = x[i][18]
for i in range(len(y)):
    x[i][0] = x[i][0]/max1
    x[i][1] = x[i][1]/max2
    x[i][2] = x[i][2]/max3
    x[i][15] = x[i][15]/maxhnr
    x[i][18] = x[i][18]/minspread1


trainx, testx, trainy, testy =train_test_split(x,y,test_size=0.3)
'''
ks = []
accs = []
f1s = []
k = 1
while k<50:
    ypredict = kfoldTrain(trainx,trainy,testx,10,k)
    truenum = 0
    a = 0
    b = 0
    c = 0
    for i in range(len(ypredict)):
        if ypredict[i] == testy[i]:
            truenum +=1
            if ypredict[i] == 1:
                c+=1
        if ypredict[i] == 1:
            a+=1
        if testy[i] == 1:
            b+=1
    f1 = 2*c/(a+b)
    acc = truenum/len(ypredict)
    print('K: ',k ,' Acc: ', acc, ' F1: ', f1)
    ks.append(k)
    accs.append(acc)
    f1s.append(f1)
    k+=2

plt.plot(ks, accs) 
plt.show()
plt.plot(ks, f1s) 
plt.show()
'''
#best performe at k=9
i = 0
learningRate = []
accs = []
f1s = []
while i<0.99:
    i+=0.08
    if i>1:
        i = 1
        trainx1 = trainx
        trainy1 = trainy
    else:
        trainx1 = []
        trainy1 = []
        lent = i*len(trainy)
        for k in range(int(lent)):
            trainx1.append(trainx[k])
            trainy1.append(trainy[k])
    ypredict = kfoldTrain(trainx1,trainy1,testx,10,9)
    #model = knn(5,trainx,trainy)
    #ypredict = model.predict(testx)
    truenum = 0
    a = 0
    b = 0
    c = 0
    for j in range(len(ypredict)):
        if ypredict[j] == testy[j]:
            truenum +=1
            if ypredict[j] == 1:
                c+=1
        if ypredict[j] == 1:
            a+=1
        if testy[j] == 1:
            b+=1
    f1 = 2*c/(a+b)
    acc = truenum/len(ypredict)
    print('LearningRate: ', "%.2f%%" % (i * 100) ,' Acc: ', acc, ' F1: ', f1)
    learningRate.append(i)
    accs.append(acc)
    f1s.append(f1)

plt.plot(learningRate, accs) 
plt.show()
plt.plot(learningRate, f1s) 
plt.show()
