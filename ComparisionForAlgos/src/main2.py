import numpy as np
import pandas
from sklearn import datasets
import matplotlib . pyplot as plt
from sklearn.preprocessing import maxabs_scale
from knn import knn
from sklearn.model_selection import train_test_split
from kfold import kfoldTrain

#2. titanic
datasets = np.array(pandas.read_csv('titanic.csv'))
#we do not need name for this dataset cuz we have siblings aboard
x = datasets[:,[1,3,4,5,6,7]]
y = datasets[:,0]
for i in range(len(y)):
    #set male to 1 an female to 0
    if x[i][1] == 'female':
        x[i][1] = 0
    if x[i][1] == 'male':
        x[i][1] = 1
    #reset fare(i found out that male with fare<10 likely to have survive = 0 and fare is just decide where the passanger would sit in real life so it could be normalized as catagories): 0-10:0 ; 10-20:1; 20-30:2; 30-50:3; 50+:4
    if x[i][5]<10:
        x[i][5] = 0
    elif x[i][5]<20:
        x[i][5] = 1
    elif x[i][5]<30:
        x[i][5] = 2
    elif x[i][5]<50:
        x[i][5] = 3
    else:
        x[i][5] = 4
    #reset age to age/100
    x[i][2] = x[i][2]/100


trainx, testx, trainy, testy =train_test_split(x,y,test_size=0.3)

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
#best performe at k=5
i = 0
learningRate = []
accs = []
f1s = []
while i<1:
    i+=0.01
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
    ypredict = kfoldTrain(trainx1,trainy1,testx,10,5)
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
'''