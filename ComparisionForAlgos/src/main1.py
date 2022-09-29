from sklearn.model_selection import learning_curve, train_test_split
import numpy as np
import pandas
from sklearn import datasets
import matplotlib . pyplot as plt
from sklearn.preprocessing import maxabs_scale
from knn import knn
from sklearn.model_selection import train_test_split
from kfold import kfoldTrain

#1. Handwritten

digits = datasets.load_digits(return_X_y=True)
digits_dataset_X = digits[0]
x = digits_dataset_X
digits_dataset_y = digits[1]
y = digits_dataset_y
#print(digits_dataset_X)
#print(digits_dataset_y)
N = len(digits_dataset_y)
'''
digit_to_show = np.random.choice(range(N),1)[0]
print("Attributes:", digits_dataset_X[digit_to_show])
print("Class:", digits_dataset_y[digit_to_show])
plt.imshow(np.reshape(digits_dataset_X[digit_to_show], (8,8)))
plt.show()
'''

trainx, testx, trainy, testy =train_test_split(x,y,test_size=0.3)

ks = []
accs = []
f1s = []
k = 3
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
#best performe at k=3
i = 0
learningRate = []
accs = []
f1s = []
while i<1:
    i+=0.05
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
    ypredict = kfoldTrain(trainx1,trainy1,testx,10,3)
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