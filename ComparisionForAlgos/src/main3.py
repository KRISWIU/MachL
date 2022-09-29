import numpy as np
import pandas
from sklearn import datasets
import matplotlib . pyplot as plt
from sklearn.preprocessing import maxabs_scale
from knn import knn
from sklearn.model_selection import train_test_split
from kfold import kfoldTrain

#3. Loan
datasets = np.array(pandas.read_csv('loan.csv'))
x = datasets[:,[1,2,3,4,5,6,7,8,9,10,11,12]]#last column is for furter use of if the candidate have CoapplicantIncome
y = datasets[:,12]
maxApplicantIncome = 0
maxCoapplicantIncome = 0
maxLoanAmount = 0
for i in range(len(y)):
    if x[i][5]>maxApplicantIncome:
        maxApplicantIncome = x[i][5]
    if x[i][6]>maxCoapplicantIncome:
        maxCoapplicantIncome = x[i][6]
    if x[i][7]>maxLoanAmount:
        maxLoanAmount = x[i][7]

for i in range(len(y)):
    #reset value in y: Y:1, N:0
    if y[i] == 'Y':
        y[i] = 1
    else:
        y[i] = 0
    #for last column in x, urban:2, semiurban:1, rural:0
    if x[i][10] == 'Rural':
        x[i][10] = 0
    elif x[i][10] == 'Semiurban':
        x[i][10] = 1
    elif x[i][10] == 'Urban':
        x[i][10] = 2
    #set male to 1 an female to 0
    if x[i][0] == 'Female':
        x[i][0] = 0
    elif x[i][0] == 'Male':
        x[i][0] = 1
    #set second column in x, Yes = 1, No = 0
    if x[i][1] == 'No':
        x[i][1] = 0
    elif x[i][1] == 'Yes':
        x[i][1] = 1
    if x[i][2] == '1':
        x[i][2] = 1
    else:
        x[i][2] = 0
    #set 3rd column in x, Graduate = 1, UnderG = 0
    if x[i][3] == 'Graduate':
        x[i][3] = 1
    else:
        x[i][3] = 0
    #self employeed or not, E = 1, not E = 0
    if x[i][4] == 'No':
        x[i][4] = 0
    else:
        x[i][4] = 1
    #loan amount term usually 360 is the largerst amount so =loan/360
    x[i][8] = x[i][8]/360
    #Since there are many people with out CoapplicantIncome, so last column is set for if this person have CoapplicantIncome
    if x[i][7]>0:
        x[i][11] = 1
    else:
        x[i][11] = 0
    x[i][5] = x[i][5]/maxApplicantIncome
    x[i][6] = x[i][6]/maxCoapplicantIncome
    x[i][7] = x[i][7]/maxLoanAmount


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
while i<1:
    i+=0.03
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
