import pandas
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

datasets = np.array(pandas.read_csv('iris.csv'))
print(datasets)

x = datasets[:,[0,1,2,3]]
y = datasets[:,4]

y_processed = []
for thing in y:
    if thing == 'Iris-setosa':
        y_processed.append(0)
    if thing == 'Iris-versicolor':
        y_processed.append(1)
    if thing == 'Iris-virginica':
        y_processed.append(2)
print(y_processed)

trainX, testX, trainY, testY = train_test_split(x, y_processed, test_size=0.2)

# distance calculation
from math import sqrt
def distance_Cal(k, testCase, trainX, trainY):
    distanceArr = []
    count = 0
    for point in trainX:
        distance0 = testCase[0] - point[0]
        distance1 = testCase[1] - point[1]
        distance2 = testCase[2] - point[2]
        distance3 = testCase[3] - point[3]
        distanceTotal = distance0*distance0 + distance1*distance1 + distance2*distance2 + distance2*distance2
        distanceTotal = sqrt(distanceTotal)
        #print("trainY[count]")
        #print(trainY[count])
        #print("distanceTotal")
        #print(distanceTotal)
        distanceArr.append([distanceTotal,trainY[count]])
        count = count+1
    distanceNP = np.array(distanceArr)
    distanceSort = distanceNP[np.argsort(distanceNP[:, 0])]
    #distanceSort = np.sort(distanceNP,axis = 0)
    #print(distanceSort)
    count0 = 0
    count1 = 0
    count2 = 0
    countk = 0
    while countk<k:
        temp = distanceSort[countk]
        if temp[1] == 0:
            count0 = count0+1
        if temp[1] == 1:
            count1 = count1+1
        if temp[1] == 2:
            count2 = count2+1
        countk = countk+1
    #print("-----")
    #print(count0)
    #print(count1)
    #print(count2)
    if count0>count1:
        if count0>count2:
            return 0
    if count1>count0:
        if count1>count2:
            return 1
    if count2>count1:
        if count2>count0:
            return 2

# for checking accuracy
def printOutPredict(prediction, accurate):
    totalNum = len(prediction)
    accNum = 0
    index = 0
    while index<totalNum:
        #print(prediction[index])
        #print(accurate[index])
        if prediction[index]==accurate[index]:
            accNum = accNum+1
        index = index+1
    #print("predicted data accuracy:")
    #print(accNum*100.0/totalNum)
    return accNum*100.0/totalNum

def train_predict(k,trainX, trainY,testX, testY):
    prediction = []
    for testData in testX:
        #print(distance_Cal(k,testData,trainX,trainY))
        prediction.append(distance_Cal(k,testData,trainX,trainY))
    acc = printOutPredict(prediction,testY)
    return prediction

prediction = train_predict(k,trainX,trainY,trainX,trainY)
#prediction = train_predict(k,trainX,trainY,testX,testY)
print(prediction)

k = 1
predictionTrainSet = []
predictionTestSet = []
while k<=51:
    index = 0
    trainSet = []
    testSet = []
    while index<20:
        trainX, testX, trainY, testY = train_test_split(x, y_processed, test_size=0.2)
        predictionTrain = train_predict(k,trainX,trainY,trainX,trainY)
        predictionTest = train_predict(k,trainX,trainY,testX,testY)
        trainSet.append(printOutPredict(predictionTrain,trainY))
        testSet.append(printOutPredict(predictionTest,testY))
        index += 1
    predictionTrainSet.append(trainSet)
    predictionTestSet.append(testSet)
    k+=2

import matplotlib.pyplot as plt
index = 1
x = []
while index<52:
    x.append(index)
    index+=2

import matplotlib.pyplot as plt
import math
def show_plot_of(accuracies,ks)->None:    
    accuracies_average = []
    accuracies_std = []
    for accuracy in accuracies:
        accuracies_average.append(sum(accuracy)/len(accuracy))
        accuracies_std.append(np.std(accuracy))
    plt.plot(ks,accuracies_average)
    plt.scatter(ks, accuracies_average)
    plt.errorbar(ks,accuracies_average,yerr=accuracies_std,capsize = 3)
    plt.xlabel('Value of k')
    plt.ylabel('Accuracy')
    plt.show

show_plot_of(predictionTrainSet,x)