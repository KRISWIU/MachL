import math
from pydoc import tempfilepager
import random
from re import A
from signal import siginterrupt
import time
from xmlrpc.client import boolean
from numpy import append
import sklearn
import numpy as np
from torch import layer_norm, sigmoid


class nn:
    def __init__(self, structure: list, lamda:int, weightinit: int = 1):
        self.structure = structure
        self.lamda = lamda
        self.weightinit = weightinit
        self.L = len(structure)
        #initial the weights for evry layer
        self.weights = []
        for i in range(self.L-1):
            temp = []
            for j in range(structure[i+1]):
                if i == self.L-1:
                    weight1 = []
                    for j in range(structure[i]+1):
                        weight1.append(self.rand())
                    weight = [self.rand()]*(structure[i]+1)
                else:
                    weight1 = []
                    for j in range(structure[i]+1):
                        weight1.append(self.rand())
                    weight = [self.rand()]*(structure[i]+1)
                w = np.array(weight1)
                temp.append(w)
            self.weights.append(temp)
        w = np.array(self.weights)
        self.weights = w
    
    def rand(self):
        return (1-(-1)) * np.random.random() + (-1)

    def benchmark1(self):
        #print(self.weights)
        x = [[0.13000],[0.42000]]
        y = [[0.90000],[0.23000]]
        self.weights = [[np.array([0.40000,0.10000]),np.array([0.30000,0.20000])],np.array([0.70000,0.50000,0.60000])]
        print('Values: 1.fx1 2.fx2 3.avalues for x1 4.avalues for x2')
        print(self.predict(x))
        avalues = self.outputEvryNeu(x[0])
        print(avalues)
        avalues = self.outputEvryNeu(x[1])
        print(avalues)
        cost = self.costFunc(x,y)
        #print(self.weights)
        print('cost for the dataset and original weights:')
        print(cost)
        print('delta values for each instance:')
        self.benchhelp(x,y)
        print('final regualrized D:')
        print(self.backP(x,y))
    
    def benchmark2(self):
        x = [[0.32000,0.68000],[0.83000,0.02000]]
        y = [[0.75000,0.98000],[0.75000,0.28000]]
        self.weights = [[np.array([0.42000,0.15000,0.40000]),np.array([0.72000,0.10000,0.54000]),np.array([0.01000,0.19000,0.42000] ),np.array([0.30000,0.35000,0.68000])]
                        ,[np.array([0.21000,0.67000,0.14000,0.96000,0.87000]),np.array([0.87000,0.42000,0.20000,0.32000,0.89000]),np.array([0.03000,0.56000,0.80000,0.69000,0.09000])]
                        ,[np.array([0.04000,0.87000,0.42000,0.53000]),np.array([0.17000,0.10000,0.95000,0.69000])]]
        print('Values: 1.fx1 2.fx2 3.avalues for x1 4.avalues for x2')
        print(self.predict(x))
        avalues = self.outputEvryNeu(x[0])
        print(avalues)
        avalues = self.outputEvryNeu(x[1])
        print(avalues)
        cost = self.costFunc(x,y)
        #print(self.weights)
        print('cost for the dataset and original weights:')
        print(cost)
        print('delta values for each instance:')
        self.benchhelp(x,y)
        print('final regualrized D:')
        print(self.backP(x,y))

    #backward
    def train(self, x, y):
        #for i in range(10) and self.costFunc(x,y)<0.1:
        cost1 = 100
        cost2 = 99
        i = 0
        startTime= time.strftime("%Y-%m-%d %H:%M:%s", time.localtime())
        print('train start at:',startTime)
        #print(self.weights)
        #print(self.costFunc(x,y))
        while i<200 or (cost1-cost2>0.0001 or cost1-cost2<-0.0001):
            self.backP(x,y)
            cost2 = cost1
            cost1 = self.costFunc(x,y)
            i+=1
        #print(self.weights)
        #print(self.costFunc(x,y))
        endTime= time.strftime("%Y-%m-%d %H:%M:%s", time.localtime())
        print('train end at:', endTime)
    
    #for evry data in dataset
    def outputEvryNeu(self, data):
        neus = []
        temp = []
        neus.append(np.append(1,data))
        for i in range(self.L-1):
            if i == 0:
                d = np.append(1,data)
                t = np.array(self.weights[i])
                #print(t)
                #print(d.T)
                neu = (t).dot(d.T)
                #rint(neu)
                for n in neu:
                    temp.append(self.sigmiod(n))
                temp = np.append(1,temp)
                neus.append(np.array(temp))
            else:
                t = np.array(self.weights[i])
                neu = (t).dot(temp.T)
                temp = []
                #print('new')
                #print(neu)
                #print(neu)
                if type(neu) == np.float64:
                    temp.append(self.sigmiod(neu))
                else:
                    for n in neu:
                        temp.append(self.sigmiod(n))
                if i == self.L-2:
                    neus.append(np.array(temp))
                else:
                    temp = np.append(1,np.array(temp))
                    neus.append(temp)
                temp = np.array(temp)
                #neus.append(neu)
        return neus
    
    def benchhelp(self,x, y):
        l = []
        D = [l for i in range(self.L)] 
        #print(D)
        for i in range(len(y)):
            xi = x[i]
            yi = y[i]
            fi = self.output(xi)
            delta = fi - yi
            deltas = [[] for i in range(self.L)]                                
            deltas[self.L-1] = delta
            avalues = self.outputEvryNeu(xi)
            layerNum = self.L
            while layerNum>2:
                layerNum-=1
                delta1 = deltas[layerNum]
                ak = avalues[layerNum-1]
                ak1 = 1 - ak
                xita = np.array(self.weights[layerNum-1])
                xitaT = []
                if xita.shape == xita.T.shape:
                    for i in xita:
                        xitaT.append([i])
                    xitat = np.array(xitaT)
                else:
                    xitat = xita.T
                temp = (xitat).dot(np.array(delta1))
                deltak = temp*ak*ak1
                deltak = np.delete(deltak, 0, axis=0)
                deltas[layerNum-1] = deltak
            layerNum = self.L
            while layerNum>1:
                layerNum-=1
                Dk = np.array(D[layerNum])
                ak = avalues[layerNum-1]
                deltak = deltas[layerNum]
                lk = len(deltak)
                if len(np.array(ak).shape) == 1:
                    deltakk = np.array(deltak).reshape((lk,1))
                if len(np.array(ak).shape) == 1:
                    ak = np.array(ak).reshape((np.array(ak).shape[0],1))
                dd = deltakk.dot(ak.T)
                if D[layerNum] == []:
                    D[layerNum] = np.array(dd)
                else:
                    D[layerNum] = Dk + np.array(dd)
                print('delta value:')
                #print(D[layerNum])
                print(dd)
    
    def backP(self,x, y):
        l = []
        D = [l for i in range(self.L)] 
        #print(D)
        for i in range(len(y)):
            xi = x[i]
            yi = y[i]
            fi = self.output(xi) 
            #print("instanse value:")
            #print(xi)
            #print(yi)
            #print(fi)
            delta = fi - yi
            #print(delta)
            deltas = [[] for i in range(self.L)]                                
            deltas[self.L-1] = delta
            avalues = self.outputEvryNeu(xi)
            #print(deltas)
            #print(avalues)
            #deltavalues = []
            layerNum = self.L
            while layerNum>2:
                layerNum-=1
                #print("new1")
                #print(layerNum)
                delta1 = deltas[layerNum]
                #print('delta:')
                #print(delta1)
                ak = avalues[layerNum-1]
                ak1 = 1 - ak
                #print('ak')
                #print(ak)
                #print(ak1)
                #temp = delta1*ak*ak1
                xita = np.array(self.weights[layerNum-1])
                #print("xita")
                #print(xita)
                xitaT = []
                if xita.shape == xita.T.shape:
                    for i in xita:
                        xitaT.append([i])
                    xitat = np.array(xitaT)
                else:
                    xitat = xita.T
                #print(xitat)
                temp = (xitat).dot(np.array(delta1))
                deltak = temp*ak*ak1
                #print(layerNum+1)
                #print('deltak:')
                #print(deltak)
                #print(deltak.shape)
                deltak = np.delete(deltak, 0, axis=0)
                #print('deltak:')
                #print(deltak)
                deltas[layerNum-1] = deltak
            layerNum = self.L
            #print(avalues)
            #print(deltas)
            
            while layerNum>1:
                layerNum-=1
                #print('new2')
                #print(layerNum)
                Dk = np.array(D[layerNum])
                ak = avalues[layerNum-1]
                deltak = deltas[layerNum]

                lk = len(deltak)
                if len(np.array(ak).shape) == 1:
                    deltakk = np.array(deltak).reshape((lk,1))
                if len(np.array(ak).shape) == 1:
                    ak = np.array(ak).reshape((np.array(ak).shape[0],1))
                
                #print(deltakk.dot(ak.T))
                dd = deltakk.dot(ak.T)
                #dd = np.transpose(np.array(ak).dot(np.array(deltak)))
                if D[layerNum] == []:
                    D[layerNum] = np.array(dd)
                else:
                    D[layerNum] = Dk + np.array(dd)
                #print('ddd')
                #print(D[layerNum])
                #print(dd)
                #print(np.array(D[layerNum]).shape)
        #print(deltas)
        #print(avalues)
        #print('d')
        #print(D)
        layerNum = self.L
        while layerNum>1:
            layerNum-=1
            #print('new3')
            #print(layerNum)
            #xitaa = self.outputEvryNeu(xi)
            xita = np.array(self.weights[layerNum-1])
            #xita = xitaa[layerNum-1]
            lamda = self.lamda
            #print(xita)
            Pk = lamda*xita
            #print(Pk)
            Pk[0] = 0
            #print(D[layerNum])
            #print(np.array(D[layerNum]).shape)
            datalen = len(y)
            if np.array(D[layerNum]).shape == (0,):
                #print('1')
                D[layerNum] = (1/datalen)*(Pk)
                #print(D[layerNum])
            else:
                D[layerNum] = (1/datalen)*(D[layerNum]+Pk)
                #print(D[layerNum])
            if layerNum!=1:
                #print('new')
                #print(self.weights[layerNum-1])
                #print(layerNum)
                #print(D[layerNum])
                self.weights[layerNum-1] = self.weights[layerNum-1] - 0.01*D[layerNum]
            elif layerNum == 1:
                #print('new1')
                #print(self.weights[layerNum-1])
                #print(layerNum)
                #print(D[layerNum])
                #print(D)
                self.weights[0] = self.weights[0]- 0.01*np.array(D[layerNum])
        return D

    #for checking convergence
    def costFunc(self, x, y):
        sumJ = 0
        n = 0
        for i in range(len(y)):
            n+=1
            #if sum(self.costFuncHelp(x[i],y[i]))>1:
                #print(x[i])
                #print(y[i])
            sumJ += sum(self.costFuncHelp(x[i],y[i]))
        J = sumJ/n
        #print(J)
        S = self.weightsSquare()
        S = (self.lamda/(2*n))*S
        return J+S

    #cakculate one cost for on data
    def costFuncHelp(self, data, y1):
        fx = self.output(data)
        #print(fx)
        fxi = np.array(fx)
        #print(fxi)
        yi = np.array(y1)
        #print(fxi)
        #print(yi)
        j = -yi*(np.log(fxi))-(1-yi)*(np.log(1-fxi))
        return j
    
    def weightsSquare(self):
        sum = 0
        for weight in self.weights:
            for w in weight:
                if type(w)==np.float64:
                    sum+=w
                else:
                    for i in w:
                        if i!=0:
                            sum+=i*i
        return sum

    #forward
    def predict(self, x):
        result = []
        for data in x:
            result.append(self.output(data))
        return result
    
    def output(self, data):
        out = self.outputEvryNeu(data)[self.L-1]
        #print(self.outputEvryNeu(data))
        return out

    def sigmiod(self, num):
        return 1/(1+(math.e)**(-num))