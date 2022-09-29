from collections import defaultdict
import math
from re import X

import numpy as np


class knn:
    def __init__(self,k,x,y):
        self.k = k
        self.x = x
        self.y = y
        dicty = defaultdict(int)
        self.ynum = 0
        for i in range(len(self.y)):
            if y[i] not in dicty:
                dicty[y[i]]+=1
                self.ynum+=1
    
    def predict(self,trainx):
        outlist = []
        lentotal = len(self.y)
        for t in trainx:
            d = self.distance(t)
            outlist.append(d)
        return outlist

    def distance(self, xi):
        distances = []
        for i in range(len(self.x)):
            distances.append(self.distanceHelp(xi,self.x[i],self.y[i]))
        npdis = np.array(distances)
        sorteddistance = npdis[npdis[:,0].argsort()]
        #print(distances)
        i = 0
        ydecide = defaultdict(int)
        while i<self.k:
            if sorteddistance[i][1] not in ydecide:
                ydecide[sorteddistance[i][1]] = 1
            else:
                ydecide[sorteddistance[i][1]]+=1
            i+=1
        maxnum = 0
        returnnum = 0
        for i in ydecide:
            if ydecide[i]>maxnum:
                maxnum = ydecide[i]
                returnnum = i
        return returnnum
    
    def distanceHelp(self,xtest,xmeaning,ymeaning): 
        i = 0
        d = 0
        while i<len(xtest):
            d+= (xtest[i]-xmeaning[i])**2
            i+=1
        return [math.sqrt(d),ymeaning]
    
    