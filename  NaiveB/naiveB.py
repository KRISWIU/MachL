from collections import defaultdict
from email.policy import default
import math
import random


class naiveB(object):
    def __init__(self,alpha) -> None:
        self.alpha = alpha
        pass

    def training(self,pos_train, neg_train, vocab):
        posdiction = defaultdict()
        negdiction = defaultdict()
        self.Ptrue = len(pos_train)/(len(pos_train) + len(neg_train))
        self.Pfalse = len(neg_train)/(len(neg_train) + len(pos_train))
        totalpos = 0
        for pos in pos_train:
            for word in pos:
                if word not in posdiction:
                    posdiction[word] = 1
                else:
                    posdiction[word] +=1
                totalpos +=1
        totalneg = 0
        for neg in neg_train:
            for word in neg:
                if word not in negdiction:
                    negdiction[word] = 1
                else:
                    negdiction[word] +=1
                totalneg +=1

        postrained = defaultdict()
        negtrained = defaultdict()
        self.posdn = posdiction
        self.negdn = negdiction
        
        for word in posdiction:
            #postrained[word] = posdiction[word]/totalpos
            postrained[word] = (posdiction[word]+self.alpha)/(totalpos+self.alpha*len(vocab))
        for word in negdiction:
            #negtrained[word] = negdiction[word]/totalneg
            negtrained[word] = (negdiction[word]+self.alpha)/(totalneg+self.alpha*len(vocab))
        self.posword = postrained
        self.negword = negtrained
        self.helpneg = (self.alpha)/(totalpos+self.alpha*len(vocab))
        self.helppos = (self.alpha)/(totalpos+self.alpha*len(vocab))
    
    def testing(self, pos_test, neg_test):
        posDict = self.posword
        negDict = self.negword
        totalpos = 0
        posnum = 0
        logpos = 0
        for pos in pos_test:
            totalpos += 1
            logPt = 0.0
            logPf = 0.0
            for word in pos:
                if word in posDict:
                    logPt += math.log(posDict[word])
                if word in negDict:
                    logPf += math.log(negDict[word])
            logPt += math.log(self.Ptrue)
            logPf += math.log(self.Pfalse)
            if logPt>logPf:
                logpos+=1

            '''
            Pt = self.Ptrue
            Pf = self.Pfalse
            for word in pos:
                if word in posDict:
                    Pt = Pt * posDict[word]
                else:
                    Pt = 0
                if word in negDict:
                    Pf = Pf * negDict[word]
                else:
                    Pf = 0
                if Pt == 0 and Pf == 0:
                    break
            if Pt>Pf:
                posnum+=1
            elif Pt==0 and Pf == 0:
                random_bit = random.getrandbits(1)
                random_boolean = bool(random_bit)
                if random_boolean:
                    posnum+=1
            if Pt == 0:
                logPt = math.log(self.Ptrue)
            else:
                logPt = math.log(Pt)
            if Pf == 0:
                logPf = math.log(self.Pfalse)
            else:
                logPf = math.log(Pf)
            if logPt>logPf:
                logpos +=1
            '''
        
        totalneg = 0
        negnum = 0
        logneg = 0
        for neg in neg_test:
            totalneg+=1
            logPt = 0.0
            logPf = 0.0
            #Pt = self.Ptrue
            #Pf = self.Pfalse
            for word in neg:
                if word in posDict:
                    logPt += math.log(posDict[word])
                if word in negDict:
                    logPf += math.log(negDict[word])
            logPt += math.log(self.Ptrue)
            logPf += math.log(self.Pfalse)
            if logPf>logPt:
                logneg+=1

            '''
            for word in neg:
                if word in posDict:
                    Pt = Pt * posDict[word]
                else:
                    Pt = 0
                if word in negDict:
                    Pf = Pf * negDict[word]
                else:
                    Pf = 0
                if Pt == 0 and Pf == 0:
                    break
            if Pt<Pf:
                negnum+=1
            elif Pt==0 and Pf==0:
                random_bit = random.getrandbits(1)
                random_boolean = bool(random_bit)
                if random_boolean:
                    negnum+=1
            if Pt == 0:
                logPt = math.log(self.Ptrue)
            else:
                logPt = math.log(Pt)
            if Pf == 0:
                logPf = math.log(self.Pfalse)
            else:
                logPf = math.log(Pf)
            if logPf>logPt:
                logneg+=1
            '''
            
        print(totalneg)
        print(totalpos)
        print("contusion matrix: ")
        print("True Positive: ", logpos)
        print("True Negative: ", logneg)
        print("False Positive: ", totalpos - logpos)
        print("False Negative: ", totalneg - logneg)
        print("Accuracy on test set: ", 1.0*(logpos+logneg)/(totalpos+totalneg))
        print("Recall: ", 1.0*logpos/(logpos+totalneg-logneg))
        print("Precision: ", 1.0*logpos/totalpos)
    
    def smoothing(self, pos_test, neg_test):
        value = 0.0001
        posDict = self.posword
        negDict = self.negword
        totalpos = 0
        posnum = 0
        logpos = 0
        for pos in pos_test:
            totalpos += 1
            logPt = 0.0
            logPf = 0.0
            for word in pos:
                if word in posDict:
                    logPt += math.log(posDict[word])
                else:
                    logPt += math.log(self.helppos)
                if word in negDict:
                    logPf += math.log(negDict[word])
                else:
                    logPf += math.log(self.helpneg)
            logPt += math.log(self.Ptrue)
            logPf += math.log(self.Pfalse)
            if logPt>logPf:
                logpos+=1
        
        totalneg = 0
        negnum = 0
        logneg = 0
        for neg in neg_test:
            totalneg+=1
            logPt = 0.0
            logPf = 0.0
            #Pt = self.Ptrue
            #Pf = self.Pfalse
            for word in neg:
                if word in posDict:
                    logPt += math.log(posDict[word])
                if word in negDict:
                    logPf += math.log(negDict[word])
            logPt += math.log(self.Ptrue)
            logPf += math.log(self.Pfalse)
            if logPf>logPt:
                logneg+=1
        
        print("contusion matrix: ")
        print("True Positive: ", logpos)
        print("True Negative: ", logneg)
        print("False Positive: ", totalpos - logpos)
        print("False Negative: ", totalneg - logneg)
        print("Accuracy on test set: ", 1.0*(logpos+logneg)/(totalpos+totalneg))
        print("Recall: ", 1.0*logpos/(logpos+totalneg-logneg))
        print("Precision: ", 1.0*logpos/totalpos)

