#CNA.py

import numpy as np


import matplotlib.pyplot as plt
import time
import scipy
from scipy import stats
from scipy.special import logsumexp
from scipy.special import softmax

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.optim import Optimizer


if __name__ == "__main__":
    from shared import *
else:
    from .shared import *


#np.random.seed(0)
#torch.manual_seed(1)




import sys
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)



class EmbedModel(nn.Module):
    def __init__(self, Nbin, Nrep, Ncall, withBAF):
        super(EmbedModel, self).__init__()

        #self.nonlin = torch.tanh
        self.nonlin = nn.ReLU()


        haploInt = 1
        if withBAF:
            haploInt = 2


        if True:#
            self.conv1 = torch.nn.Conv1d(haploInt*Ncall, 10, 10, 5)
            #self.conv1 = torch.nn.Conv1d(haploInt*Ncall, 20, 10, 5) #Try May 6 2023

            testArray = torch.zeros((1, haploInt*Ncall, Nbin))
            testArray = self.conv1(testArray)
            convSize = testArray.shape[1] * testArray.shape[2]

            #print (convSize)
            #quit()

        else:

            convSize = 500#100
            self.lin0 = torch.nn.Linear(haploInt*Ncall*Nbin, convSize)

        
        
        
        self.lin1 = torch.nn.Linear(convSize, Nrep)

        



        self.lin2 = torch.nn.Linear(Nrep, Nrep)




    def forward(self, x):

        
        shape1 = x.shape

        #x = x.reshape((shape1[0], shape1[1]*2))
        #x = self.lin1_0(x)



        if True:
            x = torch.swapaxes(x, 1, 2)
            x = self.conv1(x)
            x = self.nonlin(x)
            x = x.reshape((x.shape[0], x.shape[1]*x.shape[2]))

        else:

            x = x.reshape(( x.shape[0], x.shape[1]*x.shape[2] ))
            x = self.lin0(x)
            x = self.nonlin(x)

        x = self.lin1(x)


        return x



class InitialModel(nn.Module):
    def __init__(self, Nrep):
        super(InitialModel, self).__init__()

        self.nonlin = torch.tanh

        self.lin1 = torch.nn.Linear(Nrep, 4 )

        #self.lin1_1 = torch.nn.Linear(Nrep, Nrep)

    def forward(self, x):

        shape1 = x.shape

        x = self.nonlin(x)

        x = self.lin1(x)


        x[:, 0] = x[:, 0] - 4.6 #For now, biassing it to not stop

        return x


class StartModel(nn.Module):
    def __init__(self, Nbin, Nrep, withBAF):
        super(StartModel, self).__init__()

        self.nonlin = torch.tanh

        haploInt = 1
        if withBAF:
            haploInt = 2

        self.lin1 = torch.nn.Linear(Nrep, (Nbin*haploInt) + 1 )

        #self.lin1_1 = torch.nn.Linear(Nrep, Nrep)

    def forward(self, x):

        shape1 = x.shape

        x = self.nonlin(x)

        #x = self.lin1_1(x)
        #x = self.nonlin(x)

        x = self.lin1(x)

        return x

class EndModel(nn.Module):
    def __init__(self, Nbin, Nrep):
        super(EndModel, self).__init__()

        self.nonlin = torch.tanh

        self.lin1 = torch.nn.Linear(Nbin*2, Nrep)

        #self.lin1_1 = torch.nn.Linear(Nrep, Nrep)
        self.lin2 = torch.nn.Linear(Nrep, Nbin)

    def forward(self, x, startX):


        x = x + self.lin1(startX)

        x = self.nonlin(x)


        #x = self.lin1_1(x)
        #x = self.nonlin(x)


        x = self.lin2(x)

        return x


class CopyNumberModel(nn.Module):
    def __init__(self, Nbin, Nrep, Ncall):
        super(CopyNumberModel, self).__init__()

        self.nonlin = torch.tanh

        self.lin1 = torch.nn.Linear(Nbin*2, Nrep)
        self.lin2 = torch.nn.Linear(Nbin, Nrep)
        self.lin3 = torch.nn.Linear(Nrep, Ncall)

        #self.lin1_1 = torch.nn.Linear(Nrep, Nrep)

    def forward(self, x, startX, endX):


        x = x + self.lin1(startX)
        x = x + self.lin2(endX)

        x = self.nonlin(x)

        #x = self.lin1_1(x)
        #x = self.nonlin(x)


        x = self.lin3(x)

        return x


class CancerModel(nn.Module):
    def __init__(self, Nbin, Nrep, Ncall, withBAF):
        super(CancerModel, self).__init__()

        self.EmbedModel = EmbedModel(Nbin, Nrep, Ncall, withBAF)
        self.InitialModel = InitialModel(Nrep)
        self.CopyNumberModel = CopyNumberModel(Nbin, Nrep, Ncall)
        self.StartModel = StartModel(Nbin, Nrep, withBAF)
        self.EndModel = EndModel(Nbin, Nrep)

        self.bias = torch.nn.Parameter(torch.zeros(Nbin))
        #self.stdBias = torch.nn.Parameter(torch.zeros(1)+0.001)

    def forward(self, x):
        return x

    def embedder(self, x):
        return self.EmbedModel(x)

    def initial(self, x):
        return self.InitialModel(x)

    def starter(self, x):
        return self.StartModel(x)

    def ender(self, x, xStart):
        return self.EndModel(x, xStart)

    def caller(self, x, xStart, xEnd):
        return self.CopyNumberModel(x, xStart, xEnd)


    def biasAdjuster(self):
        
        
        #return torch.tanh(self.bias * 10) * 0.0
        #return torch.tanh(self.bias * 10) * 0.2
        return torch.tanh(self.bias * 10) * 0.1

    def normalizedBias(self, adjustment):

        #std1 = torch.abs(self.stdBias * 100) + 0.001
        adjustment = self.biasAdjuster()

        std1 = torch.mean(adjustment.detach() ** 2) ** 0.5
        std1 = std1 + 0.01

        #negativeLogProb = torch.sum( ((adjustment/std1[0]) ** 2)   + torch.log(std1[0])  )

        negativeLogProb = torch.sum( ((adjustment/std1) ** 2))

        return negativeLogProb









def fastAllArgwhere(ar):
    ar_argsort = np.argsort(ar)
    ar1 = ar[ar_argsort]
    _, indicesStart = np.unique(ar1, return_index=True)
    _, indicesEnd = np.unique(ar1[-1::-1], return_index=True) #This is probably needless and can be found from indicesStart
    indicesEnd = ar1.shape[0] - indicesEnd - 1
    return ar_argsort, indicesStart, indicesEnd



def loadnpz(name, allow_pickle=False):

    #This simple function more easily loads in compressed numpy files.

    if allow_pickle:
        data = np.load(name, allow_pickle=True)
    else:
        data = np.load(name)
    data = data.f.arr_0
    return data


def adjLog(num, adj):

    return torch.log(num + adj) - np.log(1 + adj)




def mapBAF(x):

    return (x * 0.8) + 0.05


def doChoice(x):

    #This is a simple function that selects an option from a probability distribution given by x.


    x = np.cumsum(x, axis=1) #This makes the probability cummulative
    x_norm = x[:, -1].reshape((-1, 1))
    x = x / x_norm



    randVal = np.random.random(size=x.shape[0])
    randVal = randVal.repeat(x.shape[1]).reshape(x.shape)

    x = randVal - x
    x2 = np.zeros(x.shape)
    x2[x > 0] = 1
    x2[x <= 0] = 0
    x2[:, -1] = 0

    x2 = np.sum(x2, axis=1)

    return x2


def uniqueValMaker(X):

    _, vals1 = np.unique(X[:, 0], return_inverse=True)

    for a in range(1, X.shape[1]):

        #vals2 = np.copy(X[:, a])
        #vals2_unique, vals2 = np.unique(vals2, return_inverse=True)
        vals2_unique, vals2 = np.unique(X[:, a], return_inverse=True)

        vals1 = (vals1 * vals2_unique.shape[0]) + vals2
        _, vals1 = np.unique(vals1, return_inverse=True)

    return vals1



def findPossibleStart(CNAfull, currentCNA, start1, boolDouble, withBAF):


    haploInt = 1
    if withBAF:
        haploInt = 2
    dimSize1 = CNAfull.shape[1] * haploInt
    startBool = torch.zeros((CNAfull.shape[0], dimSize1+1  ), dtype=int)
    startBool[:, -1] = 1

    for b in range(CNAfull.shape[0]):

        #startBool_mini = torch.zeros((CNAfull.shape[0], (CNAfull.shape[1] * 2)+1  ))

        doubleNow = boolDouble[b]

        #CNAfull_now0 = CNAfull[b]
        #currentCNA_now0 = currentCNA[b]

        for pair1 in range(haploInt):

            #if withBAF:
            #    CNAfull_now = CNAfull_now0[:, pair1]
            #    currentCNA_now = currentCNA_now0[:, pair1]
            #else:
            #    CNAfull_now = CNAfull_now0
            #    currentCNA_now = currentCNA_now0


            diff1 = CNAfull[b, 1:, pair1] - CNAfull[b, :-1, pair1]
            diff1 = np.abs(diff1)
            diff1 = np.argwhere(diff1 > 0)[:, 0]

            CNAfull[b, diff1, pair1]

            diff1 = diff1 + 1

            CNAfull[b, diff1, pair1]



            diff1 = np.concatenate((diff1, start1))
            diff1 = np.unique(diff1)

            CNAfull[b, diff1, pair1]

            if doubleNow == 1:
                diff1 = diff1[ (CNAfull[b, diff1, pair1] % 2) == 0 ]




            if doubleNow == 1:
                diff2 = np.abs(CNAfull[b, diff1, pair1] - (2*currentCNA[b, diff1, pair1].data.numpy()) )

                #diff2 = diff2 * (  (CNAfull[b, diff1, pair1] + 1) % 2 ) #Commented Jun 3 2023
                diff2 = diff2[np.abs(diff2) != 1] #Added Jun 3 2023

            else:


                #CNAfull[b, diff1, pair1]


                diff2 = np.abs(CNAfull[b, diff1, pair1] - currentCNA[b, diff1, pair1].data.numpy() )

            diff2 = diff1[diff2 != 0]

            startBool[b, diff2 + (pair1 * CNAfull.shape[1]) ] = 1


    return startBool


def makeChrChoiceBool(chr, startChoice, start1, end1):

    chrBool = torch.zeros((startChoice.shape[0], chr.shape[0]))

    for b in range(startChoice.shape[0]):

        chr1 = chr[startChoice[b]]

        start2, end2 = start1[chr1], end1[chr1]

        chrBool[b, start2: end2] = 1

    return chrBool


def boolProbAdjuster(startProb, startBool):


    startProb2 = startProb * startBool
    startProbSum = torch.sum(startProb2, axis=1)
    startProbSum = startProbSum.reshape((-1, 1))

    #print (np.max(startProb2.data.numpy()), np.max(startProbSum.data.numpy())  )

    #print ( np.max(startProbSum.data.numpy())  )

    startProb2 = startProb2 / startProbSum


    #print ( np.max(startProb2.data.numpy())  )

    assert np.max(startProb2.data.numpy()) <= 1.01

    return startProb2


def boolProbSampler(startProb, startBool, isExtraBool=False, extraBool=False, firstChoice=False):

    #print (startProb)

    startProb = torch.softmax(startProb, axis=1)
    startProb = startProb + 1e-6 #Prevent 0
    startProb_sum = torch.sum(startProb, axis=1).reshape((-1, 1))
    startProb = startProb / startProb_sum

    #print (startProb)


    if isExtraBool:
        startProb = boolProbAdjuster(startProb, extraBool)

    startProb2 = boolProbAdjuster(startProb, startBool)

    startChoice = doChoice(startProb2.data.numpy())

    startChoice = startChoice.astype(int)

    startChoiceProb1 = startProb[np.arange(startChoice.shape[0]), startChoice]
    startChoiceProb2 = startProb2[np.arange(startChoice.shape[0]), startChoice]


    if firstChoice:

        #print ('bye')

        assert np.max(startProb2.data.numpy()) <= 1.01

        firstChoiceProb2 = startProb2[np.arange(startChoice.shape[0]), 0]

        assert np.max(firstChoiceProb2.data.numpy()) <= 1.01

        return startChoice, startChoiceProb1, startChoiceProb2, firstChoiceProb2
    else:

        #print ('hi')
        #if startBool[0, startChoice[0]] == 0:
        #    print ('boolIsuee')
        #    quit()

        return startChoice, startChoiceProb1, startChoiceProb2


def editCurrentCNA(currentCNA, argRun, startChoice, pairChoice, endChoice, callChoice, Ncall, CNAfull=False):


    callChoice_mod = np.copy(callChoice)
    callChoice_mod[callChoice_mod >= Ncall//2] = callChoice_mod[callChoice_mod >= Ncall//2] + 1
    callChoice_mod = callChoice_mod - (Ncall//2)

    for b in range(argRun.shape[0]):


        #error1 = currentCNA[b].data.numpy() - CNAfull[b]
        #error1 = np.sum(np.abs(error1), axis=1)
        #error1_0 = np.argwhere(error1!=0).shape[0]


        startChoice1 = startChoice[b]
        endChoice1 = endChoice[b]+1
        pairChoice1 = pairChoice[b]

        #print ("b")
        #print (startChoice1, endChoice1)

        #currentCNA[argRun[b], startChoice1:endChoice1, 0] = callChoice1[b]
        #currentCNA[argRun[b], startChoice1:endChoice1, 1] = callChoice2[b]

        #print (callChoice)
        #quit()

        currentCNA[argRun[b], startChoice1:endChoice1, pairChoice1] = currentCNA[argRun[b], startChoice1:endChoice1, pairChoice1] + callChoice_mod[b]



    currentCNA[currentCNA < 0] = 0
    currentCNA[currentCNA >= Ncall] = Ncall - 1

    return currentCNA



def updateStartBool(startBoolAll, CNAfull, currentCNA, argRun, startChoice, pairChoice, endChoice, callChoice, boolDouble, Ncall, includeWGD):

    Nbin = CNAfull.shape[1]

    sizeList = (endChoice + 1 - startChoice).astype(int)
    newEndPos = np.cumsum(sizeList).astype(int)
    newStartPos = newEndPos - sizeList
    sizeFull = newEndPos[-1]

    CNAfullPaste = np.zeros(sizeFull, dtype=int)
    currentCNAPaste = np.zeros(sizeFull, dtype=int)
    startBoolNew = np.zeros(sizeFull, dtype=int)
    indexPaste = np.zeros((sizeFull,), dtype=int)

    #time1 = time.time()

    for b in range(argRun.shape[0]):

        pairChoice1 = pairChoice[b]
        startChoice1 = startChoice[b]
        endChoice1 = endChoice[b]+1

        newStartPos1 = newStartPos[b]
        newEndPos1 = newEndPos[b]

        CNAfullPaste[newStartPos1:newEndPos1] = np.copy(CNAfull[b, startChoice1:endChoice1, pairChoice1])
        currentCNAPaste[newStartPos1:newEndPos1] = np.copy(currentCNA[b, startChoice1:endChoice1, pairChoice1].data.numpy())
        indexPaste[newStartPos1:newEndPos1] = b


    #print ((time.time() - time1))
    #time1 = time.time()



    diff1 = np.abs(CNAfullPaste[1:] - CNAfullPaste[:-1])
    diff1 = np.argwhere(diff1 != 0)[:, 0]
    diff1 = diff1 + 1
    diff1 = np.concatenate((diff1, newStartPos))

    #print ((time.time() - time1))
    #time1 = time.time()

    if includeWGD:
        doublePaste = boolDouble[indexPaste[diff1]]
        doubleMultiplier = doublePaste + 1
        diff2 = np.abs(CNAfullPaste[diff1] - (currentCNAPaste[diff1] * doubleMultiplier))

        #argDoubleIssue = np.argwhere(np.logical_and(doublePaste == 1  ,  CNAfullPaste[diff1] % 2 == 1 ))[:, 0] #Jun 3 2023
        #diff2[argDoubleIssue] = 0 #Jun 3 2023

        argDoubleIssue = np.argwhere(np.logical_and(doublePaste == 1  ,  diff2 <= 1 ))[:, 0]
        diff2[argDoubleIssue] = 0


    else:
        diff2 = np.abs(CNAfullPaste[diff1] - currentCNAPaste[diff1])
    diff1 = diff1[diff2 != 0]
    startBoolNew[diff1] = 1

    #print ((time.time() - time1))
    #time1 = time.time()


    for b in range(argRun.shape[0]):

        pairChoice1 = pairChoice[b]
        startChoice1 = startChoice[b]
        endChoice1 = endChoice[b]+1

        startChoice2 = startChoice1 + (Nbin * pairChoice1)
        endChoice2 = endChoice1 + (Nbin * pairChoice1)

        newStartPos1 = newStartPos[b]
        newEndPos1 = newEndPos[b]

        startBoolAll[argRun[b], startChoice2:endChoice2] = torch.tensor(np.copy(startBoolNew[newStartPos1:newEndPos1]))


    return startBoolAll


def findPossibleContinue(CNAfull, currentCNA, boolDoubleLeft, Ncall, includeWGD, withBAF):


    if includeWGD:
        argNoDouble = np.argwhere(boolDoubleLeft == 0)[:, 0]
        argDoubleLeft = np.argwhere(boolDoubleLeft == 1)[:, 0]


        error1 = np.zeros(CNAfull.shape[0], dtype=int)

        error2 = np.abs(CNAfull[argNoDouble] - (currentCNA[argNoDouble].data.numpy()) ).astype(int)
        error2[error2 != 0] = 1
        error2 = np.sum(error2, axis=(1, 2))


        error3 = np.abs(CNAfull[argDoubleLeft] - (2*currentCNA[argDoubleLeft].data.numpy()) ).astype(int)
        error3Bool = (CNAfull[argDoubleLeft] + 1) % 2
        error3 = error3 * error3Bool
        error3[error3 != 0] = 1
        error3 = np.sum(error3, axis=(1, 2))

        error1[argNoDouble] = error2
        error1[argDoubleLeft] = error3


    else:

        error1 = np.abs(CNAfull - currentCNA.data.numpy() ).astype(int)
        error1[error1 != 0] = 1
        error1 = np.sum(error1, axis=(1, 2))




    continueBool = torch.zeros((CNAfull.shape[0] , 4  ))


    argNoError = np.argwhere(error1 == 0)[:, 0]
    argWithError = np.argwhere(error1 != 0)[:, 0]

    #continueBool[argNoError, 1] = 0
    continueBool[argNoError, 0] = 1
    #continueBool[argWithError, 0] = 0
    continueBool[argWithError, 1] = 1

    if includeWGD:
        continueBool[argDoubleLeft, 2] = 1
        continueBool[argDoubleLeft, 0] = 0



    return continueBool



def fastFindPossibleEnd(CNAfull, currentCNA, startChoice, pairChoice, boolDouble, chr, start1, end1, Ncall, includeWGD, withBAF):



    chr1 = chr[startChoice]
    endChr = end1[chr1]

    sizeList = (endChr - startChoice).astype(int)
    newEndPos = np.cumsum(sizeList).astype(int)
    newStartPos = newEndPos - sizeList
    sizeFull = newEndPos[-1]

    CNAfullPaste = np.zeros(sizeFull, dtype=int)
    currentCNAPaste = np.zeros(sizeFull, dtype=int)

    indexPaste = np.zeros((sizeFull,), dtype=int)

    #time1 = time.time()

    for b in range(CNAfull.shape[0]):

        startChoice1 = startChoice[b]
        pairChoice1 = pairChoice[b]
        endChoice1 = endChr[b]

        newStartPos1 = newStartPos[b]
        newEndPos1 = newEndPos[b]

        #assert CNAfull[b, startChoice1, pairChoice1] != currentCNA[b, startChoice1, pairChoice1].data.numpy()

        CNAfullPaste[newStartPos1:newEndPos1] = np.copy(CNAfull[b, startChoice1:endChoice1, pairChoice1])
        currentCNAPaste[newStartPos1:newEndPos1] = np.copy(currentCNA[b, startChoice1:endChoice1, pairChoice1].data.numpy())
        indexPaste[newStartPos1:newEndPos1] = b




    #diff1 = np.sum(np.abs(CNAfullPaste[1:] - CNAfullPaste[:-1]), axis=1)
    diff1 = np.abs(CNAfullPaste[1:] - CNAfullPaste[:-1])
    diff1 = np.argwhere(diff1 != 0)[:, 0]
    diff1 = diff1 + 1
    diff1 = np.concatenate((diff1, newEndPos))

    diff1 = np.sort(diff1)


    if includeWGD:


        doublePaste = boolDouble[indexPaste]
        doubleMultiplier = doublePaste + 1
        diffSign = CNAfullPaste - (currentCNAPaste*doubleMultiplier)
        diffBool = np.abs(diffSign)
        diffBool[doublePaste == 1] = diffBool[doublePaste == 1] // 2

        diffSign = np.sign(diffSign) * diffBool

        diffBool[diffBool!=0] = 1
        diff1 = diff1[diffBool[diff1-1] != 0]


    else:

        #diffBool = np.sum(np.abs(CNAfullPaste - currentCNAPaste), axis=1)
        diffSign = CNAfullPaste - currentCNAPaste
        diffBool = np.abs(diffSign)
        diffBool[diffBool!=0] = 1
        diff1 = diff1[diffBool[diff1-1] != 0]



    diffSign[diffSign > 1] = 1
    diffSign[diffSign < -1] = -1
    diffSign = diffSign + 1

    startForDiff = newStartPos[indexPaste[diff1-1]]

    sameBool = 1 - diffBool
    sameBool_cumsum = np.cumsum(sameBool)
    sameBool_cumsum = np.concatenate(( np.zeros(1, dtype=int), sameBool_cumsum ))


    #callHist = np.zeros((sizeFull+1, Ncall*Ncall), dtype=int)
    #callHist[np.arange(CNAfullPaste.shape[0])+1, (CNAfullPaste[:, 0] * Ncall) + CNAfullPaste[:, 1] ] = 1 #+1 so the first is a zero.

    #callHist = np.zeros((sizeFull+1, Ncall), dtype=int) #Jun 3 2023
    callHist = np.zeros((sizeFull+1, 3), dtype=int) #Jun 3 2023
    if False: #Jun 3 2023
        if includeWGD:
            argDouble = np.argwhere(  np.logical_or( (CNAfullPaste % 2) == 0 , doublePaste == 0 ,   )  )[:, 0]
            callHist[argDouble+1, CNAfullPaste[argDouble] ] = 1
        else:
            callHist[np.arange(CNAfullPaste.shape[0])+1, CNAfullPaste ] = 1

    else:
        callHist[np.arange(CNAfullPaste.shape[0])+1, diffSign ] = 1
        #callHist[:, 1] = 0
        #callHist[np.arange(CNAfullPaste.shape[0])+1, CNAfullPaste ] = 1

    callHist_cumsum = np.cumsum(callHist, axis=0)


    histDiff = callHist_cumsum[diff1] - callHist_cumsum[startForDiff]
    maxCallNum = np.max(histDiff, axis=1)

    sameDiff = sameBool_cumsum[diff1] - sameBool_cumsum[startForDiff]


    argGoodChange = np.argwhere(maxCallNum > sameDiff)[:, 0]


    if False:
        issue1 = 471
        issues2 = np.argwhere(indexPaste[startForDiff] == 471)[:, 0]
        for issue1 in issues2:
            print ('')
            print (issue1 in argGoodChange)
            print ('CNAfull',  CNAfullPaste[startForDiff[issue1]:diff1[issue1]])
            print ('currentCNA', currentCNAPaste[startForDiff[issue1]:diff1[issue1]])
            print ('difference', CNAfullPaste[startForDiff[issue1]:diff1[issue1]] -   (currentCNAPaste[startForDiff[issue1]:diff1[issue1]] * 2)  )
            print (maxCallNum[issue1])
            print (sameDiff[issue1])
        #quit()
        for b in range(10):
            print ('')

    #b = 0
    #for b in range(argGoodChange.shape[0]):
    #    print ("A")
    #    print ('CNAfull',  CNAfullPaste[startForDiff[argGoodChange[b]]:diff1[argGoodChange[b]]])
    #    print ('currentCNA', currentCNAPaste[startForDiff[argGoodChange[b]]:diff1[argGoodChange[b]]])
    #    print ('doublePaste', doublePaste[startForDiff[argGoodChange[b]]:diff1[argGoodChange[b]]])
    #    print ('histDiff', histDiff[argGoodChange[b]])
    #    print ('sameDiff', sameDiff[argGoodChange[b]])
    #quit()





    diff1 = diff1[argGoodChange]

    endBoolNew = np.zeros((sizeFull,), dtype=int)
    endBoolNew[diff1-1] = 1 #-1 so its in the range of 0 to last value


    endBool = torch.zeros((CNAfull.shape[0], CNAfull.shape[1]))



    #print ("A")

    for b in range(CNAfull.shape[0]):

        #print (b)
        #print (endBoolNew[newStartPos[b]:newEndPos[b]])

        startChoice1 = startChoice[b]
        endChoice1 = endChr[b]

        newStartPos1 = newStartPos[b]
        newEndPos1 = newEndPos[b]

        #assert np.sum(endBoolNew[newStartPos1:newEndPos1]) > 0

        endBool[b, startChoice1:endChoice1] = torch.tensor(endBoolNew[newStartPos1:newEndPos1])

        if np.sum(endBoolNew[newStartPos1:newEndPos1]) == 0:

            #print (b)
            #print ('double', boolDouble[b])
            plt.plot(CNAfull[b, startChoice1:endChoice1, pairChoice1])
            plt.plot(currentCNA[b, startChoice1:endChoice1, pairChoice1].data.numpy()+0.1)
            plt.show()

    #print ('done1')
    #quit()
    return endBool


def fastFindPossibleCall(CNAfull, currentCNA, startChoice, pairChoice, endChoice, boolDouble, Ncall, includeWGD, info=False):



    assert np.min(endChoice + 1 - startChoice) > 0

    sizeList = (endChoice + 1 - startChoice).astype(int)
    newEndPos = np.cumsum(sizeList).astype(int)
    newStartPos = newEndPos - sizeList
    sizeFull = newEndPos[-1]

    CNAfullPaste = np.zeros(sizeFull, dtype=int)
    currentCNAPaste = np.zeros(sizeFull, dtype=int)

    indexPaste = np.zeros((sizeFull,), dtype=int)

    #plt.imshow(currentCNA.data.numpy())
    #plt.show()
    #quit()

    #time1 = time.time()

    #print ("B")
    #print (np.max(newEndPos))
    #print (sizeFull)
    #print (CNAfullPaste.shape)

    assert np.max(newEndPos) <= CNAfullPaste.shape[0]

    for b in range(CNAfull.shape[0]):

        startChoice1 = startChoice[b]
        pairChoice1 = pairChoice[b]
        endChoice1 = endChoice[b]+1

        newStartPos1 = newStartPos[b]
        newEndPos1 = newEndPos[b]

        assert newEndPos1 <= CNAfullPaste.shape[0]


        CNAfullPaste[newStartPos1:newEndPos1] = np.copy(CNAfull[b, startChoice1:endChoice1, pairChoice1])
        currentCNAPaste[newStartPos1:newEndPos1] = np.copy(currentCNA[b, startChoice1:endChoice1, pairChoice1].data.numpy())
        indexPaste[newStartPos1:newEndPos1] = b



    #diffBool = np.sum(np.abs(CNAfullPaste - currentCNAPaste), axis=1)

    if includeWGD:
        doublePaste = boolDouble[indexPaste]
        #diffBool = np.zeros(CNAfullPaste.shape[0], dtype=int)
        diffSign = np.zeros(CNAfullPaste.shape[0], dtype=int)

        argDouble = np.argwhere(   doublePaste == 1  )[:, 0]
        argNoDouble = np.argwhere(   doublePaste == 0  )[:, 0]
        #diffBool[argDouble] = np.abs(CNAfullPaste[argDouble] - (2 * currentCNAPaste[argDouble]) )
        #diffBool[argNoDouble] = np.abs(CNAfullPaste[argNoDouble] - currentCNAPaste[argNoDouble])
        diffSign[argDouble] = CNAfullPaste[argDouble] - (2 * currentCNAPaste[argDouble])
        diffSign[argNoDouble] = CNAfullPaste[argNoDouble] - currentCNAPaste[argNoDouble]

        diffSign0 = np.copy(diffSign)

        diffBool = np.abs(diffSign)
        diffBool[argDouble] = diffBool[argDouble] // 2
        diffSign = np.sign(diffSign) * diffBool

        #diffSign = diffSign + (Ncall // 2)
        #diffSign[diffSign < 0] = 0
        #diffSign[diffSign >= Ncall] = Ncall - 1





    #else:
    #    diffBool = np.abs(CNAfullPaste - currentCNAPaste)
    diffBool[diffBool!=0] = 1


    sameBool = 1 - diffBool
    sameBool = np.concatenate(( np.zeros(1, dtype=int), sameBool ))
    sameBool_cumsum = np.cumsum(sameBool)
    #####sameBool_cumsum = np.concatenate(( np.zeros(1, dtype=int), sameBool_cumsum ))




    #callHist_pos = np.zeros((sizeFull, Ncall // 2), dtype=int)
    #callHist_neg = np.zeros((sizeFull, Ncall // 2), dtype=int)
    #diffSign[diffSign >= Ncall // 2] = (Ncall // 2) - 1
    #diffSign[diffSign <= -Ncall // 2] = -(Ncall // 2) + 1
    #callHist_pos[diffSign > 0, diffSign[diffSign>0] - 1] = 1
    #callHist_neg[diffSign < 0, (diffSign[diffSign<0] * -1) - 1] = 1
    #callHist_pos = np.cumsum(callHist_pos[:, -1::-1], axis=1)[:, -1::-1]
    #callHist_neg = np.cumsum(callHist_neg[:, -1::-1], axis=1)
    #callHist = np.concatenate(( callHist_neg, callHist_pos ), axis=1)
    #callHist = np.concatenate((  np.zeros((1, Ncall)), callHist  ), axis=0)




    callHist = np.zeros((sizeFull+1, Ncall), dtype=int)
    if np.max(diffSign) > 0:
        argPos = np.argwhere(diffSign > 0)[:, 0]
        diffSign[diffSign >= Ncall // 2] = (Ncall // 2) - 1
        callHist[argPos+1, diffSign[argPos] + (Ncall // 2) - 1  ] = 1
    if np.min(diffSign) < 0:
        argNeg = np.argwhere(diffSign < 0)[:, 0]
        diffSign[diffSign <= -Ncall // 2] = (-Ncall // 2) + 1
        callHist[argNeg+1, diffSign[argNeg] + (Ncall // 2)  ] = 1




    callHist_cumsum = np.cumsum(callHist, axis=0)


    histDiff = callHist_cumsum[newEndPos] - callHist_cumsum[newStartPos]

    histDiff_pos = histDiff[:, Ncall // 2:]
    histDiff_neg = histDiff[:, :Ncall // 2]
    histDiff_pos = np.cumsum(histDiff_pos[:, -1::-1], axis=1)[:, -1::-1]
    histDiff_neg = np.cumsum(histDiff_neg, axis=1)
    histDiff_both = np.concatenate((histDiff_neg, histDiff_pos), axis=1)
    #histDiff[histDiff!=0] = 1
    #histDiff = histDiff * histDiff_both
    histDiff = histDiff_both


    sameDiff = sameBool_cumsum[newEndPos] - sameBool_cumsum[newStartPos]
    sameDiff = sameDiff.reshape((-1, 1))


    callBool = histDiff - sameDiff
    callBool[callBool>=1] = 1
    callBool[callBool<=0] = 0


    #sum1 = np.sum(callBool, axis=1)
    #argIssue = np.argwhere(sum1 == 0)[0, 0]

    #print ('info')
    #print (argIssue)
    #print (CNAfullPaste[newStartPos[argIssue]:newEndPos[argIssue]])
    #print (currentCNAPaste[newStartPos[argIssue]:newEndPos[argIssue]]*2)
    #print (CNAfullPaste[newStartPos[argIssue]:newEndPos[argIssue]] -  (currentCNAPaste[newStartPos[argIssue]:newEndPos[argIssue]]*2) )
    #print (doublePaste[argIssue])

    #5
    #


    #print ('samBool', sameBool[newStartPos[argIssue]+1:newEndPos[argIssue]+1])
    #print ('diffSign0', diffSign0[newStartPos[argIssue]:newEndPos[argIssue]])
    #print ('doublePaste', doublePaste[newStartPos[argIssue]:newEndPos[argIssue]])
    #print (histDiff[argIssue])
    #print (sameDiff[argIssue])


    #quit()

    if np.min(np.sum(callBool, axis=1)) <= 0.5:



        sum1 = np.sum(callBool, axis=1)
        argFail = np.argwhere(sum1 == 0)[0, 0]

        print ('egg')

        print ('fail', argFail)

        print (newStartPos[argFail] in argDouble)

        #print (currentCNAPaste[newStartPos[argFail]:newEndPos[argFail]]   )
        print (CNAfullPaste[newStartPos[argFail]:newEndPos[argFail]]   )

        print (callHist[newStartPos[argFail]:newEndPos[argFail]+1]   )
        print (sameBool[newStartPos[argFail]:newEndPos[argFail]+1]   )




        print ('double', doublePaste[argFail])

        #plt.imshow(callBool)
        #plt.show()
        quit()


    callBool = torch.tensor(callBool).float()


    return callBool




def findCurrentX(currentCNA_mini, Ncall, withBAF):

    currentCNA_mini = currentCNA_mini.data.numpy().astype(int)
    shape1 = currentCNA_mini.shape

    #print (shape1)
    #quit()

    currentCNA_mini = currentCNA_mini.reshape((shape1[0]*shape1[1], shape1[2]))
    currentX = torch.zeros((shape1[0] * shape1[1], shape1[2]*Ncall))
    currentX[np.arange(currentCNA_mini.shape[0]), currentCNA_mini[:, 0]] = 1
    if withBAF:
        currentX[np.arange(currentCNA_mini.shape[0]), currentCNA_mini[:, 1]+Ncall] = 1
    currentX = currentX.reshape((shape1[0], shape1[1], shape1[2]*Ncall))

    return currentX



def modelCNAgenerator(CNAfull, chr, start1, end1, model, Ncall, info, returnReg=False, doDouble=False):


    



    withBAF = False
    if CNAfull.shape[2] == 2:
        withBAF = True


    CNAfull[CNAfull >= Ncall] = Ncall - 1

    includeWGD = True

    #maxN = 500
    #maxN = 1000
    maxN = 1500

    Nbin = CNAfull.shape[1]

    modelProbStop = torch.zeros((maxN+1, CNAfull.shape[0]))
    modelProbSum = torch.zeros((maxN+1, CNAfull.shape[0]))
    sampleProbSum = torch.zeros((maxN+1, CNAfull.shape[0]))

    treeLoss = torch.zeros(CNAfull.shape[0])
    treeLoss_quad = torch.zeros(CNAfull.shape[0])


    currentCNA = torch.ones(CNAfull.shape)
    if not withBAF:
        currentCNA[:] = 2



    #CNA_mean0 = np.mean(np.sum(CNAfull, axis=2), axis=1) / 2
    CNA_mean0 = np.median(np.sum(CNAfull, axis=2), axis=1) / 2

    
    
    CNA_mean = np.round(CNA_mean0) - 1
    argDouble = np.argwhere(CNA_mean >= 1)[:, 0]
    boolDoubleLeft = np.zeros(CNA_mean.shape[0], dtype=int)
    boolDoubleLeft[argDouble] = 1

    #print (argDouble.shape)
    #quit()





    savedCNA = np.zeros((maxN+1, CNAfull.shape[0], Nbin, CNAfull.shape[2] ), dtype=int)-1
    savedCNA[0] = 1

    stepLast = np.zeros(CNAfull.shape[0] , dtype=int)

    done1 = False


    assert np.max(CNAfull) < Ncall


    doneBool = np.zeros(CNAfull.shape[0], dtype=int)



    #CNAcodeSize = np.sum(  (end1-start1)**2 )
    #CNAcodes = torch.zeros((CNAfull.shape[0], CNAcodeSize))

    #CNAused = np.zeros((CNAfull.shape[0], maxN, 2), dtype=int)

    boolPlot = np.zeros(CNAfull.shape[1])

    step = 0



    startBoolAll = findPossibleStart(CNAfull, currentCNA, start1, boolDoubleLeft, withBAF)

    continueBoolAll = torch.zeros((CNAfull.shape[0], 4))
    continueBoolAll[:, 1] = 1


    while not done1:

        #print ('step', step)

        if step == maxN - 1: #May 23 2023
            done1 = True #May 23 2023

        argRun = np.argwhere(doneBool == 0)[:, 0]

        if step % 50 == 0:
            print (argRun.shape)

        if False:#step == maxN - 1:

            print (argRun[0])

            print (CNAfull.shape)

            plt.plot(CNAfull[argRun[0]])
            plt.show()

            plt.plot(CNAfull[argRun[0], :, 0])
            plt.show()
            #quit()

        if (argRun.shape[0] > 0):

            #print ('argRun', argRun.shape)
            #quit()

            continueBool = findPossibleContinue(CNAfull[argRun], currentCNA[argRun], boolDoubleLeft[argRun], Ncall, includeWGD, withBAF)

            #if includeWGD:
            #    argDoubleLeft = boolDoubleLeft[argRun]
            #    argDoubleLeft = np.argwhere(argDoubleLeft == 1)[:, 0]
            #    continueBool[argDoubleLeft, 2] = 1


            doneBool[argRun] = np.copy(continueBool[:, 0])


            #argContinue = np.argwhere(doneBool[argRun] == 0)[:, 0]

            startBool = startBoolAll[argRun]
            #continueBool = continueBoolAll[argRun]




            currentX = findCurrentX(currentCNA[argRun], Ncall, withBAF)


            rep1 = model.embedder(currentX)

            continueProb = model.initial(rep1)

            #print (torch.softmax(continueProb, axis=1))

            #print ("A")
            initialChoice, initialChoiceProb1, initialChoiceProb2, stopChoiceProb2 = boolProbSampler(continueProb, continueBool, firstChoice=True)

            


            assert np.max( stopChoiceProb2.data.numpy()  ) <= 1.01



            argNotStop = np.argwhere(initialChoice != 0 )[:, 0]



            if includeWGD:
                #argWGD = np.argwhere( initialChoice == 2 )[:, 0]
                #argNormal = np.argwhere( initialChoice == 3 )[:, 0]
                argWGD = argRun[initialChoice == 2]
                argNormal = argRun[initialChoice == 3]

                #print (initialChoice)



                boolDoubleLeft[argWGD] = 0
                currentCNA[argWGD] = currentCNA[argWGD] * 2
                currentCNA[argNormal] = currentCNA[argNormal] + 1



                if np.max(currentCNA.data.numpy())  >= Ncall:
                    currentCNA[currentCNA.data.numpy() >= Ncall] = Ncall - 1

                argBoth = argRun[np.isin(initialChoice, np.array([2, 3])  )]

                startBoolAll[argBoth] = findPossibleStart(CNAfull[argBoth], currentCNA[argBoth], start1, np.zeros(argBoth.shape[0], dtype=int), withBAF )



                savedCNA[step+1, argBoth] = np.copy(currentCNA[argBoth]) #AdDED May 25 2023
                stepLast[argBoth] = step+1 #AdDED May 25 2023



            argContinue = np.argwhere(initialChoice == 1 )[:, 0]


            if argContinue.shape[0] > 0:

                startBool = startBoolAll[argRun[argContinue]][:, :-1]
                rep1 = rep1[argContinue]
                boolDoubleLeftNow = boolDoubleLeft[argRun[argContinue]]

                startProb = model.starter(rep1)[:, :-1]

                treeLoss[argRun] = treeLoss[argRun] + torch.sum( -1 * torch.softmax(startProb, axis=1) * torch.log(torch.softmax(startProb, axis=1) + 1e-10 ))
                #treeLoss[argRun] = treeLoss[argRun] + torch.sum( -1 * torch.softmax(startProb, axis=1) * torch.log(  (1-torch.softmax(startProb, axis=1)) + 1e-5 ))
                treeLoss_quad[argRun] = treeLoss_quad[argRun] + treeLoss[argRun]
                



                #print ("A")
                #print (startProb.shape)
                #print (startBool.shape)


                #print ("B")
                #posChoice, startChoiceProb1, startChoiceProb2, lastChoiceProb2 = boolProbSampler(startProb, startBool, lastChoice=True)
                posChoice, startChoiceProb1, startChoiceProb2 = boolProbSampler(startProb, startBool)


                startChoice = posChoice % Nbin
                pairChoice = posChoice // Nbin

                #if startChoice[0] == 128:
                #    print (CNAfull[])


                arange1 = np.arange(startChoice.shape[0])
                assert np.min(np.sum(startBool.data.numpy(), axis=1)   ) > 0
                if np.min(startBool[arange1, startChoice + (Nbin * pairChoice)  ].data.numpy()   ) <= 0:
                    failCase = np.argwhere(    startBool[arange1, startChoice + (Nbin * pairChoice)  ].data.numpy() <= 0 )[:, 0]

                    print (posChoice[failCase])
                    print ("start prob in fail case")
                    for c in range(startProb.shape[1]):
                        print (startProb[failCase[0], c])



                #assert np.min(startBool[arange1, startChoice + (Nbin * pairChoice)  ].data.numpy()   ) > 0
                #assert np.min(np.abs(CNAfull[argRun[argContinue], startChoice, pairChoice] - currentCNA[argRun[argContinue], startChoice, pairChoice ].data.numpy()   )) > 0

                startChoiceBool = torch.zeros((argContinue.shape[0], CNAfull.shape[1]*2))
                startChoiceBool[np.arange(argContinue.shape[0]), posChoice] = 1

                #print (time.time() - time1)
                #time1 = time.time()

                endBool = fastFindPossibleEnd(CNAfull[argRun[argContinue]], currentCNA[argRun[argContinue]], startChoice, pairChoice, boolDoubleLeftNow, chr, start1, end1, Ncall, includeWGD, withBAF)

                assert np.min(np.sum( endBool.data.numpy() , axis=1 )) > 0

                extraBool = np.zeros((argContinue.shape[0], CNAfull.shape[1]))
                for b in range(argContinue.shape[0]):
                    chrChoice = chr[startChoice[b]]
                    extraBool[b, start1[chrChoice]:end1[chrChoice]] = 1
                extraBool = torch.tensor(extraBool).float()


                #endProb = modelEnd(rep1[argContinue], startChoiceBool)
                endProb = model.ender(rep1, startChoiceBool)

                #print ("C")
                endChoice, endChoiceProb1, endChoiceProb2 = boolProbSampler(endProb, endBool, isExtraBool=True, extraBool=extraBool)


                argIssue = np.argwhere( endChoice+1-startChoice <= 0  )[:, 0]
                if argIssue.shape[0] > 0:

                    endSum = np.sum( endBool.data.numpy() , axis=1 )

                    print ('issue')
                    plt.plot(endSum)
                    plt.show()

                    print (endBool.shape)
                    quit()
                    print (endChoice[argIssue])
                    print (startChoice[argContinue[argIssue]])
                    quit()

                assert np.min(endChoice+1-startChoice) > 0

                #CNAused[argRun[argContinue]]
                #CNAused[argRun[argContinue], step]
                #CNAused[argRun[argContinue], step, 0]
                #startChoice[argContinue]

                #CNAused[argRun[argContinue], step, 0] = startChoice
                #CNAused[argRun[argContinue], step, 1] = endChoice + 1


                endChoiceBool = torch.zeros((endChoice.shape[0], CNAfull.shape[1]))
                endChoiceBool[np.arange(endChoice.shape[0]), endChoice] = 1


                #print (time.time() - time1)
                #time1 = time.time()

                #copyCallProb = modelCNA(rep1[argContinue], startChoiceBool, endChoiceBool)
                copyCallProb = model.caller(rep1, startChoiceBool, endChoiceBool)


                #print (time.time() - time1)
                #time1 = time.time()


                #copyCallBool_original = findPossibleCall(CNAfull[argRun], currentCNA[argRun], startChoice, endChoice, Ncall)
                copyCallBool = fastFindPossibleCall(CNAfull[argRun[argContinue]], currentCNA[argRun[argContinue]], startChoice, pairChoice, endChoice, boolDoubleLeftNow, Ncall, includeWGD)


                #assert torch.sum(torch.abs(copyCallBool_original - copyCallBool)) == 0



                #print (copyCallBool.shape)
                #quit()

                #print ("D")
                #print (np.min(np.sum(copyCallBool.data.numpy(), axis=1)))
                callChoice, callChoiceProb1, callChoiceProb2 = boolProbSampler(copyCallProb, copyCallBool)


                if False:#0 in argRun[argContinue]:
                    #print (copyCallBool[0, callChoice[0]])
                    plt.plot(CNAfull[0, :, 0] - 0.05)
                    plt.plot(currentCNA[0, :, 0].data.numpy())

                currentCNA = editCurrentCNA(currentCNA, argRun[argContinue], startChoice, pairChoice, endChoice, callChoice, Ncall)


                if False:#0 in argRun[argContinue]:
                    print (callChoice[0] - (Ncall // 2))
                    print (copyCallBool[0, callChoice[0]])
                    #print (copyCallProb[0])
                    #print (copyCallBool[0])
                    plt.plot(currentCNA[0, :, 0].data.numpy())
                    plt.plot(chr % 2, color='grey', alpha=0.5)
                    plt.show()

                #plt.imshow(copyCallBool[:20].data.numpy())
                #plt.show()


                savedCNA[step+1, argRun[argContinue]] = np.copy(currentCNA[argRun[argContinue]])
                stepLast[argRun[argContinue]] = step+1




                startBoolAll = updateStartBool(startBoolAll, CNAfull[argRun[argContinue]], currentCNA[argRun[argContinue]], argRun[argContinue], startChoice, pairChoice, endChoice, callChoice, boolDoubleLeftNow, Ncall, includeWGD)

                #print ("Got Here")
                #quit()


                #argSelect = np.argwhere(startBoolAll[:, :-1].data.numpy() == 1)
                #if argSelect.shape[0] > 0:
                #    assert np.min(np.abs(CNAfull[argSelect[:, 0], argSelect[:, 1] % Nbin  , argSelect[:, 1] // Nbin   ] - currentCNA[argSelect[:, 0], argSelect[:, 1] % Nbin  , argSelect[:, 1] // Nbin   ].data.numpy()   )) > 0



            adj = 1e-4




            #modelProbStep = torch.log(startChoiceProb1+adj) #This one includes stop prob
            modelProbStep = torch.zeros(argRun.shape[0])
            #print (argContinue.shape)
            #print (startChoiceProb1.shape)

            #print ("A")
            #print (continueProb)
            #print (initialChoiceProb1, initialChoiceProb2)
            #if step == 10:
            #    quit()
            
            if argNotStop.shape[0] > 0:
                modelProbStep[argNotStop] = modelProbStep[argNotStop] + adjLog(initialChoiceProb1[argNotStop], adj)
            if argContinue.shape[0] > 0:
                #modelProbStep[argContinue] = adjLog(startChoiceProb1, adj)
                modelProbStep[argContinue] = modelProbStep[argContinue] + adjLog(startChoiceProb1, adj) #Sep 2 2023 Attempt
                modelProbStep[argContinue] = modelProbStep[argContinue] + adjLog(endChoiceProb1, adj) + adjLog(callChoiceProb1, adj)


            #sampleProbStep = torch.log(startChoiceProb2+adj) #This one includes stop prob
            sampleProbStep = torch.zeros(argRun.shape[0])
            if argNotStop.shape[0] > 0:
                sampleProbStep[argNotStop] = sampleProbStep[argNotStop] + adjLog(initialChoiceProb2[argNotStop], adj)
            if argContinue.shape[0] > 0:
                #sampleProbStep[argContinue] = adjLog(startChoiceProb2, adj) #Sep 2 2023 Attempt
                sampleProbStep[argContinue] = sampleProbStep[argContinue] + adjLog(startChoiceProb2, adj)
                sampleProbStep[argContinue] = sampleProbStep[argContinue] + adjLog(endChoiceProb2, adj) + adjLog(callChoiceProb2, adj)
            sampleProbStep = sampleProbStep.detach()


            #modelProbSum[argRun] = modelProbSum[argRun] + modelProbStep
            #sampleProbSum[argRun] = sampleProbSum[argRun] + sampleProbStep



            modelProbSum[step+1, argRun] = modelProbStep
            sampleProbSum[step+1, argRun] = sampleProbStep
            modelProbStop[step, argRun] = adjLog(stopChoiceProb2, adj)  # lastChoiceProb2 #not step+1 cuz this is stopping before new CNA

            assert np.max( stopChoiceProb2.data.numpy()  ) <= 1.01

            assert np.max(adjLog(stopChoiceProb2, adj).data.numpy() <= 0.01  )
            assert np.max(modelProbSum.data.numpy()) <= 0.01
            assert np.max(modelProbStop.data.numpy()) <= 0.01



            step += 1

            #print (np.argwhere(stepLast >= 1).shape)
            #print (np.min(savedCNA[1, stepLast >= 1]))
            #if step == 2:
            #    quit()


        else:

            done1 = True




    #print ("Got Here")

    #for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(
    #                                locals().items())), key= lambda x: -x[1])[:10]:
    #            print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
    #quit()

    #modelProbSum = torch.sum(modelProbSum, axis=0)
    #sampleProbSum = np.sum(sampleProbSum, axis=0)

    savedCNA = savedCNA[:(step+2)]
    modelProbSum = modelProbSum[:(step+2)]
    sampleProbSum = sampleProbSum[:(step+2)]
    modelProbStop = modelProbStop[:(step+2)]

    

    modelProbSum = torch.cumsum(modelProbSum, axis=0)
    sampleProbSum = torch.cumsum(sampleProbSum, axis=0).data.numpy()

    modelProbSum = modelProbSum + modelProbStop

    
    argValid = np.argwhere( savedCNA[:, :, 0, 0] != -1 )


    savedCNA_mapper = np.zeros( (savedCNA.shape[0], savedCNA.shape[1]), dtype=int )
    savedCNA_mapper[argValid[:, 0], argValid[:, 1]] = np.arange(argValid.shape[0])
    savedCNA = savedCNA[argValid[:, 0], argValid[:, 1]]




    CNAused = ''

    if returnReg:
        return modelProbSum, sampleProbSum, treeLoss_quad, argValid, savedCNA_mapper, savedCNA, stepLast
    else:
        return modelProbSum, sampleProbSum






def estimateCNAprob(CNAfull, N, chr, start1, end1, model, modelStart, modelEnd, modelCNA, Ncall, info):

    N0 = CNAfull.shape[0]

    arange1 = np.arange(CNAfull.shape[0]*N) % N0 #// N
    arange2 = np.arange(CNAfull.shape[0]*N)

    correctRateFull = np.zeros(arange1.shape[0])

    batchSize = 500

    Niter = ((arange1.shape[0]-1) // batchSize) + 1

    #print (Niter)
    #print (batchSize)
    #print (N0*N)
    #print (correctRateFull.shape)
    #quit()

    for a in range(Niter):

        print (a, Niter)

        if a == Niter - 1:
            args1 = arange1[batchSize*a:]
            args2 = arange2[batchSize*a:]
        else:
            args1 = arange1[batchSize*a:batchSize*(a+1)]
            args2 = arange2[batchSize*a:batchSize*(a+1)]

        CNA_batch = CNAfull[args1]

        modelProbSum, sampleProbSum = modelCNAgenerator(CNA_batch, chr, start1, end1, model, modelStart, modelEnd, modelCNA, Ncall, info, doDouble=True)
        correctRate = modelProbSum.data.numpy() - sampleProbSum

        #print ("B")

        correctRateFull[args2] = correctRate
        #correctRateFull[args2] = args1

        #print (correctRate)
        #quit()



    correctRateFull = correctRateFull.reshape((N, N0))

    #print (correctRateFull[:, 0])
    #quit()

    #plt.plot(CNAfull[3, :, 0])
    #plt.plot(CNAfull[3, :, 1])
    #plt.plot(CNAfull[9, :, 0]+0.1)
    #plt.plot(CNAfull[9, :, 1]+0.1)
    #plt.show()

    correctRateFull = torch.tensor(correctRateFull).float()

    correctRate1 =  torch.logsumexp(correctRateFull, dim=0)
    correctRate1 = correctRate1.data.numpy() - np.log(N)


    #print (correctRate1)
    #quit()
    #print (N0)
    #print (correctRate1.shape)

    return correctRate1



def givePredMeasure(CNAprofiles):

    CNAprofiles = CNAprofiles.astype(float)
    pred_RDR = CNAprofiles[:, :, 0] + CNAprofiles[:, :, 1]
    pred_BAF = np.min(CNAprofiles, axis=2) / (pred_RDR + 1e-5)
    pred_BAF = mapBAF(pred_BAF)
    for a in range(pred_RDR.shape[0]):
        pred_RDR[a] = pred_RDR[a] / np.abs(np.mean(pred_RDR[a]))

    return pred_RDR, pred_BAF


def normedFFT(X):

    #norm1 = np.sum( np.abs(X[0]) ** 2 )

    X = np.fft.fft(X, axis=1) / ( float(X.shape[1]) ** 0.5 )

    #print (np.sum( np.abs(X[0]) ** 2 ))

    X2 = np.zeros((X.shape[0], X.shape[1], 2))
    X2[:, :, 0] = np.copy(np.real(X))
    X2[:, :, 1] = np.copy(np.imag(X))
    return X2


def subset_calculateError(pred_RDR, pred_BAF, RDR, HAP, chr, boolSubset, noiseRDR, noiseBAF, withAdjust):



    assert torch.min(pred_BAF) >= 0


    def func1(x):

        #return (x ** 2) / (0.25 + torch.abs(x))
        return (x**2)

    #print (np.min(noiseRDR))
    #plt.imshow(noiseRDR)
    #plt.show()


    #noise1 = noiseRDR.reshape((noiseRDR.shape[0]*noiseRDR.shape[1],))
    epsilon = np.mean(np.abs(noiseRDR)) * 0.001
    noiseInverse = 1 / (noiseRDR + epsilon)
    #noiseInverse = noiseInverse / np.mean(noiseInverse)

    epsilonBAF = np.mean(np.abs(noiseBAF)) * 0.001
    noiseBAFInverse = 1 / (noiseBAF + epsilonBAF)


    
    noiseInverse = torch.tensor(noiseInverse).float()
    noiseBAFInverse = torch.tensor(noiseBAFInverse).float()


    
    errorMatrix = torch.zeros((pred_RDR.shape[0], RDR.shape[0]))



    #errorMatrix_RDR = np.zeros((pred_RDR.shape[0], RDR.shape[0]))
    #errorMatrix_BAF = np.zeros((pred_RDR.shape[0], RDR.shape[0]))



    #RDR_int = RDR * 100
    #BAF_int = BAF * 100
    #pred_RDR = pred_RDR * 100
    #pred_BAF = pred_BAF * 100

    if False:#not withAdjust:
        RDR_int = np.floor(RDR*100).astype(int)
        pred_RDR = torch.floor(pred_RDR*100).int()
        if type(BAF) != type(''):
            BAF_int = np.floor(BAF*100).astype(int)
            pred_BAF = torch.floor(pred_BAF*100).int()

        RDR_int = torch.tensor(RDR_int).int()
        if type(BAF) != type(''):
            BAF_int = torch.tensor(BAF_int).int()
    else:
        #RDR_int, BAF_int = RDR, BAF

        RDR_int = torch.tensor(RDR).float()
        

        

        #RDR_int = torch.tensor(RDR_int).float()
        if type(HAP) != type(''):
            HAP = torch.tensor(HAP).float()



    #print (  np.sum(  np.abs(pred_RDR[0] - RDR[0]) ** 2)   )

    #pred_RDR1 = pred_RDR[0].reshape((1, pred_RDR.shape[1], 2))
    #print (  np.sum(  np.abs(pred_RDR1 - RDR) ** 2)   )
    #quit()


    #pred_BAF = np.floor(pred_RDR*100).astype(int)

    size1 = pred_RDR.shape[1]
    weight1 = np.array([np.arange(size1)+1, size1-np.arange(size1)])
    weight1 = np.min(weight1, axis=0)
    weight1 = (1 / weight1.astype(float))





    doSquare = True


    for a in range(pred_RDR.shape[0]):#[821, 951, 935, 944]:

        #print (a, pred_RDR.shape)

        #plt.plot(pred_RDR[a].data.numpy())
        #plt.show()

        #print (pred_RDR.shape)
        pred_RDR1 = pred_RDR[a].reshape((1, pred_RDR.shape[1]))

        if type(HAP) != type(''):
            pred_BAF1 = pred_BAF[a].reshape((1, pred_BAF.shape[1]))

            #pred_BAF1 = (pred_BAF1 * 0.95) + 0.025
            pred_BAF1 = tweakBAF(pred_BAF1)
            #pred_BAF1 = pred_BAF[a]


        boolSubset1 = boolSubset[a]
        subset1 = np.argwhere(boolSubset1 == 1)[:, 0]
        subset2 = np.argwhere(boolSubset1 == 0)[:, 0]
        


        #Trying renorm
        mean2 = torch.mean(RDR_int[subset1] * noiseInverse[subset1] , axis=1 ) / torch.mean(noiseInverse[subset1], axis=1)
        mean3 = torch.mean(pred_RDR1 * noiseInverse[subset1] , axis=1 ) / torch.mean(noiseInverse[subset1], axis=1)
        ratio1 = mean2 / mean3

        pred_RDR1 = pred_RDR1 * ratio1.reshape((  ratio1.shape[0], 1 ))


        #for a in range(pred_RDR1.shape[0]):
        #    plt.plot(pred_RDR1[a].data.numpy())
        #    plt.plot(RDR_int[a].data.numpy())
        #    plt.show()
        #quit()

        #if a == 0:
        #    print ('mean1', torch.mean(torch.abs(mean2-1) ))

        #RDR_error = torch.sum(  ((pred_RDR1 - RDR_int[subset1]) * noiseInverse[subset1]) ** 2 , axis=1) + 1e-5# + 1 #+1 to avoid 0
        RDR_error = torch.sum(  func1((pred_RDR1 - RDR_int[subset1]) * noiseInverse[subset1])  , axis=1) + 1e-5# + 1 #+1 to avoid 0



        #RDR_error2 = torch.sum(  func1((pred_RDR1 - RDR_int[subset1]) )  , axis=1).data.numpy()
        #argMin1 = np.argwhere(RDR_error2 == np.min(RDR_error2) )[0, 0]
        #diff1 = pred_RDR[a].data.numpy() - RDR[argMin1]
        #noise1 = noiseInverse[argMin1].data.numpy()
        #125 bad

        #print (noise1[125])
        
        #plt.plot(diff1 * noise1)
        #plt.show()


        #print (RDR_error.shape)
        #plt.plot(pred_RDR[a].data.numpy())
        #plt.plot(RDR[argMin1])
        #plt.show()
        #quit()

        #print (torch.min(torch.abs(noiseInverse[subset1])))

        #print (torch.min(torch.abs((pred_RDR1))))

        #print (torch.min(torch.abs((RDR_int[subset1]))))

        #print (torch.min(torch.abs((pred_RDR1 - RDR_int[subset1]))))

        #print (torch.min(torch.abs((pred_RDR1 - RDR_int[subset1]) * noiseInverse[subset1])))


        #print (torch.min(RDR_error))

        #print (pred_RDR1.shape)
        #print (RDR_int[subset1].shape)
        #print (noiseInverse[subset1].shape)
        #print (RDR_error[:10])
        #quit()


        if type(HAP) != type(''):
            
            #BAF_error = ( torch.log(pred_BAF1) * HAP[subset1, :, 1]  ) + ( torch.log(1 - pred_BAF1) * HAP[subset1, :, 0]  )
            #optimalBAF = (HAP[subset1, :, 1]) / ( torch.sum(HAP[subset1], axis=2) + 1e-5 )
            #optimalBAF = (optimalBAF * 0.98) + 0.01
            #minimalError = ( torch.log(optimalBAF) * HAP[subset1, :, 1]  ) + ( torch.log(1 - optimalBAF) * HAP[subset1, :, 0]  )
            #BAF_error = minimalError - BAF_error
            #BAF_error = torch.sum(BAF_error, axis=1)
            

            if True:
                BAF_int = HAP[subset1, :, 1].float() / (torch.sum(HAP[subset1], axis=2).float() + 1e-5)
                BAF_error = torch.sum(  func1((pred_BAF1 - BAF_int) * noiseBAFInverse[subset1])  , axis=1) + 1e-5
            else: # Old method prior to FEB 7 2024 . TODO potentionally remove the unneeded code. 
                #std2 = std2 + (0.25 / weight_sum)
                weight_sum = 0.25 / noiseBAFInverse[subset1]
                weight_sum_rel = weight_sum / (torch.sum(HAP[subset1], axis=2) + 1e-5)
                weight_sum_rel[weight_sum_rel>1] = 1
                BAF_error = ( torch.log(pred_BAF1) * HAP[subset1, :, 1] * weight_sum_rel  ) + ( torch.log(1 - pred_BAF1) * HAP[subset1, :, 0] * weight_sum_rel  )

            
                BAF_error = torch.sum(BAF_error, axis=1)
                

            

            #print ( type(BAF_error))

            #print (BAF_error.shape)

            #BAF_error = BAF_error * -1

            
            #print (BAF_error.shape)
            #BAF_error = torch.sum(  (pred_BAF1 - BAF_int[subset1]) ** 2, axis=1) + 1e-5#1 #+1 to avoid 0
            #BAF_error = torch.sum(  func1(pred_BAF1 - BAF_int[subset1]) , axis=1) + 1e-5#1 #+1 to avoid 0
            #BAF_error = BAF_error.float() / float(RDR.shape[1])
            #if not withAdjust:
            #    BAF_error = BAF_error # / float(100**2)

        #print ('b')
        #ar1 = ((pred_RDR1 - RDR_int[subset1]) * noiseInverse[subset1])
        #print (torch.min( ar1  ** 2 ))
        #print (torch.min(RDR_error))
        #print (torch.max(RDR_error))


        RDR_error = RDR_error.float() #/ float(RDR.shape[1])
        #if not withAdjust:
        #    RDR_error = RDR_error / float(100**2)


        if np.isnan(RDR_error.data.numpy()).any():
            print ("RDR_error nan error")
            quit()
        
        if np.isnan(BAF_error.data.numpy()).any():
            print ("BAF_error nan error")
            quit()
        

        #print (torch.min(RDR_error))


        #prob_error = torch.log(RDR_error)
        prob_error = RDR_error
        if type(HAP) != type(''):
            

            prob_error = prob_error + BAF_error

        


        assert np.max(prob_error.data.numpy()) < 1e10


        errorMatrix[a, subset1] = prob_error
        errorMatrix[a, subset2] = 1e12#100000000

        if np.isnan(prob_error.data.numpy()).any():
            print ("is nan error")
            quit()



    
    return errorMatrix



def calculateError(CNAprofiles, RDR, BAF, chr, noiseRDR, model, withAdjust):


    doBAF = True
    if BAF == '':
        doBAF = False

    predRDR, predBAF = measurementCalculator(CNAprofiles, model, doBAF)


    boolSubset = np.ones((CNAprofiles.shape[0], RDR.shape[0]), dtype=int)


    errorMatrix = subset_calculateError(predRDR, predBAF, RDR, BAF, chr, boolSubset, noiseRDR, withAdjust)

    return errorMatrix




def efficiencyCalculateError(predRDR, predBAF, savedCNA_mapper, RDR, HAP, stepLast, modelProbSum, sampleProbSum, errorNow, chr, noiseRDR, noiseBAF, balance, withAdjust, giveError=False, doScales=False, scalesBool=False):


    def getCellWeight(div2, errorMatrix):

        div1 = div1.reshape((div1.shape[0], div1.shape[1], 1))

        mult1 = div1 - errorMatrix
        mult1 = torch.tensor(mult1).float()
        sum1 = torch.logsumexp(mult1, dim=(0, 1))

        cellProb = sum1.data.numpy()

        sum1 = sum1.reshape((1, 1, sum1.shape[0]))

        weight1 = mult1 - sum1
        weight1 = torch.logsumexp(weight1, dim=2)
        weight1 = torch.exp(weight1)

        return weight1



    #weightFull = np.zeros((modelProbSum.shape[0], modelProbSum.shape[1], RDR.shape[0]))
    weightUsed = np.zeros(modelProbSum.shape, dtype=int)

    #print (RDR)
    modelProbSum.shape[1]
    RDR.shape[0]

    multPos = np.zeros(( 20 * modelProbSum.shape[1], 2 ), dtype=int)
    multPaste = np.zeros(( 20 * modelProbSum.shape[1], RDR.shape[0] ))
    errorPaste = np.zeros(( 20 * modelProbSum.shape[1], RDR.shape[0] ))


    if giveError:
        errorFull = torch.zeros((modelProbSum.shape[0], modelProbSum.shape[1], RDR.shape[0]))
        errorFull[:] = 1e15
    else:
        errorFull = ''




    div1 = modelProbSum - sampleProbSum
    div1 = div1.reshape((div1.shape[0], div1.shape[1], 1))
    div1 = torch.tensor(div1).float()

    stepNow = np.copy(stepLast).astype(int)

    if doScales:
        errorNow[scalesBool==1] = 100000000


    if withAdjust:
        boolSubset = np.ones((savedCNA_mapper.shape[1], RDR.shape[0]), dtype=int)
        arange1 = np.arange(stepNow.shape[0]).astype(int)

        predBAF_now = ''
        if type(predBAF) != type(''):
            predBAF_now = predBAF[ savedCNA_mapper[stepNow, arange1]  ]

        #print ('hi')
        #print (predRDR[stepNow, arange1].shape)
        #print (arange1.shape)
        #print (np.max(stepNow))
        #print (predRDR.shape)
        #print (boolSubset.shape)

        #errorNow_copy = np.copy(errorNow.data.numpy())

        #print (errorNow.shape)
        #print (boolSubset.shape)

        #plt.imshow( predRDR[stepNow, arange1].data.numpy()  )
        #plt.show()
        #quit()

        #quit()
        #quit()
        errorNow = subset_calculateError(predRDR[ savedCNA_mapper[stepNow, arange1] ], predBAF_now, RDR, HAP, chr, boolSubset, noiseRDR, noiseBAF, withAdjust)
        #errorNow = errorNow.data.numpy()

        #errorNow_np = errorNow.data.numpy()

        #print (np.min(errorNow_copy), np.max(errorNow_copy))
        #print (np.min(errorNow_np), np.max(errorNow_np))

        #plt.imshow(errorNow_np - errorNow_copy)
        #plt.show()

        #errorNow = torch.tensor(errorNow_copy).float()

        #plt.imshow(errorNow_np)
        #plt.show()



    argCheck = np.arange(modelProbSum.shape[1])
    cellProb = np.zeros(RDR.shape[0])

    cellProbFull = torch.zeros((modelProbSum.shape[0], RDR.shape[0])).float()
    cellProbFull[:] = -1 * 1e13

    count1 = 0

    a = 0
    while argCheck.shape[0] > 0:

        if giveError:
            #errorFull[stepNow[argCheck], argCheck] = np.copy(errorNow)
            errorFull[stepNow[argCheck], argCheck] = errorNow

        #print (div2.shape, errorNow.shape, RDR.shape)

        div2 = div1[stepNow[argCheck], argCheck]
        #mult1 = (div2 * balance) - errorNow

        mult1 = div2 - (errorNow / float(balance))

        

        


        #print (multPos.shape, count1)

        count2 = count1 + argCheck.shape[0]

        if count2 >= (multPos.shape[0] - 1):
            multPos0 = np.zeros(( 20 * modelProbSum.shape[1], 2 ), dtype=int)
            multPaste0 = np.zeros(( 20 * modelProbSum.shape[1], RDR.shape[0] ))
            errorPaste0 = np.zeros(( 20 * modelProbSum.shape[1], RDR.shape[0] ))
            multPos = np.concatenate((multPos, multPos0), axis=0)
            multPaste = np.concatenate((multPaste, multPaste0), axis=0)
            errorPaste = np.concatenate((errorPaste, errorPaste0), axis=0)
            del multPos0
            del multPaste0
            del errorPaste0


        #print (mult1.shape)
        #quit()

        multPos[count1:count2, 0] = np.copy(stepNow[argCheck])
        multPos[count1:count2, 1] = np.copy(argCheck)
        multPaste[count1:count2] = np.copy(mult1.data.numpy())
        errorPaste[count1:count2] = np.copy(errorNow.data.numpy())
        count1 = count2


        #plt.plot(np.mean(mult1.data.numpy(), axis=1))
        #plt.show()
        #quit()



        #weightFull[stepNow[argCheck], argCheck] = np.copy(mult1)
        weightUsed[stepNow[argCheck], argCheck] = 1

        #print (np.max(np.exp(weightFull)))

        sum1 = torch.logsumexp(mult1, axis=0)


       

        cellProbFull[a] = sum1

        if a == 0:
            cellProb = sum1.data.numpy()
        else:
            cellProb = np.concatenate((  cellProb.reshape((-1, 1))  , sum1.reshape((-1, 1)).data.numpy() ), axis=1 )
            cellProb = logsumexp(cellProb, axis=1)


        #print (np.mean(cellProb))


        cellProb_reshape = cellProb.reshape((1, -1))
        weight1 = mult1.data.numpy() - cellProb_reshape


        #plt.plot(np.sum(np.exp(weight1), axis=0))
        #plt.show()
        #quit()

        cutOff = -1 * np.log(100000)

        boolSubset = np.zeros(weight1.shape, dtype=int)
        boolSubset[weight1 > cutOff] = 1

        #plt.imshow(weight1)
        #plt.show()
        #quit()

        boolSubsetSum = np.sum(boolSubset, axis=1)
        argCheck = argCheck[boolSubsetSum >= 1]
        boolSubset = boolSubset[boolSubsetSum >= 1]

        #print ('sum1', np.sum(boolSubset))

        stepNow[argCheck] = stepNow[argCheck] - 1

        #print ('sum2', np.sum(boolSubset[argCheck]))


        #boolSubset[:] = 1 #TEMPORARY!!! May 25 2023

        boolSubset = boolSubset[stepNow[argCheck] >= 0]
        argCheck = argCheck[stepNow[argCheck] >= 0]



        print ('argCheck', argCheck.shape, np.sum(boolSubset))

        if argCheck.shape[0] > 0:


            #errorNow = subset_calculateError(savedCNA[stepNow[argCheck], argCheck], RDR, BAF, chr, boolSubset, noiseRDR)

            predRDR_now = predRDR[  savedCNA_mapper[stepNow[argCheck], argCheck] ]

            
            predBAF_now = ''
            if type(HAP) != type(''):
                predBAF_now = predBAF[  savedCNA_mapper[stepNow[argCheck], argCheck]  ]

            #print ("A")

            #plt.imshow(boolSubset)
            #plt.show()
            #quit()

            errorNow = subset_calculateError(predRDR_now, predBAF_now, RDR, HAP, chr, boolSubset, noiseRDR, noiseBAF, withAdjust)

            #print ("B")
            #errorNow = errorNow.data.numpy()

        a += 1


    #print ("did loop")

    multPos = multPos[:count1]
    multPaste = multPaste[:count1]

    #print ('done1')
    #quit()


    weightUsed_reshape = weightUsed.reshape((weightUsed.shape[0], weightUsed.shape[1], 1))


    del cellProb

    cellProb = torch.logsumexp(cellProbFull, axis=0)


    #cellProbFull_np = cellProbFull.data.numpy()
    #cellProbFull_np[cellProbFull_np < -1 * 1e10] = 0
    #plt.imshow(cellProbFull_np)
    #plt.show()
    #quit()

    cellProb_reshape2 = cellProb.reshape((1, -1)).data.numpy()
    multPaste = multPaste - cellProb_reshape2

    



    multPaste2 = logsumexp(multPaste, axis=1)
    multPaste2 = multPaste2 - np.log(cellProb.shape[0])
    multPaste2 = np.exp(multPaste2)



    multPaste3 = np.zeros((modelProbSum.shape[0], modelProbSum.shape[1]))
    multPaste3[multPos[:, 0], multPos[:, 1]] = multPaste2


    assert np.max(multPaste3) <= 1
    weightFull0 = ''


    #print (np.mean(cellProb))
    #quit()

    #print ('did full generation to return')

    return multPaste3, cellProb, weightFull0, errorFull, multPaste, multPos



def measurementCalculator(savedCNA, model, doBAF):

    

    lastDim = len(savedCNA.shape) - 1

    predRDR = np.sum(savedCNA, axis=lastDim).astype(float)
    predRDR = np.abs(predRDR)


    predBAF = ''
    if doBAF:
        #print (savedCNA.shape)
        #print (lastDim)
        #quit()
        if lastDim == 2:
            predBAF = np.abs(savedCNA[:, :, 1]).astype(float)
        if lastDim == 3:
            predBAF = np.abs(savedCNA[:, :, :, 1]).astype(float)

        #del savedCNA

        #print (  np.min(predBAF) )
        #print (  np.min(predRDR) )

        predBAF = torch.tensor(predBAF / (predRDR + 1e-5)   ).float() 

        
        #print (np.min( predBAF.data.numpy() ))
        
        assert np.min( predBAF.data.numpy() ) >= 0
    

    

    #print ('pred_RDR2')
    #print (np.min(np.abs((predRDR))))
    #if np.isnan(predRDR).any():
    #    print ("is nan error")
    #    quit()


    predRDR = torch.tensor(predRDR).float()


    mean1 = torch.mean(predRDR, axis=lastDim-1)


    if False: #TODO: delete not needed code. 
        biasAdjustment = model.biasAdjuster() + 1


        if lastDim == 2:
            predRDR = predRDR * biasAdjustment.reshape((1, biasAdjustment.shape[0]))

            predRDR = predRDR / mean1.reshape((-1, 1))

        if lastDim == 3:
            predRDR = predRDR * biasAdjustment.reshape((1, 1, biasAdjustment.shape[0]))

            predRDR = predRDR / mean1.reshape((mean1.shape[0], mean1.shape[1], 1))


    #print ('pred_RDR4')
    #print (torch.min(torch.abs((biasAdjustment))))
    #if np.isnan(biasAdjustment.data.numpy()).any():
    #    print ("is nan error")
    #    quit()




    #print ('pred_RDR3')
    #print (torch.min(torch.abs((predRDR))))
    #if np.isnan(predRDR.data.numpy()).any():
    #    print ("is nan error")
    #    quit()


    #print (savedCNA.shape)
    #print (biasAdjustment.shape)
    #quit()

    #measurementCalculator(savedCNA, model, doBAF)

    



    return predRDR, predBAF



def OLD_CNAbulkEst(bestCNA, RDR, HAP):

    inverse1 = uniqueProfileMaker(bestCNA)

    _, index1, count1 = np.unique(inverse1, return_index=True, return_counts=True)

    index1 = index1[count1 >= 5]

    CNA_new = np.zeros( (index1.shape[0], bestCNA.shape[1], bestCNA.shape[2] ) , dtype=int)

    for a in range(index1.shape[0]):
        args1 = np.argwhere(inverse1 == index1[a])[:, 0]

        RDR_now, HAP_now = RDR[args1], HAP[args1]
        RDR_now = np.mean(RDR_now, axis=0)
        HAP_now = np.sum(HAP_now, axis=0)
        BAF_now = HAP_now[:, 1] / (np.sum(HAP_now, axis=1) + 1e-5)

        ploidy1 = np.mean(bestCNA[index1[a]]) * 2
        RDR_now = RDR_now * ploidy1
        totalNow = np.round(RDR_now).astype(int)

        Bcopy = np.round(totalNow * BAF_now)

        CNA1 = np.array( [totalNow - Bcopy, Bcopy]  ).T

        CNA_new[a] = CNA1
    

    print ('CNA_new')
    print (CNA_new.shape)

    CNA_both = np.concatenate((bestCNA, CNA_new), axis=0)

    inverse2 = uniqueProfileMaker(CNA_both)

    argGood = np.argwhere(np.isin( inverse2, inverse2[:bestCNA.shape[0]] ) == False)[:, 0]

    CNA_new = CNA_both[argGood]

    print (CNA_new.shape)
    
    #quit()

    return CNA_new




def CNAbulkEst(bestCNA, RDR, HAP):

    
    
    #CNA_new = np.zeros( bestCNA.shape , dtype=int)
    CNA_new = np.copy(bestCNA)

    #plt.imshow(  CNA_new[0::20, :, 1] / ( np.sum(CNA_new[0::20], axis=2) + 1e-5 )  , cmap='bwr' )
    #plt.show()   

    #haplotypePlotter(CNA_new, doCluster=False)

    for a in range(bestCNA.shape[1]):

        inverse1 = uniqueValMaker(bestCNA[:, a])
        unique1, counts1 = np.unique(inverse1, return_counts=True)


        for b in range(unique1.shape[0]):
            args1 = np.argwhere(inverse1 == unique1[b])[:, 0]
            if args1.shape[0] > 10:

                copy1 = bestCNA[args1[0], a]
                copy_total = float(np.sum(copy1))

                B_options = np.arange(copy_total+1)
                A_options = copy_total - B_options

                HAP_now = np.sum(HAP[args1, a], axis=0)
                #BAF_now = HAP_now[1] / (np.sum(HAP_now) + 1e-5)
                #B_copy = np.round(copy_total * BAF_now)
                #A_copy = copy_total - B_copy

                

                #print (BAF_now)

                lossList = np.zeros(A_options.shape[0])
                for c in range(A_options.shape[0]):
                    BAF1 = B_options[c] / (A_options[c] + B_options[c] + 1e-5)
                    BAF1 = (BAF1 * 0.9) + 0.05
                    logA = np.log(1 - BAF1)
                    logB = np.log(BAF1)
                    loss1 = (logA * HAP_now[0]) + (logB * HAP_now[1])
                    lossList[c] = loss1
                argBest = np.argmax(lossList)
                A_copy, B_copy = A_options[argBest], B_options[argBest]


                good1 = True

                if False:#copy_total == 1: #TODO just an attempt
                    if np.sum(HAP_now) > 100:
                        if np.min(HAP_now) / np.sum(HAP_now) > 0.3:
                            A_copy, B_copy = 1, 1

                            print ("Triggered")
                            print (copy1, A_copy, B_copy)
                            print (HAP_now)


                #if a == bestCNA.shape[1] - 1:
                #    print ('A')
                #    print (copy1, A_copy, B_copy)
                #    print (HAP_now)

                if 0 in [A_copy, B_copy]:

                    if np.sum(HAP_now) < 100:
                        good1 = False
                    
                    #print ("copy")
                    #print (copy1, A_copy, B_copy)
                    #print (HAP_now)

                    weird1 = False 
                    if (copy1[0] != 0) and (A_copy == 0):
                        weird1 = True
                    if (copy1[1] != 0) and (B_copy == 0):
                        weird1 = True

                    #if weird1:
                    #    print ("copy")
                    #    print (copy1, A_copy, B_copy)
                    #    print (HAP_now)


                if good1:
                    CNA_new[args1, a, 0] = A_copy
                    CNA_new[args1, a, 1] = B_copy


    
    
    #print ('CNA_new')
    #print (CNA_new.shape)
                
    #plt.imshow(  CNA_new[0::20, :, 1] / ( np.sum(CNA_new[0::20], axis=2) + 1e-5 )  , cmap='bwr' )
    #plt.show()    

    if False:
        CNA_both = np.concatenate((bestCNA, CNA_new), axis=0)

        inverse2 = uniqueProfileMaker(CNA_both)

        argGood = np.argwhere(np.isin( inverse2, inverse2[:bestCNA.shape[0]] ) == False)[:, 0]

        CNA_new = CNA_both[argGood]
    
    #print (CNA_new.shape)
    
    #haplotypePlotter(CNA_new, doCluster=False)

    
    #quit()

    return CNA_new



def trainModel(CNAfull, chr, RDR, HAP, originalError, modelName, predict_file, Ncall, noiseRDR, noiseBAF, withAdjust, balance, doDouble=False, stopIter=False):







    #initialUniqueIndex_file = './data/inputResults/S' + '1' + '_initialIndex.npz'
    #uniqueIndex = loadnpz(initialUniqueIndex_file)






    _, start1 = np.unique(chr, return_index=True)
    end1 = np.concatenate((start1[1:], np.zeros(1) + chr.shape[0])).astype(int)

    withBAF = True
    if type(HAP) == type(''):
        withBAF = False

    
    #    BAF = np.min(np.array([BAF, 1 - BAF]), axis=0)


    #Ncall = 10
    CNAfull[CNAfull >= Ncall] = Ncall - 1

    CNAfull_now = np.copy(CNAfull)

    #Nrep = 50
    #Nrep = 100
    #Nrep = 200
    Nrep = 500
    #Nrep = 1000

    #CNAcodeSize = np.sum(  (end1-start1)**2 )

    Nbin = CNAfull.shape[1]

    model = CancerModel(Nbin, Nrep, Ncall, withBAF)
    #model = torch.load('./data/ACT10x/model_1.pt')


    #model = torch.load(modelName)


    #errorFull = np.zeros((100000, x.shape[0]))
    #errorDict = {}



    

    learningRate = 3e-3 

    

    #learningRate = 1e-3

    #learningRate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr = learningRate, betas=(0.9, 0.99))
    #optimizer = torch.optim.Adam(model.parameters(), lr = learningRate, betas=(0.8, 0.99))
    #optimizer = torch.optim.RMSprop(model.parameters(), lr = learningRate, alpha=(0.9))

    #optimizer = torch.optim.SGD(model.parameters(), lr = learningRate) #TODO delete unneeded. Tried for memory issue. 



    converged = False

    peakAccuracy = -100000

    stopCheck = False

    

    cellProbList = []


    #Niter = 1000000
    Niter = 1

    continue1 = True
    iter = -1

    #doScales = True
    doScales = False


    iterPass = 20


    counterAll = np.zeros(CNAfull.shape[0])


    while continue1:
        iter += 1

        print ('iterations', iter)

        #gapLearn = 2 #Default 2
        gapLearn = 2 #Default 2
        gapTime = 10#5 #Default 5

        if False:#len(cellProbList) > gapTime:
            cellProb_array = np.array(cellProbList)
            cellProb_max1 = np.max(cellProb_array[:-gapTime])
            cellProb_max2 = np.max(cellProb_array[-gapTime:])
            print ("maxs", cellProb_max1, cellProb_max2)
            print (cellProb_max1 + gapLearn, cellProb_max2)
            #if cellProb_max2 < cellProb_max1 + gapLearn:
            #    print ('stopping')
            #    continue1 = False

        if stopIter:
            #if iter == 100:
            if iter == 400:
                continue1 = False


        if continue1:
            #print (iter)
            info = [iter]


            modelProbSum, sampleProbSum, treeLoss, argValid, savedCNA_mapper, savedCNA, stepLast = modelCNAgenerator(CNAfull_now, chr, start1, end1, model, Ncall, info, returnReg=True, doDouble=True)

            
            




            time1 = time.time()

            #print ('min', np.min(CNAfull))

            #plt.hist(stepLast, bins=100)
            #plt.show()



            modelProbSum_np = modelProbSum.data.numpy()

            assert np.max(modelProbSum_np) <= 0.01
            assert np.max(modelProbSum_np - sampleProbSum) <= 0.01



            #print ('savedCNA')
            #if np.isnan(savedCNA).any():
            #    print ("is nan error0")
            #    quit()

            print ('doing measurementCalculator')

            predRDR, predBAF = measurementCalculator(savedCNA, model, withBAF)


            print ('doing efficiencyCalculateError')
            

            #balance = 10.0
            #balance = 5.0
            #balance = 1.0
            weight1, cellProb, weightFull0, errorMatrix, multPaste, multPos = efficiencyCalculateError(predRDR, predBAF, savedCNA_mapper, RDR, HAP, stepLast, modelProbSum_np, sampleProbSum, originalError, chr, noiseRDR, noiseBAF, balance, withAdjust)


            bestFit = np.argmax(multPaste, axis=0) #for later use

            
            del predRDR 
            del predBAF
            del weightFull0
            del errorMatrix
            del multPaste


            #for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(
            #                        locals().items())), key= lambda x: -x[1])[:10]:
            #    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))

            #for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(
            #                        locals().items())), key= lambda x: -x[1])[:10]:
            #    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
            
            #plt.imshow(weight1)
            #plt.show()
            #quit()

            #print (np.sum(weight1))
            #quit()

            if True: #iter > iterPass:
                argWeight = np.argwhere(weight1 > (0.5 / RDR.shape[0]))
                CNAfull_best = savedCNA[savedCNA_mapper[argWeight[:, 0], argWeight[:, 1]]]

                bestCNA = savedCNA[savedCNA_mapper[ multPos[bestFit, 0], multPos[bestFit, 1] ] ]
                CNA_new = CNAbulkEst(bestCNA, RDR, HAP)


                CNA_randoms = np.copy(CNAfull_best)
                chrRand = np.random.randint(22, size=CNA_randoms.shape[0])
                for a in range(CNA_randoms.shape[0]):
                    perm_CNA = np.random.permutation(CNA_randoms.shape[0])
                    bDone = False
                    b = 0
                    while not bDone:
                        diff1 = np.sum(np.abs( CNA_randoms[a, chr==chrRand[a]] -  CNA_randoms[perm_CNA[b], chr==chrRand[a]] ))
                        if diff1 <= 3:
                            bDone = True
                            CNA_randoms[a, chr==chrRand[a]] = np.copy(CNA_randoms[perm_CNA[b], chr==chrRand[a]])
                        b += 1
                        if b == perm_CNA.shape[0]:
                            bDone = True
                



                CNAfull_temp = np.concatenate((CNAfull, CNAfull_best), axis=0)
                inverse_temp = uniqueValMaker( CNAfull_temp.reshape((CNAfull_temp.shape[0],  CNAfull_temp.shape[1]*CNAfull_temp.shape[2] )) )
                _, index_temp = np.unique(inverse_temp, return_index=True)

                #print ("A")
                #print (np.argwhere(index_temp >=  CNAfull.shape[0] ).shape)
                #print (np.argwhere(index_temp <  CNAfull.shape[0] ).shape)

                #quit()
                #CNAfull_now = np.concatenate((CNAfull, CNAfull_best), axis=0)
                #CNAfull_now = np.concatenate((CNAfull, CNAfull_best, CNA_randoms), axis=0)
                CNAfull_now = np.concatenate((CNAfull, CNAfull_best, CNA_randoms, CNA_new), axis=0)

                inverse1 = uniqueValMaker( CNAfull_now.reshape((CNAfull_now.shape[0],  CNAfull_now.shape[1]*CNAfull_now.shape[2] )) )
                _, index1 = np.unique(inverse1, return_index=True)
                index1 = np.sort(index1)


                del CNAfull_best
                del CNA_randoms
                del CNAfull_temp

                

                

                #img1 = np.zeros((savedCNA.shape[0], savedCNA.shape[1]))
                #img1[argWeight[:, 0], argWeight[:, 1]] = 1
                #img1[stepLast, np.arange(stepLast.shape[0])] = 2
                #plt.imshow(img1)
                #plt.show()




                CNAfull_now = CNAfull_now[index1]



            


            #print ('CNAfull_best', CNAfull_best.shape, CNAfull.shape, index1.shape)

            #plt.imshow(CNAfull_best[:, :, 0])
            #plt.show()

            #for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(
            #                        locals().items())), key= lambda x: -x[1])[:10]:
            #    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))



            #reward = np.copy(weight1)
            reward = torch.tensor(weight1).float() #NOTE: One can NOT do reward - np.mean(reward)! This is because the non-sampled cases have reward 0 by definition!
            del weight1

            lossAll = -1 * modelProbSum * reward

            #loss = torch.mean(lossAll)
            loss = torch.mean(torch.sum(lossAll, axis=(0, 1)))

            del reward

            #print (torch.mean(treeLoss))
            #loss = loss + (torch.mean(treeLoss) * 1e-6) #0.001
            #loss = loss + (torch.mean(treeLoss) * 1e-4)




            negativeLogProb = model.normalizedBias(model.biasAdjuster())

            #lossAdjustment = (-1 * torch.mean(cellProb)) * 10
            lossAdjustment = (-1 * torch.mean(cellProb)) + (negativeLogProb / RDR.shape[0])
            lossAdjustment = lossAdjustment #* 10

            #* 10

            

            #print ('')
            #print (cellProb)
            #print (torch.mean(cellProb))
            #print ('cell', torch.max(cellProb))
            #print (np.mean(modelProbSum.data.numpy() ))
            cellProb_mean = torch.mean(cellProb).data.numpy()
            cellProbList.append(cellProb_mean)

            #for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(
            #                        locals().items())), key= lambda x: -x[1])[:10]:
            #    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))


            #reg = 0
            #for param in model.parameters():
            #    reg += torch.sum(param ** 2)

            if True:

                print ('bestFit', np.unique(bestFit).shape, bestFit.shape)

                bestCNA = savedCNA[savedCNA_mapper[ multPos[bestFit, 0], multPos[bestFit, 1] ] ]

                #plt.imshow(bestCNA[:, :, 0])
                #plt.show()


                diff1 = bestCNA[:, 1:, 0] - bestCNA[:, :-1, 0]
                diff1[diff1!=0] = 1
                #print ('diff1', np.sum(diff1))

                np.savez_compressed(predict_file, bestCNA)


                #print ('counterAll', np.argwhere( counterAll >= 1).shape, np.argwhere( counterAll >= (iter // 2) + 1  ).shape ,iter,  (iter // 2) + 1 )



            
            del savedCNA



            #for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(
            #                        locals().items())), key= lambda x: -x[1])[:10]:
            #    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))

            
            #print ("A")
            optimizer.zero_grad()
            #print ("B")
            loss.backward()
            #print ("C")
            #if withAdjust:
            #    print ("C1")
            #    lossAdjustment.backward()
            #print ('C2')
            optimizer.step()
            #print ("D")



            torch.save(model, modelName)




            



def findBestCNA(CNAfull, chr, RDR, BAF, originalError, modelName, bestCNAFile, Ncall, Ncheck, balance, withAdjust, noSave=False, doDouble=False):



    print ('hi')

    CNAfull[CNAfull >= Ncall] = Ncall - 1

    model = torch.load(modelName)

    _, start1 = np.unique(chr, return_index=True)
    end1 = np.concatenate((start1[1:], np.zeros(1) + chr.shape[0])).astype(int)

    #'''

    #Ncheck = 50
    #Ncheck = 10
    #Ncheck = 2
    #Ncheck = 1
    modelProb = np.zeros((Ncheck, CNAfull.shape[0]))
    sampleProb = np.zeros((Ncheck, CNAfull.shape[0]))

    info = ['']

    for a in range(Ncheck):

        print (a, Ncheck)

        modelProbSum, sampleProbSum, treeLoss, CNAused, savedCNA, stepLast = modelCNAgenerator(np.copy(CNAfull), chr, start1, end1, model, Ncall, info, returnReg=True, doDouble=True)#doDouble)
        modelProbSum_np = modelProbSum.data.numpy()


        #print (np.max(modelProbSum_np))
        #print (np.max(sampleProbSum))

        #print (np.max(modelProbSum_np - sampleProbSum))

        assert np.max(modelProbSum_np - sampleProbSum) < 0.01

        #print (modelProbSum_np.shape)
        #print (savedCNA.shape)
        #quit()

        modelProbSum_np = modelProbSum_np[stepLast, np.arange(stepLast.shape[0])]
        sampleProbSum = sampleProbSum[stepLast, np.arange(stepLast.shape[0])]

        assert np.max(modelProbSum_np - sampleProbSum) < 0.01

        modelProb[a] = np.copy(modelProbSum_np)
        sampleProb[a] = np.copy(sampleProbSum)



    

    modelProb = logsumexp(modelProb, axis=0) - np.log(Ncheck) #TODO check  if this is correct
    sampleProb = logsumexp(sampleProb, axis=0) - np.log(Ncheck) #TODO check  if this is correct

    np.savez_compressed('./temp/modelProb.npz', modelProb)
    np.savez_compressed('./temp/sampleProb.npz', sampleProb)
    #'''


    modelProb = loadnpz('./temp/modelProb.npz')
    sampleProb = loadnpz('./temp/sampleProb.npz')

    

    #print (modelProb.shape)
    #print (sampleProb.shape)
    #print (originalError.shape)
    #quit()


    #print (type(originalError))

    div1 = modelProb.reshape((-1, 1)) - sampleProb.reshape((-1, 1))


    originalError = originalError.detach().data.numpy()

    weight1 = (div1 * balance) - originalError


    #from scipy.special import softmax

    #plt.imshow(  softmax(weight1, axis=0))
    #plt.show()
    #quit()


    #balance

    #weight1 = ( 0 * ( modelProb.reshape((-1, 1)) - sampleProb.reshape((-1, 1)) )) - originalError



    if False:
        for a in range(weight1.shape[1]):
            weight1[:, a] = weight1[:, a] - np.max(weight1[:, a])
            weight1[:, a] = softmax(weight1[:, a])

        plt.plot(np.sum(weight1, axis=1))
        plt.show()


    bestFitList = []
    for a in range(weight1.shape[1]):

        #plt.plot(originalError[:, a])
        #plt.plot(modelProb - sampleProb)
        #plt.plot(np.sum(CNAfull, axis=(1, 2))  )
        #plt.show()

        argMax = np.argmax(weight1[:, a])
        bestFitList.append(argMax)

        #print (CNAfull.shape)
        #print (originalError.shape)

        argMax0 = np.argmin(originalError[:, a])
        #argMax0 = argMax

        #plt.plot(np.sum(CNAfull[argMax], axis=1))
        #plt.plot(RDR[a] * np.mean(CNAfull[argMax], axis=(0, 1))*2  )
        #plt.show()


    bestFitList = np.array(bestFitList).astype(int)

    #print ('mean1')
    #print (np.mean(CNAfull[bestFitList].astype(float)))
    #print (np.min(np.mean(CNAfull[bestFitList].astype(float), axis=(1, 2))))

    '''
    initialUniqueIndex_file = './data/inputResults/S' + '1' + '_initialIndex.npz'
    indexOriginal = loadnpz(initialUniqueIndex_file)

    boolScale1 = np.round(np.mean(CNAfull[bestFitList].astype(float), axis=(1, 2)))
    boolScale2 = np.round(np.mean(CNAfull[indexOriginal].astype(float), axis=(1, 2)))

    print (boolScale1)

    argIssue = np.argwhere( np.logical_and(boolScale1 == 2, boolScale2 == 1))[:, 0]

    plt.plot(np.sum(CNAfull[bestFitList[argIssue[0]]], axis=1))
    plt.plot(np.sum(CNAfull[indexOriginal[argIssue[0]]], axis=1))
    plt.show()
    #'''


    #print (np.argwhere( np.logical_and(boolScale1 == 1, boolScale2 == 1) ).shape)
    #print (np.argwhere( np.logical_and(boolScale1 == 2, boolScale2 == 2) ).shape)
    #print (np.argwhere( np.logical_and(boolScale1 == 1, boolScale2 == 2) ).shape)
    #print (np.argwhere( np.logical_and(boolScale1 == 2, boolScale2 == 1) ).shape)

    #quit()

    if noSave:
        return bestFitList
    else:
        print ("Saved")
        np.savez_compressed(bestCNAFile, bestFitList)





def simpleTrain(RDR_file, HAP_file, chr_file, initialCNA_file, initialUniqueCNA_file, originalError_file, modelName, predict_file, Ncall, noise_file, BAF_noise_file, balance, withAdjust, stopIter=False):


    
    scales = loadnpz(initialCNA_file)

    CNAfull = loadnpz(initialUniqueCNA_file)

    CNAfull = CNAfull.reshape((CNAfull.shape[0], CNAfull.shape[1], 2))


    RDR = loadnpz(RDR_file)
    if HAP_file == '':
        HAP_file = ''
    else:
        HAP = loadnpz(HAP_file)
    chr = loadnpz(chr_file)# - 1 #FOR NOW! May 22 2023

    noiseRDR = ''
    if noise_file != '':
        noiseRDR = loadnpz(noise_file)
        noiseBAF = loadnpz(BAF_noise_file)

    

    if False:#not withAdjust:
        if False:
            print ("Skipping error calculation")
            originalError = loadnpz(originalError_file)
        else:
            print ("Do Error Calculation")
            #originalError = calculateError(CNAfull, RDR, BAF, chr, noiseRDR)
            #model = torch.load(modelName)
            originalError = calculateError(CNAfull, RDR, HAP, chr, noiseRDR, model, withAdjust)
            np.savez_compressed(originalError_file, originalError)

        #quit()

        originalError = torch.tensor(originalError).float()

    else:
        originalError = ''

    

    trainModel(CNAfull, chr, RDR, HAP, originalError, modelName, predict_file, Ncall, noiseRDR, noiseBAF, withAdjust, balance, stopIter=stopIter)





def simplePredict(RDR_file, BAF_file, chr_file, initialCNA_file, initialUniqueCNA_file, originalError_file, modelName, predict_file, Ncall, noise_file, withAdjust):




    CNAfull = loadnpz(initialUniqueCNA_file)
    RDR = loadnpz(RDR_file)
    if BAF_file == '':
        BAF = ''
    else:
        BAF = loadnpz(BAF_file)
    chr = loadnpz(chr_file) - 1 



    originalError = loadnpz(originalError_file)
    originalError = torch.tensor(originalError).float()



    bestCNAFile = './temp/bestCNA3.npz'
    Ncheck = 2
    print ('findBest')

    CNAfull = CNAfull.reshape((CNAfull.shape[0], CNAfull.shape[1], 1))

    balance = 1.0

    #findBestCNA(CNAfull, chr, RDR, BAF, originalError, modelName, bestCNAFile, Ncall, Ncheck, balance, withAdjust, doDouble=False) #Attempt


    bestCNA = loadnpz(bestCNAFile)


    #np.savez_compressed(predict_file, CNAfull[bestCNA])
    #quit()

    #pred_file = './temp/predCNA3.npz'

    print ('predictCNA')

    noiseRDR = loadnpz(noise_file)

    predictCNA(CNAfull, chr, RDR, BAF, originalError, '', '', modelName, bestCNAFile, Ncall, predict_file, noiseRDR, balance, withAdjust, doDouble=False) #Attempt








def easyRunRL(outLoc):


    RDR_file = outLoc + '/binScale/filtered_RDR_avg.npz'

    #RDR_file = './data/' + folder1 + '/binScale/RDR_adjusted.npz'

    HAP_file = outLoc + '/binScale/filtered_HAP_avg.npz'
    chr_file = outLoc + '/binScale/chr_avg.npz'


    noise_file = outLoc + '/binScale/filtered_RDR_noise.npz'
    BAF_noise_file = outLoc + '/binScale/BAF_noise.npz'
    region_file = outLoc + '/binScale/regions.npz'
    BAF_file = ''

    initialCNA_file = outLoc + '/binScale/initialCNA.npz'
    initialUniqueCNA_file = outLoc + '/binScale/initialUniqueCNA.npz'
    originalError_file = outLoc + '/originalError.npz' #2

    modelName =  outLoc + '/model/model_now.pt'
    predict_file = outLoc + '/model/pred_now.npz'
    Ncall = 20
    withAdjust = True

    #balance = 5.0
    #balance = 2.0
    balance = 1.0

    simpleTrain(RDR_file, HAP_file, chr_file, initialCNA_file, initialUniqueCNA_file, originalError_file, modelName, predict_file, Ncall, noise_file, BAF_noise_file, balance, withAdjust, stopIter=True)
