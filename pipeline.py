#CNA.py

import time
import sys
import numpy as np

from run import runAllSteps
from process import runProcessFull
from scaler import scalorRunAll
from scaler import saveReformatCSV
from RLCNA import easyRunRL


def getValuesSYS(listIn, keyList):

    valueList = []
    for key1 in keyList:
        arg1 = np.argwhere(listIn == key1)[0, 0]
        value1 = listIn[arg1+1]
        valueList.append(value1)
    return valueList


def runEverything(bamLoc, refLoc, outLoc, refGenome):

    runAllSteps(bamLoc, refLoc, outLoc, refGenome)
    runProcessFull(outLoc, refLoc, refGenome)
    scalorRunAll(outLoc)
    easyRunRL(outLoc)
    saveReformatCSV(outLoc)





#bamLoc = './data/TN3_FullMerge.bam'
#refLoc = './data/refNew'
#outLoc = './data/newTN3'
#refGenome = 'hg38'


#runAllSteps


keyList = ['-input', '-ref', '-output', '-refGenome']
listIn = np.array(sys.argv)
values1 = getValuesSYS(listIn, keyList)
bamLoc, refLoc, outLoc, refGenome = values1[0], values1[1], values1[2], values1[3]




runEverything(bamLoc, refLoc, outLoc, refGenome)





