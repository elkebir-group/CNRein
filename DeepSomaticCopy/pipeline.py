#CNA.py

#import time
import sys
import numpy as np
#import importlib
#import os

#location1 = os.path.abspath(__file__)
#location1 = '/'.join(location1.split('/')[:-1])
#print (location1)
#quit()

#process = importlib.import_module(location1 + '/process')
#runBAM = importlib.import_module(location1 + '/runBAM')
#scaler = importlib.import_module(location1 + '/scaler')
#RLCNA = importlib.import_module(location1 + '/RLCNA')


#process = importlib.import_module('process', './')
#process = importlib.import_module('process', location1)
#runBAM = importlib.import_module('runBAM', location1)
#scaler = importlib.import_module('scaler', location1)
#RLCNA = importlib.import_module('RLCNA', location1)


#import DeepCopy1_stefanivanovic99.process #import process


#from DeepCopy1_stefanivanovic99.process import runProcessFull #Seems self referential but I guess this is how it's supposed to be done
#from DeepCopy1_stefanivanovic99.runBAM import runAllSteps
#from DeepCopy1_stefanivanovic99.scaler import scalorRunAll
#from DeepCopy1_stefanivanovic99.scaler import saveReformatCSV
#from DeepCopy1_stefanivanovic99.RLCNA import easyRunRL

#from RLCNA.RLCNA import easyRunRL

#print (__name__)

#from . import process
from .process import runProcessFull
from .runBAM import runAllSteps
from .scaler import scalorRunAll
from .scaler import saveReformatCSV
from .RLCNA import easyRunRL



#print ('hi')
#quit()

#from runBAM import runAllSteps
#from scaler import scalorRunAll
#from scaler import saveReformatCSV
#from RLCNA import easyRunRL




def getValuesSYS(listIn, keyList):

    valueList = []
    for key1 in keyList:
        arg1 = np.argwhere(listIn == key1)[0, 0]
        value1 = listIn[arg1+1]
        valueList.append(value1)
    return valueList


def runEverything(bamLoc, refLoc, outLoc, refGenome):

    #print (bamLoc, refLoc, outLoc, refGenome, location1)
    #quit()

    #command1 = location1 + '/runBAM.py ' + ' ' + bamLoc + ' ' + refLoc  + ' ' +  outLoc  + ' ' +  refGenome
    #os.system(command1)
    #print (command1)
    #quit()
    runAllSteps(bamLoc, refLoc, outLoc, refGenome)
    runProcessFull(outLoc, refLoc, refGenome)
    scalorRunAll(outLoc)
    easyRunRL(outLoc)
    saveReformatCSV(outLoc)


def printCheck(bamLoc, refLoc, outLoc, refGenome):

    print ("Basic Print Check")
    print ('bamLoc', bamLoc, 'refLoc', refLoc, 'outLoc', outLoc, 'refGenome', refGenome)



#bamLoc = './data/TN3_FullMerge.bam'
#refLoc = './data/refNew'
#outLoc = './data/newTN3'
#refGenome = 'hg38'


#runAllSteps
#quit()

#print ('hi')

if False:
    if __name__ == "__main__":
        keyList = ['-input', '-ref', '-output', '-refGenome']
        listIn = np.array(sys.argv)
        values1 = getValuesSYS(listIn, keyList)
        bamLoc, refLoc, outLoc, refGenome = values1[0], values1[1], values1[2], values1[3]

        runEverything(bamLoc, refLoc, outLoc, refGenome)





