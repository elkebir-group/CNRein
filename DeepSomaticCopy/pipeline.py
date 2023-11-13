import sys
import numpy as np

from .process import runProcessFull
from .runBAM import runAllSteps
from .scaler import scalorRunAll
from .scaler import saveReformatCSV
from .scaler import scalorRunBins
from .scaler import runNaiveCopy
from .RLCNA import easyRunRL



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
    saveReformatCSV(outLoc, isNaive=False)

def scriptRunEverything():
    import sys
    listIn = np.array(sys.argv)

    if (('-h' in listIn) or ('-help' in listIn)) or ('--help' in listIn):

        print ("Usage instructions:")
        print ('')
        print ('Help information :')
        print ('"DeepCopyRun -h" or "DeepCopyRun -help" or "DeepCopyRun --help" ')
        print ('')
        print ('Running pipeline:')
        print ('DeepCopyRun -input <BAM file location> -ref <reference folder location> -output <location to store results> -refGenome <either "hg19" or "hg38"> ' )
        print ('')
        print ('Running part of pipeline: ')
        print ('DeepCopyRun -step <name of step to be ran> -input <BAM file location> -ref <reference folder location> -output <location to store results> -refGenome <either "hg19" or "hg38"> ' )


    elif not '-step' in listIn:

        keyList = ['-input', '-ref', '-output', '-refGenome']
        
        values1 = getValuesSYS(listIn, keyList)
        bamLoc, refLoc, outLoc, refGenome = values1[0], values1[1], values1[2], values1[3]
        runEverything(bamLoc, refLoc, outLoc, refGenome)

    else:

        stepVal = getValuesSYS(listIn, ['-step'])
        stepVal = stepVal[0]

        

        if stepVal == 'processing':
            keyList = ['-input', '-ref', '-output', '-refGenome']
            values1 = getValuesSYS(listIn, keyList)
            bamLoc, refLoc, outLoc, refGenome = values1[0], values1[1], values1[2], values1[3]

            runAllSteps(bamLoc, refLoc, outLoc, refGenome)
            runProcessFull(outLoc, refLoc, refGenome)
            scalorRunBins(outLoc)
        
        if stepVal == 'NaiveCopy':
            values1 = getValuesSYS(listIn, ['-output'])
            outLoc = values1[0]
            runNaiveCopy(outLoc)

        if stepVal == 'DeepCopy':
            values1 = getValuesSYS(listIn, ['-output'])
            outLoc = values1[0]
            easyRunRL(outLoc)
            saveReformatCSV(outLoc, isNaive=False)
        

        if stepVal == 'processBams':
            keyList = ['-input', '-ref', '-output', '-refGenome']
            values1 = getValuesSYS(listIn, keyList)
            bamLoc, refLoc, outLoc, refGenome = values1[0], values1[1], values1[2], values1[3]
            runAllSteps(bamLoc, refLoc, outLoc, refGenome)
        
        if stepVal == 'variableBins':
            keyList = ['-ref', '-output', '-refGenome']
            values1 = getValuesSYS(listIn, keyList)
            refLoc, outLoc, refGenome = values1[0], values1[1], values1[2]

            runProcessFull(outLoc, refLoc, refGenome)
            scalorRunBins(outLoc)






def printCheck(bamLoc, refLoc, outLoc, refGenome):
    print ("Basic Print Check")
    print ('bamLoc', bamLoc, 'refLoc', refLoc, 'outLoc', outLoc, 'refGenome', refGenome)

def scriptCheck():
    import sys
    print (sys.argv)


def respondCheck():
    print ('check success')






