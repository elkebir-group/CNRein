import sys
import numpy as np

from .process import runProcessFull
from .runBAM import runAllSteps
from .scaler import scalorRunAll
from .scaler import saveReformatCSV
from .scaler import scalorRunBins
from .scaler import runNaiveCopy
from .RLCNA import easyRunRL
from .shared import findTreeFromFile


def getValuesSYS(listIn, keyList):

    valueList = []
    for key1 in keyList:
        arg1 = np.argwhere(listIn == key1)[0, 0]
        value1 = listIn[arg1+1]
        valueList.append(value1)
    return valueList


def checkInvalidArg(listIn, keyList):

    listIn_ar = np.array(listIn)
    badArg = listIn_ar[np.isin(listIn_ar,  np.array(keyList))]



    if badArg.shape[0] != 0:

        print ("Invalid arguements:")
        for a in range(badArg.shape[0]):
            print (badArg[a])
        quit()






def runEverything(bamLoc, refLoc, outLoc, refGenome, doCB=False, maxPloidy=10):

    runAllSteps(bamLoc, refLoc, outLoc, refGenome, useCB=doCB)
    runProcessFull(outLoc, refLoc, refGenome)
    scalorRunAll(outLoc, maxPloidy=maxPloidy)
    easyRunRL(outLoc)
    saveReformatCSV(outLoc, isNaive=False)

def scriptRunEverything():
    import sys
    listIn = np.array(sys.argv)

    maxPloidy = 10.1



    doCB = False
    if '-CB' in listIn:
        doCB = True
        useCB = True


    #keyList = [ 'python', 'python3', '-output', '-hap1', '-hap2', '-chr', ]
    #checkInvalidArg(listIn, ['maxPloidy', 'maxPliody'], ['-maxPloidy'])
    

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

    elif '-tree' in listIn:

        if '-chr' in listIn:

            if '-hap1' in listIn:
                keyList = ['-output', '-hap1', '-hap2', '-chr']
                values1 = getValuesSYS(listIn, keyList)
                outLoc, hap1File, hap2File, chrFile = values1[0], values1[1], values1[2], values1[3]
                findTreeFromFile(outLoc, runEasy=False, fileMatrix=[hap1File, hap2File], fileChr=chrFile)

            if '-hap' in listIn:
                keyList = ['-output', '-CNA', '-chr']
                values1 = getValuesSYS(listIn, keyList)
                outLoc, hapFile, chrFile = values1[0], values1[1], values1[2]
                findTreeFromFile(outLoc, runEasy=False, fileMatrix=[hapFile], fileChr=chrFile)



        else:
            
            keyList = ['-output']
            values1 = getValuesSYS(listIn, keyList)
            outLoc = values1[0]
            findTreeFromFile(outLoc)




    elif not '-step' in listIn:

        
        keyList = ['-input', '-ref', '-output', '-refGenome']
        values1 = getValuesSYS(listIn, keyList)
        if '-maxPloidy' in listIn:
            maxPloidy = float(getValuesSYS(listIn, ['-maxPloidy'])[0])

        bamLoc, refLoc, outLoc, refGenome = values1[0], values1[1], values1[2], values1[3]
        runEverything(bamLoc, refLoc, outLoc, refGenome, doCB=doCB, maxPloidy=maxPloidy)

    else:

        stepVal = getValuesSYS(listIn, ['-step'])
        stepVal = stepVal[0]

        

        if stepVal == 'processing':
            keyList = ['-input', '-ref', '-output', '-refGenome']
            values1 = getValuesSYS(listIn, keyList)
            bamLoc, refLoc, outLoc, refGenome = values1[0], values1[1], values1[2], values1[3]

            runAllSteps(bamLoc, refLoc, outLoc, refGenome, useCB=doCB)
            runProcessFull(outLoc, refLoc, refGenome)
            scalorRunBins(outLoc)
        
        if stepVal == 'NaiveCopy':
            values1 = getValuesSYS(listIn, ['-output'])
            outLoc = values1[0]

            if '-maxPloidy' in listIn:
                maxPloidy = float(getValuesSYS(listIn, ['-maxPloidy'])[0])

            runNaiveCopy(outLoc, maxPloidy=maxPloidy)

        if stepVal == 'DeepCopy':
            values1 = getValuesSYS(listIn, ['-output'])
            outLoc = values1[0]
            easyRunRL(outLoc)
            saveReformatCSV(outLoc, isNaive=False)
        

        if stepVal == 'processBams':
            keyList = ['-input', '-ref', '-output', '-refGenome']
            values1 = getValuesSYS(listIn, keyList)
            bamLoc, refLoc, outLoc, refGenome = values1[0], values1[1], values1[2], values1[3]
            runAllSteps(bamLoc, refLoc, outLoc, refGenome, useCB=doCB)
        
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






