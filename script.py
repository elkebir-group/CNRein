

import sys
import numpy as np

from DeepSomaticCopy.pipeline import runEverything

def getValuesSYS(listIn, keyList):
    valueList = []
    for key1 in keyList:
        arg1 = np.argwhere(listIn == key1)[0, 0]
        value1 = listIn[arg1+1]
        valueList.append(value1)
    return valueList


if __name__ == "__main__":
    keyList = ['-input', '-ref', '-output', '-refGenome']
    listIn = np.array(sys.argv)

    doCB = False
    if '-CB' in listIn:
        listIn = listIn[listIn!='-CB']
        doCB = True

    values1 = getValuesSYS(listIn, keyList)
    bamLoc, refLoc, outLoc, refGenome = values1[0], values1[1], values1[2], values1[3]

    runEverything(bamLoc, refLoc, outLoc, refGenome, doCB=doCB)

