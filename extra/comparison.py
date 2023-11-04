#CNA.py

import numpy as np


import matplotlib.pyplot as plt
import time
import scipy
from scipy import stats
from scipy.special import logsumexp



import pandas as pd
import matplotlib as mpl
import os
import seaborn as sns


#from scaler import *

from shared import *


def loadnpz(name, allow_pickle=False):

    #This simple function more easily loads in compressed numpy files.

    if allow_pickle:
        data = np.load(name, allow_pickle=True)
    else:
        data = np.load(name)
    data = data.f.arr_0
    return data


    



def saveAlternateMethodPred():

    def subsetToLook(data, toLook):

        row = data[0]
        #toLook = ['chr', 'start', 'end', 'cell_id', 'BAF.signals', 'state_AS_phased.signals', 'reads', 'gc', 'map', 'BAF.chisel', 'state_AS_phased.chisel', 'read_count.chisel']
        indexList = []
        for a in range(len(toLook)):
            arg1 = np.argwhere(row == toLook[a])[0, 0]
            indexList.append(arg1)
        indexList = np.array(indexList).astype(int)

        data = data[:, indexList]

        return data

    def rename10X(data):
        
        for a in range(1, data.shape[0]):
            name1 = data[a, 3]
            name1 = name1.split('-')
            section = name1[1][-1]
            barcode = name1[2]
            name1 = section + '-' + barcode
            data[a, 3] = name1

        return data
    

    def findCopyNumbers(data):

        #print ("A")

        converter = {}
        for a in range(1, 23):
            converter[str(a)] = a-1
        converter['X'] = 22

        unique_chr, index_chr, inverse_chr = np.unique(data[:, 0], return_index=True, return_inverse=True)
        unique_chr_mod = np.zeros(unique_chr.shape[0], dtype=int)-1

        for a in range(unique_chr.shape[0]):
            unique_chr_mod[a] = converter[unique_chr[a]]
        #print (index_chr.shape, inverse_chr.shape)
        #_, index_chr_rel = np.unique(index_chr, return_inverse=True)
        inverse_chr = unique_chr_mod[inverse_chr]

        #print (unique_chr)
        #print (unique_chr_mod)
        #quit()

        unique_pos, inverse_pos = np.unique(data[:, 1].astype(int), return_inverse=True)

        inverse_all = (inverse_chr * (unique_pos.shape[0]+1)) + inverse_pos
        unique_all, index_all, inverse_all = np.unique(inverse_all, return_index=True, return_inverse=True)

        positionList = data[index_all, :2]

        sampleList, inverse_sample = np.unique(data[:, 3], return_inverse=True)

        copyMatrix = np.zeros(( sampleList.shape[0], unique_all.shape[0], 2  ), dtype=int) - 1

        copyNumber = data[:, 4]
        copyNumber = list(copyNumber)
        copyNumber = [a.split('|') for a in copyNumber]
        copyNumber = np.array(copyNumber)

        copyMatrix[inverse_sample, inverse_all] = copyNumber


        BAFMatrix = np.zeros(( sampleList.shape[0], unique_all.shape[0]  ), dtype=float) - 1
        BAF = data[:, 5]
        BAFMatrix[inverse_sample, inverse_all] = BAF

        return copyMatrix, BAFMatrix, sampleList, positionList
    

    def removeMissing(matrix1, positionList):


        chr = positionList[:, 0]
        chr[chr=='X'] = 50
        chr = chr.astype(int)

        
        argBad = np.argwhere(matrix1 == -1)
        size1 = argBad.shape[0] * 2
        #print (argBad.shape)
        while argBad.shape[0] != size1:
            print (argBad.shape)
            size1 = argBad.shape[0]
            bool1 = chr[argBad[:, 1]] - chr[argBad[:, 1]-1]
            argBad = argBad[bool1 == 0]

            matrix1[argBad[:, 0], argBad[:, 1], argBad[:, 2]] = matrix1[argBad[:, 0], argBad[:, 1] - 1, argBad[:, 2]]
            argBad = np.argwhere(matrix1 == -1)
            

        argBad = np.argwhere(matrix1 == -1)
        while argBad.shape[0] != 0:
            print (argBad.shape)
            matrix1[argBad[:, 0], argBad[:, 1], argBad[:, 2]] = matrix1[argBad[:, 0], argBad[:, 1] + 1, argBad[:, 2]]
            argBad = np.argwhere(matrix1 == -1)


        
        
        return matrix1




    if False:
        toLook = ['chr', 'start', 'end', 'cell_id', 'state_AS_phased', 'BAF']
        data = np.loadtxt('./data/signatures_dataset/benchmarking/10X/signals.csv', delimiter=',', dtype=str)
        data = subsetToLook(data, toLook)
        data = rename10X(data)
        np.savez_compressed('./data/comparison/input/10x_signals.npz', data)

    if False:
        toLook = ['chr', 'start', 'end', 'cell_id', 'state_AS_phased', 'BAF']
        data = np.loadtxt('./data/signatures_dataset/benchmarking/10X/chisel.csv', delimiter=',', dtype=str)
        data = subsetToLook(data, toLook)
        data = rename10X(data)
        np.savez_compressed('./data/comparison/input/10x_chisel.npz', data)

    
    if False:
        toLook2 = ['chr', 'start', 'end', 'cell_id', 'state_AS_phased.signals', 'BAF.signals', 'reads', 'gc', 'map']
        toLook3 = ['chr', 'start', 'end', 'cell_id', 'state_AS_phased.chisel', 'BAF.chisel', 'read_count.chisel']
        data = np.loadtxt('./data/signatures_dataset/benchmarking/persample/OV2295_combined_nogc.csv', delimiter=',', dtype=str)
        data2 = subsetToLook(np.copy(data), toLook2)
        np.savez_compressed('./data/comparison/input/DLP_signals.npz', data2)

        data3 = subsetToLook(np.copy(data), toLook3)
        np.savez_compressed('./data/comparison/input/DLP_chisel.npz', data3)

    #quit()



    if False:

        data = loadnpz('./data/comparison/input/10x_signals.npz')
        data = data[1:]

        copyMatrix, BAFMatrix, sampleList, positionList = findCopyNumbers(data)
        
        np.savez_compressed('./data/comparison/input/10x_signals_copyNumbers.npz', copyMatrix)
        np.savez_compressed('./data/comparison/input/10x_signals_BAF.npz', BAFMatrix)
        np.savez_compressed('./data/comparison/input/10x_signals_cell.npz', sampleList)
        np.savez_compressed('./data/comparison/input/10x_signals_positions.npz', positionList)

    if False:
        data = loadnpz('./data/comparison/input/10x_chisel.npz')
        data = data[1:]

        copyMatrix, BAFMatrix, sampleList, positionList = findCopyNumbers(data)
        
        np.savez_compressed('./data/comparison/input/10x_chisel_copyNumbers.npz', copyMatrix)
        np.savez_compressed('./data/comparison/input/10x_chisel_BAF.npz', BAFMatrix)
        np.savez_compressed('./data/comparison/input/10x_chisel_cell.npz', sampleList)
        np.savez_compressed('./data/comparison/input/10x_chisel_positions.npz', positionList)

    if False:
        data = loadnpz('./data/comparison/input/DLP_chisel.npz')
        data = data[1:]

        copyMatrix, BAFMatrix, sampleList, positionList = findCopyNumbers(data)
        
        np.savez_compressed('./data/comparison/input/DLP_chisel_copyNumbers.npz', copyMatrix)
        np.savez_compressed('./data/comparison/input/DLP_chisel_BAF.npz', BAFMatrix)
        np.savez_compressed('./data/comparison/input/DLP_chisel_cell.npz', sampleList)
        np.savez_compressed('./data/comparison/input/DLP_chisel_positions.npz', positionList)

    if False:
        data = loadnpz('./data/comparison/input/DLP_signals.npz')
        data = data[1:]

        copyMatrix, BAFMatrix, sampleList, positionList = findCopyNumbers(data)

        print (BAFMatrix.shape)
        
        np.savez_compressed('./data/comparison/input/DLP_signals_copyNumbers.npz', copyMatrix)
        np.savez_compressed('./data/comparison/input/DLP_signals_BAF.npz', BAFMatrix)
        np.savez_compressed('./data/comparison/input/DLP_signals_cell.npz', sampleList)
        np.savez_compressed('./data/comparison/input/DLP_signals_positions.npz', positionList)

    


    if False:
        copyMatrix = loadnpz('./data/comparison/input/10x_chisel_copyNumbers.npz')
        positionList = loadnpz('./data/comparison/input/10x_chisel_positions.npz')
        copyMatrix = removeMissing(copyMatrix, positionList)
        np.savez_compressed('./data/comparison/input/10x_chisel_copyNoMissing.npz', copyMatrix)

    if False:
        copyMatrix = loadnpz('./data/comparison/input/DLP_signals_copyNumbers.npz')
        positionList = loadnpz('./data/comparison/input/DLP_signals_positions.npz')
        copyMatrix = removeMissing(copyMatrix, positionList)
        np.savez_compressed('./data/comparison/input/DLP_signals_copyNoMissing.npz', copyMatrix)
        

    if False:
        copyMatrix = loadnpz('./data/comparison/input/DLP_chisel_copyNumbers.npz')
        positionList = loadnpz('./data/comparison/input/DLP_chisel_positions.npz')
        copyMatrix = removeMissing(copyMatrix, positionList)
        np.savez_compressed('./data/comparison/input/DLP_chisel_copyNoMissing.npz', copyMatrix)

    if False:
        copyMatrix = loadnpz('./data/comparison/input/10x_signals_copyNumbers.npz')
        positionList = loadnpz('./data/comparison/input/10x_signals_positions.npz')
        copyMatrix = removeMissing(copyMatrix, positionList)
        np.savez_compressed('./data/comparison/input/10x_signals_copyNoMissing.npz', copyMatrix)


#saveAlternateMethodPred()
#quit()







def alignToSingle():

    

    def subsetToSharedCells(copyMatrix, copyMatrix_missing, BAFMatrix, sampleList, predCNA, naiveCNA, RDR, HAP, cellNames):
        
        
        subset1 = np.argwhere( np.isin( sampleList, cellNames ) )[:, 0]
        subset2 = np.argwhere( np.isin( cellNames, sampleList ) )[:, 0]
        subset1 = subset1[np.argsort(sampleList[subset1])]
        subset2 = subset2[np.argsort(cellNames[subset2])]

        #print (subset2[674])
        #quit()

        copyMatrix = copyMatrix[subset1]
        copyMatrix_missing = copyMatrix_missing[subset1]
        BAFMatrix = BAFMatrix[subset1]
        predCNA = predCNA[subset2]
        naiveCNA = naiveCNA[subset2]
        RDR = RDR[subset2]
        HAP = HAP[subset2]
        cellNames = cellNames[subset2]

        

        return copyMatrix, copyMatrix_missing, BAFMatrix, predCNA, naiveCNA, RDR, HAP, cellNames, subset2


    def convertTo100k(chr, chrAll, argGood, positionList):

        
        newSize = 100000

        M = 500000 // newSize

        M2 = newSize // 10000

        converter = {}
        for a in range(1, 23):
            converter[str(a)] = a-1
        converter['X'] = 22

        chrNames = list((np.arange(22) + 1).astype(int).astype(str))
        chrNames.append('X')


        
        argSubsetOurs = np.zeros( chrAll.shape[0], dtype=int )
        argSubsetTheres = np.zeros( chrAll.shape[0], dtype=int )

        
        

        count1 = 0


        for a in range(22):

            print (a)
            chrName = chrNames[a]

            #print (np.unique(our_positionList[:, 0]))

            argOurMethod = np.argwhere(chr == a)[:, 0]
            argGoodOurs = argGood[chr == a]
            argGoodOurs = argGoodOurs - np.argwhere(chrAll == a)[0, 0]

            argOtherMethod = np.argwhere(positionList[:, 0] == chrName)[:, 0]
            

            maxOurs = int(np.max(argGoodOurs))
            maxTheres = int(np.max(positionList[argOtherMethod[-1], 1].astype(int)) // 500000)



            
            
            copyMatrix_mini = np.zeros( maxTheres+1 , dtype=int) - 1
            copyMatrix_mini[positionList[argOtherMethod, 1].astype(int) // 500000 ] = argOtherMethod
            arange1 = np.arange(copyMatrix_mini.shape[0]).repeat(M)
            copyMatrix_mini = copyMatrix_mini[arange1]



            predCNA_mini = np.zeros(  maxOurs+1 , dtype=int) - 1
            predCNA_mini[argGoodOurs] = argOurMethod

            minCut = min( predCNA_mini.shape[0], copyMatrix_mini.shape[0] )
            predCNA_mini = predCNA_mini[:minCut]
            copyMatrix_mini = copyMatrix_mini[:minCut]


            subsetGood = np.argwhere(  np.logical_and( predCNA_mini >= 0, copyMatrix_mini >= 0  ) )[:, 0]
            predCNA_mini = predCNA_mini[subsetGood]
            copyMatrix_mini = copyMatrix_mini[subsetGood]

            size1 = predCNA_mini.shape[0]

            
            #sprint (count1, size1)
            #print (argGoodOurs.shape)
            #print (argGoodOurs[count1:count1+size1].shape, predCNA_mini.shape)

            argSubsetOurs[count1:count1+size1] = predCNA_mini
            argSubsetTheres[count1:count1+size1] = copyMatrix_mini

            count1 += size1
        

        argSubsetOurs = argSubsetOurs[:count1]
        argSubsetTheres = argSubsetTheres[:count1]

        
        return argSubsetOurs, argSubsetTheres





    def saveComparison(folder1, method1):


        folder2 = folder1 + '_' + method1

        BAFMatrix = loadnpz('./data/comparison/input/' + folder2 + '_BAF.npz')

        #plt.hist(BAFMatrix[BAFMatrix>=0].reshape((-1,)) , bins=100 )
        #plt.show()
        #quit()
    
        copyMatrix = loadnpz('./data/comparison/input/' + folder2 + '_copyNoMissing.npz') 
        copyMatrix_missing = loadnpz('./data/comparison/input/' + folder2 + '_copyNumbers.npz')
        sampleList = loadnpz('./data/comparison/input/' + folder2 + '_cell.npz')
        positionList = loadnpz('./data/comparison/input/' + folder2 + '_positions.npz')

        #positionList_file = './data/' + folder1 + '/initial/binPositions.npz'
        initialCNA_file = './data/' + folder1 + '/binScale/initialCNA.npz'
        #predict_file = './data/' + folder1 + '/model/pred_good.npz' #TODO switch to "good" not "1"
        predict_file = './data/' + folder1 + '/model/pred_3.npz'

        totalRead_file = './data/' + folder1 + '/initial/totalReads.npz'


        cellNames = loadnpz('./data/' + folder1 + '/initial/cellNames.npz')
        naiveCNA = loadnpz(initialCNA_file)
        predCNA = loadnpz(predict_file)
        
        #print (naiveCNA.shape)
        #print (predCNA.shape)
        #quit()




        hapHist_file = './data/' + folder1 + '/initial/HAP_100k.npz'
        hist_file = './data/' + folder1 + '/initial/RDR_100k.npz'
        adjustment_file = './data/' + folder1 + '/initial/gc_adjustment.npz'
        goodSubset_file = './data/' + folder1 + '/initial/subset.npz'
        chr_file = './data/' + folder1 + '/initial/chr_100k.npz'
        chrAll_file = './data/' + folder1 + '/initial/allChr_100k.npz'
        bins_file = './data/' + folder1 + '/binScale/bins.npz'




        adjustment = loadnpz(adjustment_file)
        argGood = loadnpz(goodSubset_file)
        RDR = loadnpz(hist_file)#[:, argGood] / adjustment.reshape((1, -1))
        HAP = loadnpz(hapHist_file)#[:, argGood]
        chr = loadnpz(chr_file)
        chrAll = loadnpz(chrAll_file)
        bins = loadnpz(bins_file)

        #print (np.unique(bins).shape)
        #_, index1 = np.unique(bins, return_index=True)
        #print (index1[:10])
        #quit()

        


        
        copyMatrix, copyMatrix_missing, BAFMatrix, predCNA, naiveCNA, RDR, HAP, cellNames, subset2 = subsetToSharedCells(copyMatrix, copyMatrix_missing, BAFMatrix, sampleList, predCNA, naiveCNA, RDR, HAP, cellNames)
        
        #np.savez_compressed('./data/comparison/CNA/' + folder2 + '_subsetOurs.npz', subset2)
        #quit()

        predCNA = predCNA[:, bins]
        naiveCNA = naiveCNA[:, bins]

        #print (predCNA.shape)
        #print (RDR.shape)
        #quit()

        #print (argGood.shape)
        #print (predCNA.shape)
        #quit()
        
        #convertRDR100k(RDR, HAP, adjustment, argGood)

        #print (cellNames[674])
        argGoodOurs, argGoodTheres = convertTo100k(chr, chrAll, argGood, positionList)
        predCNA, naiveCNA, HAP_full, RDR_full, chr_full = predCNA[:, argGoodOurs], naiveCNA[:, argGoodOurs], HAP[:, argGoodOurs], RDR[:, argGoodOurs], chr[argGoodOurs]
        copyMatrix, copyMatrix_missing, BAF_full = copyMatrix[:, argGoodTheres], copyMatrix_missing[:, argGoodTheres], BAFMatrix[:, argGoodTheres]
        

        
        #argGoodOurs = loadnpz('./data/comparison/CNA/' + folder2 + '_argSubsetOurs.npz')
        argGoodOurs = argGood[argGoodOurs]
        posToBin = np.zeros(chrAll.shape[0], dtype=int) - 1
        posToBin[argGoodOurs] = np.arange(argGoodOurs.shape[0])

        print (predCNA.shape, chr_full.shape)

        folder2 = folder2 + '_mod'
        #np.savez_compressed('./data/comparison/CNA/' + folder2 + '_deep.npz', predCNA)
        #quit()

        np.savez_compressed('./data/comparison/CNA/' + folder2 + '_posToBin.npz', posToBin)
        np.savez_compressed('./data/comparison/CNA/' + folder2 + '_argSubsetOurs.npz', argGoodOurs)
        np.savez_compressed('./data/comparison/CNA/' + folder2 + '_deep.npz', predCNA)
        np.savez_compressed('./data/comparison/CNA/' + folder2 + '_naive.npz', naiveCNA)
        np.savez_compressed('./data/comparison/CNA/' + folder2 + '_there.npz', copyMatrix)
        np.savez_compressed('./data/comparison/CNA/' + folder2 + '_missing.npz', copyMatrix_missing)
        np.savez_compressed('./data/comparison/CNA/' + folder2 + '_BAF.npz', BAF_full)
        np.savez_compressed('./data/comparison/CNA/' + folder2 + '_RDR_adj.npz', RDR_full)
        np.savez_compressed('./data/comparison/CNA/' + folder2 + '_HAP.npz', HAP_full)
        np.savez_compressed('./data/comparison/CNA/' + folder2 + '_chr.npz', chr_full)
        np.savez_compressed('./data/comparison/CNA/' + folder2 + '_subsetOurs.npz', subset2)

        print ("Saved")
        




    #folder1 = '10x'
    folder1 = 'DLP'
    method1 = 'signals'
    #method1 = 'chisel'
    folder2 = folder1 + '_' + method1
    
    #saveComparison(folder1, method1)
    #quit()








def alignMethods():

    

    def subsetToSharedCells(cellNames_signals, cellNames_chisel, cellNames):

        interName = np.intersect1d(cellNames_signals, cellNames_chisel)
        interName = np.intersect1d(interName, cellNames)
        
        
        subset1 = np.argwhere( np.isin( cellNames_signals, interName ) )[:, 0]
        subset2 = np.argwhere( np.isin( cellNames_chisel, interName ) )[:, 0]
        subset3 = np.argwhere( np.isin( cellNames, interName ) )[:, 0]


        subset1 = subset1[np.argsort(cellNames_signals[subset1])]
        subset2 = subset2[np.argsort(cellNames_chisel[subset2])]
        subset3 = subset3[np.argsort(cellNames[subset3])]        

        return subset1, subset2, subset3


    def convertTo100k(chr, chrAll, argGood, positionList_signals, positionList_chisel):

        
        newSize = 100000

        M = 500000 // newSize

        M2 = newSize // 10000

        converter = {}
        for a in range(1, 23):
            converter[str(a)] = a-1
        converter['X'] = 22

        chrNames = list((np.arange(22) + 1).astype(int).astype(str))
        chrNames.append('X')


        
        argSubsetOurs = np.zeros( chrAll.shape[0], dtype=int )
        argSubset_signals = np.zeros( chrAll.shape[0], dtype=int )
        argSubset_chisel = np.zeros( chrAll.shape[0], dtype=int )

        
        

        count1 = 0


        for a in range(22):

            print (a)
            chrName = chrNames[a]

            #print (np.unique(our_positionList[:, 0]))

            argOurMethod = np.argwhere(chr == a)[:, 0]
            argGoodOurs = argGood[chr == a]
            argGoodOurs = argGoodOurs - np.argwhere(chrAll == a)[0, 0]

            arg_signals = np.argwhere(positionList_signals[:, 0] == chrName)[:, 0]
            arg_chisel = np.argwhere(positionList_chisel[:, 0] == chrName)[:, 0]
            

            maxOurs = int(np.max(argGoodOurs))
            max_signals = int(np.max(positionList_signals[arg_signals[-1], 1].astype(int)) // 500000)
            max_chisel = int(np.max(positionList_chisel[arg_chisel[-1], 1].astype(int)) // 500000)



            
            
            mini_signals = np.zeros( max_signals+1 , dtype=int) - 1
            mini_chisel = np.zeros( max_chisel+1 , dtype=int) - 1
            mini_signals[positionList_signals[arg_signals, 1].astype(int) // 500000 ] = arg_signals
            mini_chisel[positionList_chisel[arg_chisel, 1].astype(int) // 500000 ] = arg_chisel
            arange1_signals = np.arange(mini_signals.shape[0]).repeat(M)
            arange1_chisel = np.arange(mini_chisel.shape[0]).repeat(M)
            mini_signals = mini_signals[arange1_signals]
            mini_chisel = mini_chisel[arange1_chisel]



            predCNA_mini = np.zeros(  maxOurs+1 , dtype=int) - 1
            predCNA_mini[argGoodOurs] = argOurMethod

            minCut = min( predCNA_mini.shape[0], mini_signals.shape[0] )
            minCut = min(minCut, mini_chisel.shape[0] )
            predCNA_mini = predCNA_mini[:minCut]
            mini_signals = mini_signals[:minCut]
            mini_chisel = mini_chisel[:minCut]


            subsetGood = np.argwhere(  np.logical_and(np.logical_and( predCNA_mini >= 0, mini_signals >= 0  ),  mini_chisel   ) )[:, 0]
            predCNA_mini = predCNA_mini[subsetGood]
            mini_signals = mini_signals[subsetGood]
            mini_chisel = mini_chisel[subsetGood]

            size1 = predCNA_mini.shape[0]

            
            #sprint (count1, size1)
            #print (argGoodOurs.shape)
            #print (argGoodOurs[count1:count1+size1].shape, predCNA_mini.shape)

            argSubsetOurs[count1:count1+size1] = predCNA_mini
            argSubset_signals[count1:count1+size1] = mini_signals
            argSubset_chisel[count1:count1+size1] = mini_chisel

            count1 += size1
        

        argSubsetOurs = argSubsetOurs[:count1]
        argSubset_signals = argSubset_signals[:count1]
        argSubset_chisel = argSubset_chisel[:count1]

        
        return argSubsetOurs, argSubset_signals, argSubset_chisel





    def saveComparison(folder1):


        folder2s = folder1 + '_signals'
        folder2c = folder1 + '_chisel'

        #BAFMatrix = loadnpz('./data/comparison/input/' + folder2 + '_BAF.npz')

        #plt.hist(BAFMatrix[BAFMatrix>=0].reshape((-1,)) , bins=100 )
        #plt.show()
        #quit()
    
        signals = loadnpz('./data/comparison/input/' + folder2s + '_copyNoMissing.npz') 
        missing_signals = loadnpz('./data/comparison/input/' + folder2s + '_copyNumbers.npz')
        cellNames_signals = loadnpz('./data/comparison/input/' + folder2s + '_cell.npz')
        positionList_signals = loadnpz('./data/comparison/input/' + folder2s + '_positions.npz')

        chisel = loadnpz('./data/comparison/input/' + folder2c + '_copyNoMissing.npz') 
        missing_chisel = loadnpz('./data/comparison/input/' + folder2c + '_copyNumbers.npz')
        cellNames_chisel = loadnpz('./data/comparison/input/' + folder2c + '_cell.npz')
        positionList_chisel = loadnpz('./data/comparison/input/' + folder2c + '_positions.npz')



        
        initialCNA_file = './data/' + folder1 + '/binScale/initialCNA.npz'
        #predict_file = './data/' + folder1 + '/model/pred_good.npz' #TODO switch to "good" not "1"
        predict_file = './data/' + folder1 + '/model/pred_good.npz'

        cellNames = loadnpz('./data/' + folder1 + '/initial/cellNames.npz')
        naiveCNA = loadnpz(initialCNA_file)
        predCNA = loadnpz(predict_file)
        
        



        hapHist_file = './data/' + folder1 + '/initial/HAP_100k.npz'
        hist_file = './data/' + folder1 + '/initial/RDR_100k.npz'
        #adjustment_file = './data/' + folder1 + '/initial/gc_adjustment.npz'
        goodSubset_file = './data/' + folder1 + '/initial/subset.npz'
        chr_file = './data/' + folder1 + '/initial/chr_100k.npz'
        chrAll_file = './data/' + folder1 + '/initial/allChr_100k.npz'
        bins_file = './data/' + folder1 + '/binScale/bins.npz'



        argGood = loadnpz(goodSubset_file)
        RDR = loadnpz(hist_file)#[:, argGood] / adjustment.reshape((1, -1))
        HAP = loadnpz(hapHist_file)#[:, argGood]
        chr = loadnpz(chr_file)
        chrAll = loadnpz(chrAll_file)
        bins = loadnpz(bins_file)

        #print (np.unique(bins).shape)
        #_, index1 = np.unique(bins, return_index=True)
        #print (index1[:10])
        #quit()

        


        
        subset1, subset2, subset3 = subsetToSharedCells(cellNames_signals, cellNames_chisel, cellNames)
        
        signals = signals[subset1]
        missing_signals = missing_signals[subset1]

        chisel = chisel[subset2]
        missing_chisel = missing_chisel[subset2]

        predCNA = predCNA[subset3]
        naiveCNA = naiveCNA[subset3]
        RDR = RDR[subset3]
        HAP = HAP[subset3]

        predCNA = predCNA[:, bins]
        naiveCNA = naiveCNA[:, bins]

        


        argGoodOurs, argSubset_signals, argSubset_chisel = convertTo100k(chr, chrAll, argGood, positionList_signals, positionList_chisel)
        predCNA, naiveCNA, HAP_full, RDR_full, chr_full = predCNA[:, argGoodOurs], naiveCNA[:, argGoodOurs], HAP[:, argGoodOurs], RDR[:, argGoodOurs], chr[argGoodOurs]


        signals, missing_signals = signals[:, argSubset_signals], missing_signals[:, argSubset_signals]
        chisel, missing_chisel = chisel[:, argSubset_chisel], missing_chisel[:, argSubset_chisel]
        
        

        
        #argGoodOurs = loadnpz('./data/comparison/CNA/' + folder2 + '_argSubsetOurs.npz')
        argGoodOurs = argGood[argGoodOurs]
        posToBin = np.zeros(chrAll.shape[0], dtype=int) - 1
        posToBin[argGoodOurs] = np.arange(argGoodOurs.shape[0])

        print (predCNA.shape, chr_full.shape)

        #folder2 = folder2 + '_mod'
        #np.savez_compressed('./data/comparison/CNA/' + folder2 + '_deep.npz', predCNA)
        #quit()

        np.savez_compressed('./data/comparison/CNA/' + folder1 + '_posToBin.npz', posToBin)
        np.savez_compressed('./data/comparison/CNA/' + folder1 + '_argSubsetOurs.npz', argGoodOurs)
        np.savez_compressed('./data/comparison/CNA/' + folder1 + '_deep.npz', predCNA)
        np.savez_compressed('./data/comparison/CNA/' + folder1 + '_naive.npz', naiveCNA)

        np.savez_compressed('./data/comparison/CNA/' + folder1 + '_signals.npz', signals)
        np.savez_compressed('./data/comparison/CNA/' + folder1 + '_signals_missing.npz', missing_signals)
        np.savez_compressed('./data/comparison/CNA/' + folder1 + '_chisel.npz', chisel)
        np.savez_compressed('./data/comparison/CNA/' + folder1 + '_chisel_missing.npz', missing_chisel)

        np.savez_compressed('./data/comparison/CNA/' + folder1 + '_RDR.npz', RDR_full)
        np.savez_compressed('./data/comparison/CNA/' + folder1 + '_HAP.npz', HAP_full)
        np.savez_compressed('./data/comparison/CNA/' + folder1 + '_chr.npz', chr_full)
        np.savez_compressed('./data/comparison/CNA/' + folder1 + '_subsetOurs.npz', subset3)
        np.savez_compressed('./data/comparison/CNA/' + folder1 + '_subsetSignals.npz', subset1)
        np.savez_compressed('./data/comparison/CNA/' + folder1 + '_subsetChisel.npz', subset2)

        print ("Saved")
        




    folder1 = '10x'
    #folder1 = 'DLP'
    #method1 = 'signals'
    #method1 = 'chisel'
    #folder2 = folder1 + '_' + method1
    
    #saveComparison(folder1)
    quit()

    

    #Chisel is clearly wrong on DLP
    #Chisel is sometimes (rarely) clearly wrong on 10x


    from scaler import haplotypePlotter
    

    
    predCNA = loadnpz('./data/comparison/CNA/' + folder2 + '_deep.npz').astype(float)
    naiveCNA = loadnpz('./data/comparison/CNA/' + folder2 + '_naive.npz').astype(float)
    copyMatrix = loadnpz('./data/comparison/CNA/' + folder2 + '_there.npz').astype(float)

    RDR_full = loadnpz('./data/comparison/CNA/' + folder2 + '_RDR_adj.npz')

    HAP_full = loadnpz('./data/comparison/CNA/' + folder2 + '_HAP.npz')

    BAFMatrix = loadnpz('./data/comparison/CNA/' + folder2 + '_BAF.npz')
    subset2 = loadnpz('./data/comparison/CNA/' + folder2 + '_subsetOurs.npz')
    chr = loadnpz('./data/comparison/CNA/' + folder2 + '_chr.npz')

    if False:
        from scipy.cluster.hierarchy import linkage
        linkage_matrix = linkage(predCNA.reshape((predCNA.shape[0],  predCNA.shape[1] * 2 ))  , method='ward', metric='euclidean')
        np.savez_compressed('./data/comparison/CNA/' + folder2 + '_linkageOurs.npz', linkage_matrix)
        quit()

    linkage_matrix = loadnpz('./data/comparison/CNA/' + folder2 + '_linkageOurs.npz')


    HAP_sum = np.mean(HAP_full, axis=(0, 2))
    #HAP_sum = np.sum(HAP_full, axis=(0, 2))

   

    

    #plt.imshow(BAFMatrix, cmap='bwr')
    #plt.show()
    #quit()


    #plt.hist(BAFMatrix[BAFMatrix>=0].reshape((-1,)) , bins=100 )
    #plt.show()
    #quit()

    #BAFMatrix[BAFMatrix<0] = (BAFMatrix[BAFMatrix<0] + 1e-5 )/0

    '''
    perm1 = np.random.permutation(predCNA.shape[0])

    M = 10
    HAP_sum = rebin(HAP_sum.T, M).T
    HAP_sum = HAP_sum[np.arange(HAP_sum.shape[0]).repeat(M) ]

    #perm1 = np.array([674])
    for a in range(perm1.shape[0]):
        print (perm1[a])
        HAP1 = HAP_full[perm1[a]]

        #BAF1 = HAP1[:, 1] / (np.sum(HAP1, axis=1) + 1e-5)
        sumPred1 = np.sum(predCNA[perm1[a]], axis=1).astype(float)
        sumNaive1 = np.sum(naiveCNA[perm1[a]], axis=1).astype(float)
        #sumPred1 = sumPred1 / np.mean(sumPred1)

        sumOther1 = np.sum(copyMatrix[perm1[a]], axis=1  ).astype(float)
        #sumOther1 = sumOther1 / np.mean(sumOther1)

        BAFPred1 = predCNA[perm1[a], :, 1] /  (np.sum(predCNA[perm1[a]], axis=1).astype(float) + 1e-5)
        BAFOther1 = copyMatrix[perm1[a], :, 1] /  (np.sum(copyMatrix[perm1[a]], axis=1).astype(float) + 1e-5)
        
        
        M = 5
        RDR1 = rebin(RDR_full[perm1[a]], M)
        HAP1 = rebin(HAP1.T, M).T
        BAF1 = HAP1[:, 1] / (np.sum(HAP1, axis=1) + 1e-5)
        BAF1 = BAF1[np.arange(RDR1.shape[0]).repeat(M) ]
        HAP1 = HAP1[np.arange(RDR1.shape[0]).repeat(M) ]

        
        


        BAF2 = BAFMatrix[perm1[a]]

        if True:
            plt.plot(RDR1[np.arange(RDR1.shape[0]).repeat(M)  ] * np.mean(sumNaive1)  )
            #plt.plot(sumPred1 / np.mean(sumPred1))
            plt.plot(sumNaive1 )#/ np.mean(sumNaive1))
            #plt.plot(sumOther1 / np.mean(sumOther1))
            True

        if False:
            plt.plot(  np.argwhere(BAF2 >=0)[:, 0], BAF2[BAF2>= 0]   )
            plt.plot(  BAF1 )
            #plt.plot(  np.log(np.sum(HAP1, axis=1)) * 0.2  )
            #plt.plot(np.log(HAP_sum) * 0.2)
            #plt.plot(BAFPred1)
            #plt.plot(BAFOther1, c='red')
            #plt.plot( 1 -  BAFOther1 , c='red')

        if False:
            plt.plot(  np.argwhere(BAF2 >=0)[:, 0], BAF2[BAF2>= 0]   )
            plt.plot(BAFPred1)
            plt.plot(BAFOther1)
            plt.plot()
        plt.show()
    quit()
    #'''


    BAF_naive = naiveCNA[:, :, 1] / (np.sum(naiveCNA, axis=2) + 1e-5)
    BAF_pred = predCNA[:, :, 1] / (np.sum(predCNA, axis=2) + 1e-5)
    BAF_other = copyMatrix[:, :, 1] / (np.sum(copyMatrix, axis=2) + 1e-5)

    sum_pred = np.sum(predCNA, axis=2)
    sum_other = np.sum(copyMatrix, axis=2)
    sum_naive = np.sum(naiveCNA, axis=2)
    sum_pred[sum_pred>10] = 10
    sum_other[sum_other > 10] = 10
    sum_naive[sum_naive > 10] = 10

    #sns.clustermap(  sum_pred  , col_cluster=False, row_cluster=True,linewidths=0.0)
    #plt.show()

    HAP_full = HAP_full.astype(float)
    BAF_full = (HAP_full[:, :, 1]  + 1e-5 )/ (np.sum(HAP_full, axis=2) + 2e-5)
    

    BAFMatrix[BAFMatrix<0] = 0.5

    f, axarr = plt.subplots(2)
    #sns.clustermap(  sum_pred  , col_cluster=False, row_cluster=True,linewidths=0.0)
    #plt.show()
    #axarr[0].imshow(BAF_pred[:, chr == 2], cmap='bwr')
    #axarr[1].imshow(BAF_other[:, chr == 2], cmap='bwr')

    #sns.clustermap(  BAF_other[:, chr == 2]  , col_cluster=False, row_cluster=True,linewidths=0.0)
    #plt.show()
    #quit()

    #args2 = np.argwhere(  np.logical_and( np.max(predCNA[:, 200], axis=1) == 3, np.min(predCNA[:, 200], axis=1) == 2  )   )[:, 0]
    #args2 = args2[5:] #Getting rid of a few early bad cases

    #print (subset2[args2])
    #quit()

    #haplotypePlotter(copyMatrix[:, :1000].astype(int))
    #haplotypePlotter(predCNA[:, :5000].astype(int), doCluster=False)
    #quit()

    


#alignMethods()
#quit()


def plotHeatmaps():

    folder1 = 'DLP'
    #folder1 = '10x'
    #method1 = 'signals'
    #method1 = 'chisel'
    #folder2 = folder1 + '_' + method1
    

    #from scaler import haplotypePlotter
    

    
    predCNA = loadnpz('./data/comparison/CNA/' + folder1 + '_deep.npz').astype(float)
    naiveCNA = loadnpz('./data/comparison/CNA/' + folder1 + '_naive.npz').astype(float)
    signalsCNA = loadnpz('./data/comparison/CNA/' + folder1 + '_signals.npz').astype(float)
    chiselCNA = loadnpz('./data/comparison/CNA/' + folder1 + '_chisel.npz').astype(float)
    
    subset2 = loadnpz('./data/comparison/CNA/' + folder1 + '_subsetOurs.npz')
    chr = loadnpz('./data/comparison/CNA/' + folder1 + '_chr.npz')

    qBool = loadnpz('./data/comparison/chrArm/' + folder1 + '_Qbool.npz')

    argGoodOurs = loadnpz('./data/comparison/CNA/' + folder1 + '_argSubsetOurs.npz')
    chrAll = loadnpz('./data/' + folder1 + '/initial/allChr_100k.npz')
    

    #print (predCNA.shape)
    #quit()
 
    

    if False:
        from scipy.cluster.hierarchy import linkage
        linkage_matrix = linkage(predCNA.reshape((predCNA.shape[0],  predCNA.shape[1] * 2 ))  , method='ward', metric='euclidean')
        np.savez_compressed('./data/comparison/CNA/' + folder2 + '_linkageOurs.npz', linkage_matrix)
        quit()

    #linkage_matrix = loadnpz('./data/comparison/CNA/' + folder2 + '_linkageOurs.npz')

    


    
    from scipy.cluster.hierarchy import linkage
    ##sns.clustermap( predTotal  , col_cluster=False, row_cluster=True,linewidths=0.0)

    if True:
        
        np.random.seed(0)

        N = 200
        #perm1 = np.random.permutation(predCNA.shape[0])
        #np.savez_compressed('./data/comparison/CNA/zoomedPerm_' + folder1 + '.npz', perm1)
        perm1 = loadnpz('./data/comparison/CNA/zoomedPerm_' + folder1 + '.npz')

        perm1 = perm1[:N]

        #print (perm1[:5])
        #quit()
        if folder1 == '10x':
            #args1 = np.argwhere(np.isin(chr+1,  np.array([6]) ) )[:, 0]
            chr_startBin = np.argwhere(chrAll+1 == 6)[0, 0]
            args1 = np.argwhere(np.isin(chr+1,  np.array([6]) ) )[:, 0]
        else:
            args1 = np.argwhere(np.isin(chr+1,  np.array([1]) ) )[:, 0]
            chr_startBin = np.argwhere(chrAll+1 == 1)[0, 0]
        predCNA = predCNA[perm1][:, args1]
        naiveCNA = naiveCNA[perm1][:, args1]
        signalsCNA = signalsCNA[perm1][:, args1]
        chiselCNA = chiselCNA[perm1][:, args1]
        chr1 = qBool[args1]

        #plt.plot(chr1)
        #plt.show()
        #quit()

        

        argPos = argGoodOurs[args1] - chr_startBin

        #print (np.round(argPos[100::100] / 10))
        #print (chr1[100::100]))
        #quit()

        #predCNA[:, 0::100] = 10

        #plt.imshow(predCNA[:, :, 0])
        #plt.show()
        #quit()

        print (np.unique(uniqueProfileMaker(predCNA)).shape)
        print (np.unique(uniqueProfileMaker(signalsCNA)).shape)
        print (np.unique(uniqueProfileMaker(chiselCNA)).shape)
        quit()



        print ('A')
        
        linkage_matrix = linkage(predCNA.reshape((predCNA.shape[0],  predCNA.shape[1] * 2 ))  , method='ward', metric='euclidean')

        #haplotypePlotter(predCNA.astype(int), doCluster=True, chr=[chr1], withLinkage=[linkage_matrix], saveFile='./images/heatmap/compare_' + folder1 + '_deep_zoom_temp2.png', plotSize=[10, 4])

        haplotypePlotter(predCNA.astype(int), doCluster=True, chr=[], withLinkage=[linkage_matrix], saveFile='./images/heatmap/compare_' + folder1 + '_deep_zoom_line1.png', plotSize=[20, 8], vertLine=[100])
        haplotypePlotter(naiveCNA.astype(int), doCluster=True, chr=[], withLinkage=[linkage_matrix], saveFile='./images/heatmap/compare_' + folder1 + '_naive_zoom_line1.png', plotSize=[20, 8], vertLine=[100])
        haplotypePlotter(signalsCNA.astype(int), doCluster=True, chr=[], withLinkage=[linkage_matrix], saveFile='./images/heatmap/compare_' + folder1 + '_signals_zoom_line1.png', plotSize=[20, 8], vertLine=[100])
        haplotypePlotter(chiselCNA.astype(int), doCluster=True, chr=[], withLinkage=[linkage_matrix], saveFile='./images/heatmap/compare_' + folder1 + '_chisel_zoom_line1.png', plotSize=[20, 8], vertLine=[100])
        quit()
    
    if False:
        print ('hi')
        
        linkage_matrix = linkage(predCNA.reshape((predCNA.shape[0],  predCNA.shape[1] * 2 ))  , method='ward', metric='euclidean')

        haplotypePlotter(predCNA.astype(int), doCluster=True, chr=[chr], withLinkage=[linkage_matrix], saveFile='./images/heatmap/compare_' + folder1 + '_deep.png', plotSize=[10, 4])
        haplotypePlotter(naiveCNA.astype(int), doCluster=True, chr=[chr], withLinkage=[linkage_matrix], saveFile='./images/heatmap/compare_' + folder1 + '_naive.png', plotSize=[10, 4])
        haplotypePlotter(signalsCNA.astype(int), doCluster=True, chr=[chr], withLinkage=[linkage_matrix], saveFile='./images/heatmap/compare_' + folder1 + '_signals.png', plotSize=[10, 4])
        haplotypePlotter(chiselCNA.astype(int), doCluster=True, chr=[chr], withLinkage=[linkage_matrix], saveFile='./images/heatmap/compare_' + folder1 + '_chisel.png', plotSize=[10, 4])



plotHeatmaps()
quit()
#    







def saveDistMatrix():

    
    
    




    folder1 = '10x'
    #folder1 = 'DLP'
    #method1 = 'signals'
    #method1 = 'chisel'
    #folder2 = folder1 + '_' + method1

    chr = loadnpz('./data/comparison/CNA/' + folder1 + '_chr.npz')
    #predCNA = loadnpz('./data/comparison/CNA/' + folder1 + '_deep.npz')#.astype(float)
    #signals = loadnpz('./data/comparison/CNA/' + folder1 + '_signals.npz')
    #chisel = loadnpz('./data/comparison/CNA/' + folder1 + '_chisel.npz')#.astype(float)
    naive = loadnpz('./data/comparison/CNA/' + folder1 + '_naive.npz')


    #distMatrix_pred = calcDiffMatrix(predCNA, chr, doMissing=False)
    #distMatrix_signals = calcDiffMatrix(signals, chr, doMissing=False)
    #distMatrix_chisel = calcDiffMatrix(chisel, chr, doMissing=False)
    distMatrix_naive = calcDiffMatrix(naive, chr)
    
    
    
    #np.savez_compressed('./data/comparison/tree/dist_' + folder1 + '_deep.npz', distMatrix_pred)
    #np.savez_compressed('./data/comparison/tree/dist_' + folder1 + '_signals.npz', distMatrix_signals)
    #np.savez_compressed('./data/comparison/tree/dist_' + folder1 + '_chisel.npz', distMatrix_chisel)
    np.savez_compressed('./data/comparison/tree/dist_' + folder1 + '_naive.npz', distMatrix_naive)


#saveDistMatrix()
#quit()







def findTree():


    #folder1 = 'DLP'
    folder1 = '10x'
    #method1 = 'signals'
    #method1 = 'chisel'
    #folder2 = folder1 + '_' + method1
    
    #compareMethod = 'ours'
    #compareMethod = ''
    #compareMethod = 'random'
    #folder3 = folder2
    #if compareMethod == 'ours':
    #    folder3 = folder3 + '_' + compareMethod


    import dendropy

    from skbio import DistanceMatrix
    from skbio.tree import nj
    import sys

    sys.setrecursionlimit(10000) 



    #data_deep = loadnpz('./data/comparison/tree/dist_' + folder1 + '_deep.npz')
    #data_signals = loadnpz('./data/comparison/tree/dist_' + folder1 + '_signals.npz')
    #data_chisel = loadnpz('./data/comparison/tree/dist_' + folder1 + '_chisel.npz')
    data_naive = loadnpz('./data/comparison/tree/dist_' + folder1 + '_naive.npz')
    

    #matrix1_deep, tree1_deep = getTree(data_deep)
    #matrix1_signals, tree1_signals = getTree(data_signals)
    #matrix1_chisel, tree1_chisel = getTree(data_chisel)
    matrix1_naive, tree1_naive = getTree(data_naive)

    
    #np.savez_compressed('./data/comparison/tree/tree_' + folder1 + '_deep.npz',  np.array([tree1_deep]))
    #np.savez_compressed('./data/comparison/tree/tree_' + folder1 + '_signals.npz',  np.array([tree1_signals]))
    #np.savez_compressed('./data/comparison/tree/tree_' + folder1 + '_chisel.npz',  np.array([tree1_chisel]))
    np.savez_compressed('./data/comparison/tree/tree_' + folder1 + '_naive.npz',  np.array([tree1_naive]))

    #np.savez_compressed('./data/comparison/tree/clades_' + folder1 + '_deep.npz',  matrix1_deep)
    #np.savez_compressed('./data/comparison/tree/clades_' + folder1 + '_signals.npz',  matrix1_signals)
    #np.savez_compressed('./data/comparison/tree/clades_' + folder1 + '_chisel.npz',  matrix1_chisel)
    np.savez_compressed('./data/comparison/tree/clades_' + folder1 + '_naive.npz',  matrix1_naive)
                 

#findTree()
#quit()



#from Bio import Align, Phylo
#from Bio.Phylo import TreeConstruction
#scorer = Phylo.TreeConstruction.ParsimonyScorer()
#quit()


def doTreeParsimony():


    
    

    def listRemove1(list1):

        list2 = []
        for a in range(len(list1)):
            if a != 1:
                list2.append(list1[a])
        return list2
    

        


        

    #folder1 = 'DLP'
    folder1 = '10x'


    #tree1 = loadnpz('./data/comparison/tree/tree_' + folder1 + '_deep.npz')[0]
    #tree1 = modifyTree(tree1)
    #predCNA = loadnpz('./data/comparison/CNA/' + folder1 + '_deep.npz')

    if False:
    
        for method1 in ['naive']:#  ['deep', 'signals', 'chisel']:
            print (method1)


            tree1 = loadnpz('./data/comparison/tree/tree_' + folder1 + '_' + method1 + '.npz')[0]
            tree1 = modifyTree(tree1)
            predCNA = loadnpz('./data/comparison/CNA/' + folder1 + '_' + method1 + '.npz')

            chr = loadnpz('./data/comparison/CNA/' + folder1 + '_chr.npz')

            tree_original = tree1
            treeInternal_choice, cladeSizes, pairList, pairListLength, errors, treeWithLength = runParsimony(tree1, predCNA, chr)

            #print (len(treeWithLength.split(')')))
            #print (len(treeWithLength.split('(')))
            #quit()

            #print (pairListLength)
            #quit()

            np.savez_compressed('./data/comparison/parsimony/internalVals_' + folder1 + '_' + method1 + '.npz', treeInternal_choice )
            np.savez_compressed('./data/comparison/parsimony/cladeSizes_' + folder1 + '_' + method1 + '.npz', cladeSizes )
            np.savez_compressed('./data/comparison/parsimony/cladePairs_' + folder1 + '_' + method1 + '.npz', pairList )
            np.savez_compressed('./data/comparison/parsimony/pairListLength_' + folder1 + '_' + method1 + '.npz', pairListLength )
            np.savez_compressed('./data/comparison/parsimony/errors_' + folder1 + '_' + method1 + '.npz', errors )
            np.savez_compressed('./data/comparison/parsimony/treeWithLength_' + folder1 + '_' + method1 + '.npz', np.array([treeWithLength]) )

    
    #quit()
    if True:

        errors_deep = loadnpz('./data/comparison/parsimony/errors_' + folder1 + '_deep.npz')
        errors_signals = loadnpz('./data/comparison/parsimony/errors_' + folder1 + '_signals.npz')
        errors_chisel = loadnpz('./data/comparison/parsimony/errors_' + folder1 + '_chisel.npz')
        errors_naive = loadnpz('./data/comparison/parsimony/errors_' + folder1 + '_naive.npz')
        #errors_naive = [0, 0]

        methodList = ['DeepCopy', 'NaiveCopy', 'SIGNALS', 'CHISEL']
        errorList = [errors_deep[0], errors_naive[0], errors_signals[0], errors_chisel[0]]
        palette = ['blue', 'lightblue', 'orange', 'green']

        
        #print (errorList)
        #quit()


        methodList = listRemove1(methodList)
        errorList = listRemove1(errorList)
        palette = listRemove1(palette)

        



        plotData = {}
        plotData['index'] = methodList
        plotData['parsimony (ZCNT)'] = errorList
        plotData['label1'] = methodList

        import pandas as pd

        

        df = pd.DataFrame(data=plotData)


        ax = sns.barplot(df, x='index', y='parsimony (ZCNT)', label='label1', palette=palette)

        #plt.yscale('log')
        
        plt.xticks(rotation = 90)
        plt.gcf().set_size_inches(2.5, 3)

        #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        #ax.style('sci')



        
        saveFile = './images/parsimony/scores_' + folder1 + '.pdf'

        plt.tight_layout()
        plt.savefig(saveFile)
        plt.show()
        
    quit()

    if True:
    
        for method1 in ['deep', 'signals', 'chisel']:
            print (method1)


            tree1 = loadnpz('./data/comparison/tree/tree_' + folder1 + '_' + method1 + '.npz')[0]
            tree1 = modifyTree(tree1)
            predCNA = loadnpz('./data/comparison/CNA/' + folder1 + '_' + method1 + '.npz')
            chr = loadnpz('./data/comparison/CNA/' + folder1 + '_chr.npz')

            predCNA = insertBoundaries(predCNA, chr)

            

            tree_original = tree1

            treeInternal = loadnpz('./data/comparison/parsimony/internalVals_' + folder1 + '_' + method1 + '.npz')
            cladeSizes = loadnpz('./data/comparison/parsimony/cladeSizes_' + folder1 + '_' + method1 + '.npz')
            pairList = loadnpz('./data/comparison/parsimony/cladePairs_' + folder1 + '_' + method1 + '.npz')
            errors = loadnpz('./data/comparison/parsimony/errors_' + folder1 + '_' + method1 + '.npz')

            tree1 = 'C' + str(len(pairList) - 1)

            for a0 in range(len(pairList)):
                a = len(pairList) - 1 - a0

                if cladeSizes[a] > 200:

                    name1 = 'C' + str(a)

                    pairNow = pairList[a]
                    pairNow_str = '(' + pairNow + ')'

                    #print (pairNow_str)

                    tree1 = tree1.replace(name1, pairNow_str)

                    pairNow = pairList[a]
                    pairNow_str = '(' + pairNow + ')'
                    pairNow = pairNow.split(',')
                    
                    if len(pairNow) == 3:
                        pairNow.remove( str(int(predCNA.shape[0]))  )

                    valueParent = a
                    valueChild1 = pairNow[0]
                    valueChild2 = pairNow[1]

                    vectorParent = treeInternal[valueParent]

                    #print (valueChild1, valueChild2)

                    if valueChild1[0] == 'C':
                        vector1 = treeInternal[int(valueChild1[1:])]
                    else:
                        vector1 = predCNA[int(valueChild1)]
                    if valueChild2[0] == 'C':
                        vector2 = treeInternal[int(valueChild2[1:])]
                    else:
                        vector2 = predCNA[int(valueChild2)]



                    error1 = calculateZNT(vectorParent, vector1)
                    error2 = calculateZNT(vectorParent, vector2)

                    print (error1, error2)

                    




            #assert tree1 == tree_original

            print (tree1)
                



            #quit()


    quit()

    



doTreeParsimony()
quit()



def saveArm():

    chrValid = (np.arange(22)+1).astype(int).astype(str)

    if False:
        file1 = open('./data/comparison/chrArm/chromeArm.txt', 'r')
        Lines = file1.readlines()
        Lines = Lines[1:-1]
        
        posList = np.zeros((22, 2), dtype=int)

        for a in range(len(Lines)):
            line1 = Lines[a]
            chr1 = line1.split('chr')[1].split(' ')[0]

            if chr1 in chrValid:
                chrInt = int(chr1) - 1
                pq = 0
                if ' q ' in line1:
                    pq = 1

                pos1 = line1.split(' ')[-1][:-1]
                pos1 = int(pos1)

                posList[chrInt, pq] = pos1

        np.savez_compressed('./data/comparison/chrArm/chromeArmHG19.npz', posList)

    if False:
        data = np.loadtxt('./data/comparison/chrArm/hg38ARM.csv', delimiter=',', dtype=str)

        posList = np.zeros((22, 2), dtype=int)

        for a in range(data.shape[0]):
            if len(data[a][1]) < 10:

                chr1 = data[a][1].split('chr')[1]
                chr1 = chr1.replace('"', '')
                if chr1 in chrValid:
                    chrInt = int(chr1) - 1

                    pq = 0
                    if 'q' in data[a][2]:
                        pq = 1

                    pos1 = int(data[a][3])

                    posList[chrInt, pq] = pos1
        
        np.savez_compressed('./data/comparison/chrArm/chromeArmHG38.npz', posList)

    #quit()

    for folder1 in ['10x', 'DLP']:
        
        if folder1 == '10x':
            posList = loadnpz('./data/comparison/chrArm/chromeArmHG38.npz')
        if folder1 == 'DLP':
            posList = loadnpz('./data/comparison/chrArm/chromeArmHG19.npz')

        print (posList)
        quit()

        argGoodOurs = loadnpz('./data/comparison/CNA/' + folder1 + '_argSubsetOurs.npz')

        
        #chr = loadnpz('./data/comparison/CNA/' + folder1 + '_chr.npz')
        chrAll = loadnpz('./data/' + folder1 + '/initial/allChr_100k.npz')
        qBool1 = np.zeros( chrAll.shape[0] , dtype=int)
        #chrConvert = np.zeros(chrAll.shape[0], dtype=int)
        for a in range(22):
            args1 = np.argwhere(chrAll == a)[:, 0]
            pLength = posList[a, 0] // 100000

            #print (pLength)
            #print (args1.shape)

            qBool1[args1[pLength:]] = 1

            #print (np.unique(qBool1[args1]))



        boolGood = np.zeros(chrAll.shape[0])
        boolGood[argGoodOurs] = 1

        #plt.plot(boolGood)
        #plt.plot(qBool1)
        #plt.show()

        #print (np.sum(  np.abs(argGoodOurs - np.sort(argGoodOurs)) ))
        #quit()

        qBool2 = qBool1[argGoodOurs]

        #plt.plot(qBool2)
        #plt.show()

        np.savez_compressed('./data/comparison/chrArm/' + folder1 + '_Qbool.npz', qBool2)

        #quit()



#saveArm()
#quit()


def saveOnco():

    import csv 

    file1 = open('./data/comparison/other/Census_allWed.csv', 'r')
    Lines = file1.readlines()
    print (len(Lines))
    Lines = [l for l in csv.reader(Lines, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)]
    Lines = np.array(Lines)

    #print (Lines[:5])
    #quit()

    #Lines = Lines[:, np.array([3, 11, 12, 14])]
    Lines = Lines[:, np.array([3, 9, 10, 14, 0])]

    Lines2 = []
    for a in range(1, len(Lines)):
        line0 = Lines[a]
        #print (line0)
        chr1 = line0[0].split(':')
        chr1, pos1 = chr1[0], chr1[1]
        pos1 = pos1.split('-')
        pos1, pos2 = pos1[0], pos1[1]
        line1 = [chr1, pos1, pos2, line0[1], line0[2], line0[3], line0[4]]
        Lines2.append(line1)
    Lines = np.array(Lines2)

    #print (Lines)
    #quit()

    goodChr = (np.arange(22)+1).astype(int).astype(str)

    
    good1_breast = np.zeros(Lines.shape[0], dtype=int)
    good1_ovarian = np.zeros(Lines.shape[0], dtype=int)
    for a in range(Lines.shape[0]):

        if True:#('oncogene' in Lines[a, 5]) != ('TSG' in Lines[a, 5]): #!= gives exclusive or

            chromesome = Lines[a, 0]#.split(':')[0]

            if False:
                if 'oncogene' in Lines[a, 5]:
                    Lines[a, 5] = 'oncogene'
                if 'TSG' in Lines[a, 5]:
                    Lines[a, 5] = 'TSG'

            #print (chromesome)
            if chromesome in goodChr:
                if ('breast' in Lines[a, 3]) or ('breast' in Lines[a, 4]):
                    good1_breast[a] = 1
                if ('ovarian' in Lines[a, 3]) or ('ovarian' in Lines[a, 4]):
                    good1_ovarian[a] = 1
    
    #print (Lines.shape)

    Lines = Lines[:, np.array([0, 1, 2, 4, 5, 6])]

    Lines_breast = Lines[good1_breast == 1]
    Lines_ovar = Lines[good1_ovarian == 1]

    print (Lines_breast.shape)
    print (Lines_ovar.shape)

    np.savez_compressed('./data/comparison/other/onco_breast_bigger.npz', Lines_breast)
    np.savez_compressed('./data/comparison/other/onco_ovarian_bigger.npz', Lines_ovar)

#saveOnco()
#quit()


def checkOncoGene():

    #folder1 = 'DLP'
    folder1 = '10x'

    argGoodOurs = loadnpz('./data/comparison/CNA/' + folder1 + '_argSubsetOurs.npz')
    chr = loadnpz('./data/comparison/CNA/' + folder1 + '_chr.npz')
    chrAll = loadnpz('./data/' + folder1 + '/initial/allChr_100k.npz')
    

    if folder1 == 'DLP':
        gene = loadnpz('./data/comparison/other/onco_ovarian.npz')
    if folder1 == '10x':
        gene = loadnpz('./data/comparison/other/onco_breast.npz')

    #print (gene.shape)
    #quit()


    predCNA = loadnpz('./data/comparison/CNA/' + folder1 + '_deep.npz')
    #predCNA = loadnpz('./data/comparison/CNA/' + folder1 + '_signals.npz')


    chrConvert = np.zeros(chrAll.shape[0], dtype=int)
    for a in range(22):
        args1 = np.argwhere(chrAll == a)[:, 0]
        chrConvert[args1] = args1 - args1[0]

    #print (gene)

    geneIndexs = []

    #print (np.argwhere(chr == 16).shape)
    #quit()


    for a in range(predCNA.shape[0]):

        

        pred1 = predCNA[a]
        for b in range(22):
            args1 = np.argwhere(chr == b)[:, 0]

            posTrue = chrConvert[argGoodOurs[args1]] * 100000



            pred2 = pred1[args1]
            changePoints = np.sum(np.abs(pred2[1:] - pred2[:-1]), axis=1)
            diff1 = np.argwhere(changePoints != 0)[:, 0]

            
            diff2 = diff1[1:] - diff1[:-1]

            diff1 = np.array([diff1[:-1], diff1[1:]]).T 
            #diff1 = diff1[diff2 <= 20]

            values = pred2[diff1[:, 0]+1]
            valuesBefore = pred2[diff1[:, 0]]

            N = 40
            values = values[diff2 <= N]
            diff1 = diff1[diff2 <= N]

            diff1 = posTrue[diff1]


            diff1[:, 0] = diff1[:, 0] - 500000
            diff1[:, 1] = diff1[:, 1] + 500000

            #diff1[:, 0] = diff1[:, 0] - 500000
            #diff1[:, 1] = diff1[:, 1] + 500000
            #print (gene)
            #print (diff1)

            for c in range(gene.shape[0]):
                if str(b + 1) == gene[c, 0]:
                    genePos1 = int(gene[c, 1])
                    genePos2 = int(gene[c, 2])
                    #if b == 16:
                    #  print (diff1)
                    #   print (gene[c])
                    overlap1 = np.argwhere( np.logical_and( diff1[:, 0] < genePos2, diff1[:, 1] > genePos1,  ) )[:, 0]


                    if overlap1.shape[0] > 0:
                        geneIndexs.append(c)
                        #print (valuesBefore[overlap1])
                        #print (values[overlap1])
                        #print (gene[c])

        #if a == 100:
        #    quit()

    geneIndexs = np.array(geneIndexs).astype(int)
    geneIndexs = np.unique(geneIndexs)

    print (gene[geneIndexs])






#checkOncoGene()
#quit()





def plotVertTree():


    def give10xClusters():

        cluster1 = np.loadtxt('./data/comparison/mapping.tsv', delimiter='\t', dtype=str)

        cellNames = loadnpz('./data/' + folder1 + '/initial/cellNames.npz')
        subset_ours = loadnpz('./data/comparison/CNA/' + folder1 + '_subsetOurs.npz')
        cellNames = cellNames[subset_ours]

        #_, counts1 = np.unique(cluster1, return_counts=True)
        #print (np.sort(counts1)[-1::-1][:10])
        #quit()

        cluster1 = cluster1[np.isin(cluster1[:, 0],  cellNames)]
        cluster1 = cluster1[np.argsort(cluster1[:, 0])]

        for a in range(cellNames.shape[0]):
            assert cellNames[a] == cluster1[a, 0]
        
        cluster1 = cluster1[:, 1]
        _, cluster1, cluster1_counts = np.unique(cluster1, return_counts=True, return_inverse=True)
        N = 50
        cluster1[cluster1_counts[cluster1] < N] = cluster1.shape[0]
        _, cluster1 = np.unique(cluster1, return_inverse=True)




    def simplifyClusterTree(tree1, pairVals, cladeSizes, Nmin, Nmin2):

        

        pairInfo = np.zeros(( len(pairList), 2 ), dtype=int)

        
        #Nmin2 = 10

        #Nmin = 150
        #Nmin2 = 150

        for a in range(len(pairVals)):
            pairVal = pairVals[a]
            #pairVal_str = '(' + pairVal + ')'
            pairVal_str = pairVal

            pairVal = pairVal[1:-1].split(',')

            cladeSize1 = 1
            cladeSize2 = 1

            if pairVal[0][0] == 'C':
                key1 = int(pairVal[0].split(':')[0][1:])
                cladeSize1 = cladeSizes[key1]
            if pairVal[1][0] == 'C':
                key2 = int(pairVal[1].split(':')[0][1:])
                cladeSize2 = cladeSizes[key2]

            dist1 = float(pairVal[0].split(':')[1])
            dist2 = float(pairVal[1].split(':')[1])
            distBoth = dist1 + dist2
            #print (distBoth)

            if (cladeSize1 < Nmin) or (cladeSize2 < Nmin) or (distBoth == 0):
                cladeName = 'C' + str(a)
                #print (pairVal_str)
                #print (pairVal_str in tree1)
                tree1 = tree1.replace(pairVal_str, cladeName)
            
            #print (cladeSize1, cladeSize2)
            #quit()

        #(206:0.0,210:0.0)

        lastPoint = 0
        newPoint = 0
        a = 0
        while a < len(tree1):
            if tree1[a] in ['(', ')',',', ':']:
                lastPoint = newPoint
                newPoint = a
                if newPoint - lastPoint >= 2:
                    if tree1[a] == ':':
                        key1 = tree1[lastPoint+1:newPoint]
                        cladeSize1 = 1
                        if 'C' == key1[0]:
                            cladeSize1 = cladeSizes[int(key1[1:])]
                        
                        if cladeSize1 < Nmin2:

                            b = newPoint
                            while tree1[b] not in [',', '(', ')']:
                                b += 1

                            a = lastPoint - 1
                            #print (tree1[lastPoint+1:b])
                            #quit()
                            tree1 = tree1[:lastPoint+1] + tree1[b:]


                        #print ([tree1[lastPoint+1:newPoint]])
                        #if (newPoint - lastPoint >= 2) and (tree1[lastPoint:][:2] != '):'):

            a += 1



            
        a = 0
        while a < len(tree1):
            if tree1[a:a+4] == '(,(,':
                paren1 = 1
                paren2 = 0
                b = a
                while paren1 != paren2:
                    b += 1
                    if tree1[b] == '(':
                        paren1 += 1
                    if tree1[b] == ')':
                        paren2 += 1

                hitColon = 0
                b2 = b
                while hitColon < 2:
                    if tree1[b2] == ':':
                        hitColon += 1
                    b2 -= 1
                b2 += 1

                lengthPart = tree1[b2:b]
                #print (lengthPart)
                #print (lengthPart)
                lengthPart = lengthPart.replace(',', '')
                lengthPart = lengthPart.replace(')', '')
                lengthPart = lengthPart.replace('(', '')
                lengthPart = lengthPart[1:].split(':')
                lengthPart = float(lengthPart[0]) + float(lengthPart[1])
                lengthPart2 = ':' + str(lengthPart) + ')' 
                #print ("A")
                #print (tree1)

                
                tree1 = tree1[:a] + tree1[a+2:b2] + lengthPart2 + tree1[b+1:]
                
                #print (tree1)
                #quit()
                a = a - 1
                #quit()


            a += 1


        #print (tree1)
        #quit()

        #print (tree1)

        tree1 = tree1.replace('(,', '(')
        tree1 = tree1.replace(',)', ')')


        if tree1[:2] == '((':
            tree1_list = tree1.split(':')
            #tree1_list[-1] = '0.0)'
            tree1_list = tree1_list[:-1]
            tree1 = ':'.join(tree1_list)
            tree1 = tree1[1:]

        #print (tree1)
        #quit()

        

        return tree1






    


    def giveParents(name1, pairList):
        
        oldLength = -1
        parentList = [name1]
        while len(parentList) != oldLength:
            oldLength = len(parentList)
            name2 = ''
            for a in range(len(pairList)):
                #if parentList[-1] in pairList[a]:
                if parentList[-1] + ':' in pairList[a]:
                    name2 = 'C' + str(a)
            if name2 != '':
                parentList.append(name2)

        return parentList
    
    def giveLeastCommonAncestor(name1, name2, pairList):

        parentList1 = giveParents(name1, pairList)
        parentList2 = giveParents(name2, pairList)

        

        a = 0
        while parentList1[a] not in parentList2:
            a += 1

        assert parentList1[a] in parentList2
        
        return parentList1[a]
    

    def getEdgeCNA(name1, name2, name3, pairList, treeInternal_choice, chr, qBool):

        common1 = giveLeastCommonAncestor(name1, name2, pairList)
        if name3 != '':
            common2 = giveLeastCommonAncestor(name1, name3, pairList)
            common3 = giveLeastCommonAncestor(name2, name3, pairList)
            print (common2, common3)
            assert common2 == common3 

        else:
            common2 = common1
            common1 = name1

        common1 = int(common1[1:])
        common2 = int(common2[1:])
        CNA1 = treeInternal_choice[common1]
        CNA2 = treeInternal_choice[common2]

        #diff1 = np.sum(np.abs(CNA1 - CNA2), axis=1) 
        diff1 = CNA1 - CNA2

        diff1_sum = np.sum(np.abs(diff1))
        diffWGD_sum = np.sum(np.abs(CNA1 - (CNA2 * 2)   ) )
        if diffWGD_sum < diff1_sum:
            print ("WGD")
            diff1 = CNA1 - (CNA2 * 2)
        else:
            print ('noWGD')

        #plt.plot(diff1[chr==16, 0])
        #plt.plot(diff1[chr==16, 1])
        #plt.show()
        #plt.plot(diff1[:, 0])
        #plt.plot(diff1[:, 1])
        #plt.show()
        grid1 = np.zeros((2, 2), dtype=int)
        #editedChr = []

        eventList = []
        for a in range(22):
            args1 = np.argwhere(chr == a)[:, 0]

            
            grid1[0, 0] += min(np.min(diff1[args1, 0]),0)
            grid1[0, 1] += max(np.max(diff1[args1, 0]),0)
            grid1[1, 0] += min(np.min(diff1[args1, 1]),0)
            grid1[1, 1] += max(np.max(diff1[args1, 1]),0)

            diff2 = diff1[args1]

            matrix1 = np.zeros((2, 2, 2), dtype=int)

            for b in range(2):
                for c in range(2):
                    diff3 = diff2[ qBool[args1] == b , c ]

                    if diff3.shape[0] > 0:
                    
                        pq = ['p', 'q'][b]

                        min1, max1 = np.min(diff3), np.max(diff3)
                        if min1 < 0:
                            matrix1[b, c, 0] = 1
                            #eventList.append([str(a+1), pq, 'del', min1])
                        if max1 > 0:
                            matrix1[b, c, 1] = 1
                            #eventList.append([str(a+1), pq, 'dup', max1])

            for b in range(2):
                alleleName = ['A', 'B'][b]
                for c in range(2):
                    delAmp = ['del', 'dup'][c]
                    if matrix1[0, b, c] != matrix1[1, b, c]:
                        if matrix1[0, b, c] == 1:
                            eventList.append([str(a+1), 'p', alleleName, delAmp])
                        if matrix1[1, b, c] == 1:
                            eventList.append([str(a+1), 'q', alleleName, delAmp])
                    else:
                        if matrix1[0, b, c]:
                            eventList.append([str(a+1), '', alleleName, delAmp])

                    
                




            #sum1 = np.sum(np.abs(diff1[args1]))
            #if sum1 != 0:
            #    editedChr.append(a+1)

        eventNames = []
        for a in range(len(eventList)):
            event1 = eventList[a]
            #pqName = ''
            #if len(event1[1]) != 0:
            #    pqName = '.' + event1[1]
            name1 = 'Chr' + event1[0] + event1[1] + ' ' + event1[2] + ' ' +  event1[3]
            eventNames.append(name1)

        grid2 = np.sum(grid1, axis=0).astype(int)
        delNum, dupNum = str(abs(grid2[0])), str(abs(grid2[1]))

        gridName = delNum + ' del ' + dupNum + ' dup'
        
        
        return eventList, eventNames, grid1, gridName






    folder1 = 'DLP'
    #folder1 = '10x'
    #method1 = 'signals'
    #method1 = 'chisel'
    #folder2 = folder1 + '_' + method1

    #tree1 = loadnpz('./data/comparison/tree/tree_' + folder2 + '_ours.npz')[0]
    #tree1 = loadnpz('./data/comparison/tree/tree_' + folder2 + '.npz')[0]
    #tree1 = loadnpz('./data/comparison/tree/tree_' + folder1 + '_deep.npz')[0]

    #[['1' '26696033' '26782110' '' 'TSG' 'ARID1A']
    #['17' '43044295' '43125483' 'breast, ovarian' 'TSG' 'BRCA1']
    #['17' '39462039' '39535146' '' 'TSG' 'CDK12']
    #['17' '39700080' '39728662' '' 'oncogene' 'ERBB2']]


    #np.savez_compressed('./data/comparison/parsimony/internalVals_' + folder1 + '_' + method1 + '.npz', treeInternal_choice )

    predCNA = loadnpz('./data/comparison/CNA/' + folder1 + '_deep.npz').astype(float)

    print (predCNA.shape)
    method1 = 'deep'
    #method1 = 'signals'
    #method1 = 'chisel'
    #method1 = 'naive'
    tree1 = loadnpz('./data/comparison/parsimony/treeWithLength_' + folder1 + '_' + method1 + '.npz')[0]
    pairList = loadnpz('./data/comparison/parsimony/pairListLength_' + folder1 + '_' + method1 + '.npz')
    #pairList = loadnpz('./data/comparison/parsimony/cladePairs_' + folder1 + '_' + method1 + '.npz')
    cladeSizes = loadnpz('./data/comparison/parsimony/cladeSizes_' + folder1 + '_' + method1 + '.npz')
    treeInternal = loadnpz('./data/comparison/parsimony/internalVals_' + folder1 + '_' + method1 + '.npz')
    chr = loadnpz('./data/comparison/CNA/' + folder1 + '_chr.npz')

    print ('method1')

    


    qBool = loadnpz('./data/comparison/chrArm/' + folder1 + '_Qbool.npz')



    showLeaf = False

    


    miniTree = True

    if miniTree:
        if folder1 == 'DLP':
            Nmin = 30
            Nmin2 = 10
        if folder1 == '10x':
            Nmin = 100
            Nmin2 = 50
        tree1 = simplifyClusterTree(tree1, pairList, cladeSizes, Nmin, Nmin2)

    else:

        tree1 = simplifyClonesTree(tree1)

    

    #print (tree1)

    #editedChr = getEdgeCNA('C433', 'C477', 'C555', pairList, treeInternal, chr)
    #editedChr = getEdgeCNA('C433', 'C555', 'C377', pairList, treeInternal, chr)
    #editedChr = getEdgeCNA('C174', 'C377', 'C555', pairList, treeInternal, chr)
    #editedChr = getEdgeCNA('C174', 'C377', 'C555', pairList, treeInternal, chr)
    #editedChr = getEdgeCNA('C45', 'C135', 'C555', pairList, treeInternal, chr)


    if False:
        #eventList, grid1 = getEdgeCNA('C555', 'C135', 'C620', pairList, treeInternal, chr, qBool)
        eventList, eventNames, grid1, gridName = getEdgeCNA('C555', 'C498', 'C377', pairList, treeInternal, chr, qBool)
        print (len(eventNames))
        if True:#len(eventNames) < 10:
            print (' '.join(eventNames))
        print (gridName)
        quit()



    
    

    tree1 = tree1 + ';'

    #print (len(tree1.split(')')))
    #print (len(tree1.split('(')))
    #quit()


    tree1 = tree1.replace('root', '')
    #print (tree1)
    #quit()

    #if folder1 == '10x':
    #    give10xClusters()
    
    
        



    from ete3 import Tree, faces, AttrFace, TreeStyle, NodeStyle, TextFace

    #tree1 = tree1[:-len('49.959152);')] + '200);'


    #print (tree1)

    t = Tree(tree1)
    ts = TreeStyle()
    ts.show_leaf_name = False
    ts.rotation = 90

    

    totalSize = 0

    for l in t.iter_leaves():

        if miniTree:
            cladeString = str(l).split('-')[-1]
            cladeInt = int(str(l).split('C')[1])
            cladeSize = cladeSizes[cladeInt]
            totalSize += cladeSize

            sizeRatio = (float(cladeSize) / float(np.sum(cladeSizes)))

            nstyle = NodeStyle()
            nstyle["fgcolor"] = "red"
            #nstyle["size"] = 100000 * sizeRatio
            nstyle["size"] = 2000 * (sizeRatio ** 0.5)

            
            if showLeaf:
                l.add_face(TextFace(str(cladeSize), fsize=50), column=0)
            else:
                l.add_face(TextFace(cladeString, fsize=50), column=0)

            l.set_style(nstyle)
        #print (l)

    print (totalSize)

    #quit()

    #ts.branch_vertical_margin = 50
    #ts.hz_line_width = 1
    #ts.scale = 30 #Proportional to height!

    if miniTree:
        if folder1 == 'DLP':
            ts.scale = 2#5 #Proportional to height!
        if folder1 == '10x':
            #ts.scale = 240 #Proportional to height!
            ts.scale = 10#20
    else:
        if folder1 == 'DLP':
            ts.scale = 5 #Proportional to height!
        if folder1 == '10x':
            #ts.scale = 240 #Proportional to height!
            ts.scale = 10

    #ts.show_scale = True
    #ts.show_branch_length = True
    ts.min_leaf_separation = 0
    ts.branch_vertical_margin = 0
    t.show(tree_style=ts)
    quit()


    

plotVertTree()
quit()




def OLD_plotTree():

    folder1 = 'DLP'
    #folder1 = '10x'
    method1 = 'signals'
    #method1 = 'chisel'
    folder2 = folder1 + '_' + method1

    #tree1 = loadnpz('./data/comparison/tree/tree_' + folder2 + '_ours.npz')[0]
    tree1 = loadnpz('./data/comparison/tree/tree_' + folder2 + '.npz')[0]
    tree1 = tree1.replace('root', '')
    #print (tree1)
    #quit()


    from ete3 import Tree, faces, AttrFace, TreeStyle, NodeStyle

    def layout(node):
        if node.is_leaf():
            N = AttrFace("name", fsize=30)
            faces.add_face_to_node(N, node, 0, position="aligned")

    def get_example_tree():

        cellNames = loadnpz('./data/' + folder1 + '/initial/cellNames.npz')


    

        cellNames = loadnpz('./data/' + folder1 + '/initial/cellNames.npz')
        subset2 = loadnpz('./data/comparison/CNA/' + folder2 + '_subsetOurs.npz')
        cellNames = cellNames[subset2]
        samples = np.copy(cellNames)
        for a in range(samples.shape[0]):
            samples[a] = samples[a].split('-')[0]
        samples_unique, samples_inverse = np.unique(samples, return_inverse=True)

        #samples_inverse = np.concatenate((samples_inverse, np.zeros(1))) #the root



        #"LightSteelBlue"
        #"Moccasin"
        #"DarkSeaGreen"
        #"Khaki"



        #t = Tree("((((a1,a2),a3), ((b1,b2),(b3,b4))), ((c1,c2),c3));")
        t = Tree(tree1)
        #for n in t.traverse():
        #for n in t.traverse():
        for n in t.iter_leaves():
            string1 = str(n)
            string1 = string1.split('-')[-1]
            string1 = int(string1)

            if string1 != samples_inverse.shape[0]: #not the root. 

                sampleValue = samples_inverse[string1]

                n.dist = 1

                nstyle = NodeStyle()
                #nstyle["fgcolor"] = "red"
                
                if sampleValue == 0:
                    nstyle["bgcolor"] = "DarkSeaGreen"
                if sampleValue == 1:
                    nstyle["bgcolor"] = "Moccasin"
                if sampleValue == 2:
                    nstyle["bgcolor"] = "LightSteelBlue"
                if sampleValue == 3:
                    nstyle["bgcolor"] = "Khaki"
                if sampleValue == 4:
                    nstyle["bgcolor"] = "pink"

                nstyle["size"] = 15
                n.set_style(nstyle)

                #if len(string1) <= 100:
                #    if len(string1.split('-')) == 3:
                #        nodeName = string1.split('-')[-1]
                #        nodeName = int(nodeName)
                #        print (str(n))
                #        print ([nodeName])
                    
                    
                    

        #n1 = t.get_common_ancestor("a1", "a2", "a3")
        #n1.set_style(nst1)
        #n2 = t.get_common_ancestor("b1", "b2", "b3", "b4")
        #n2.set_style(nst2)
        #n3 = t.get_common_ancestor("c1", "c2", "c3")
        #n3.set_style(nst3)
        #n4 = t.get_common_ancestor("b3", "b4")
        #n4.set_style(nst4)

        #n1 = t.get_common_ancestor("560", "598", "474")
        #n1.set_style(nstyle)




        ts = TreeStyle()
        ts.layout_fn = layout
        ts.show_leaf_name = False
        

        ts.mode = "c"
        ts.root_opening_factor = 1

        ts.show_leaf_name = False

        return t, ts


    t, ts = get_example_tree()
    #t.render("node_background.png", w=400, tree_style=ts)
    t.show(tree_style=ts)

    

#plotTree()
#quit()





def compareEarlyTreeSNV():

    #folder1 = 'DLP'
    folder1 = '10x'
    method1 = 'signals'
    #method1 = 'chisel'
    folder2 = folder1 + '_' + method1

    clade_there = loadnpz('./data/comparison/tree/clades_' + folder2 + '.npz')
    #clade_there = loadnpz('./data/comparison/tree/clades_' + folder2 + '_random.npz')
    clade_ours = loadnpz('./data/comparison/tree/clades_' + folder2 + '_ours.npz')
    SNVMatrix = loadnpz('./data/comparison/SNV/' + folder1 + '_SNV.npz')

    predCNA = loadnpz('./data/comparison/CNA/' + folder2 + '_deep.npz')
    cellNames = loadnpz('./data/' + folder1 + '/initial/cellNames.npz')
    subset2 = loadnpz('./data/comparison/CNA/' + folder2 + '_subsetOurs.npz')

    cell_unique = loadnpz('./data/comparison/SNV/' + folder1 + '_SNV_cell.npz')



    cellNames = cellNames[subset2]
    SNVMatrix = SNVMatrix[np.isin(cell_unique, cellNames)]
    cell_unique = cell_unique[np.isin(cell_unique, cellNames)]



    predCNA = predCNA[np.isin(cellNames, cell_unique)]
    clade_there = clade_there[:, np.isin(cellNames, cell_unique)]
    clade_ours = clade_ours[:, np.isin(cellNames, cell_unique)]
    cellNames = cellNames[np.isin(cellNames, cell_unique)]
    


    #clade_there = np.concatenate(( clade_there , 1-clade_there ), axis=0)
    #clade_ours = np.concatenate(( clade_ours , 1-clade_ours ), axis=0)

    SNVMatrix = SNVMatrix[:, :, 0]

    #SNVMatrix = SNVMatrix[:, np.sum(SNVMatrix, axis=0) >= 10]
    #SNVMatrix = SNVMatrix[:, np.sum(SNVMatrix, axis=0) >= 50]

    #mult1 = np.matmul(clade_ours, clade_ours.T)]


    

    SNVcladeSizes_ours = findCladeSizes(clade_ours, SNVMatrix)
    SNVcladeSizes_there = findCladeSizes(clade_there, SNVMatrix)

    SNVcladeSizes_ours = clade_ours.shape[1] - SNVcladeSizes_ours
    SNVcladeSizes_there = clade_ours.shape[1] - SNVcladeSizes_there

    percentTrunk_ours = np.argwhere(SNVcladeSizes_ours == 0).shape[0] / SNVcladeSizes_ours.shape[0]
    percentTrunk_there = np.argwhere(SNVcladeSizes_there == 0).shape[0] / SNVcladeSizes_there.shape[0]

    print ('percent ours: ', percentTrunk_ours)
    print ('percent theres: ', percentTrunk_there)
    

    #quit()

    print (np.mean( SNVcladeSizes_ours ))
    print (np.mean( SNVcladeSizes_there))
    plt.hist(SNVcladeSizes_ours, bins=100, histtype='step')
    plt.hist(SNVcladeSizes_there, bins=100, histtype='step')
    #plt.hist(SNVcladeSizes_ours - SNVcladeSizes_there, bins=100, histtype='step')
    plt.show()
    
    #quit()


    chr = loadnpz('./data/comparison/CNA/' + folder2 + '_chr.npz')

    clade_check = clade_ours
    #clade_check = clade_there

    for a in range(clade_check.shape[0]):
        #print (predCNA.shape)
        #print (np.sum(clade_there[a]))
        print (clade_check.shape[1] - np.sum(clade_check[a]) )
        
        if np.sum(clade_check[a]) >= clade_check[a].shape[0] - 250:
            if np.sum(clade_check[a]) < clade_check[a].shape[0] - 1:
                
                #haplotypePlotter(predCNA[clade_there[a] == 1, 0::10][0::5], doCluster=True, withLinkage=[], saveFile='')#, chr=[chr])
                haplotypePlotter(predCNA[clade_check[a] == 0, :], doCluster=True, withLinkage=[], saveFile='')#, chr=[chr])
    quit()


#compareEarlyTreeSNV()
#quit()




def plotDistances():

    folder1 = 'DLP'
    method1 = 'signals'
    folder2 = folder1 + '_' + method1
    
    data = loadnpz('./data/comparison/tree/dist_' + folder2 + '.npz')
    data_ours = loadnpz('./data/comparison/tree/dist_' + folder2 + '_ours.npz')

    #plt.imshow(data_ours)
    #plt.show()
    #quit()

    print (np.mean(data_ours[:, -1]))
    print (np.mean(data[:, -1]))

    plt.hist(data_ours[:, -1], bins=100)
    plt.hist(data[:, -1], bins=100)
    plt.show()



#plotDistances()
#quit()



def findSNVMatrix():

    folder1 = '10x'

    if folder1 == 'DLP':
        data = np.loadtxt('./data/comparison/SNV/DLP_variant_data.filt.tsv', delimiter='\t', dtype=str)
    if folder1 == '10x':
        data = np.loadtxt('./data/comparison/SNV/10x_variant_data_autosomes_filtered.tsv', delimiter='\t', dtype=str)
    #print (data.shape)

    #position = data[:, 1]
    position = data[:, :2]
    cell = data[:, 2]
    totalReads = data[:, -1].astype(int)
    variantReads = data[:, -2].astype(int)

    

    
    position_inverse = uniqueValMaker(position)
    _, position_index = np.unique( position_inverse, return_index=True )
    position_unique = position[position_index]
    #position_unique = np.unique(position_inverse)
    cell_unique, cell_inverse = np.unique(cell, return_inverse=True)

    #print (cell_unique[:10])
    if folder1 == '10x':
        dict1 = {}
        cellNames = loadnpz('./data/' + folder1 + '/initial/cellNames.npz')
        for cell1 in cellNames:
            cell2 = cell1[2:]
            dict1[cell2] = cell1


    for a in range(cell_unique.shape[0]):
        cell_unique[a] = cell_unique[a].replace('"', '')
        if cell_unique[a][-2:] == '-1':
            cell_unique[a] = cell_unique[a][:-2]
        if folder1 == '10x':
            cell_unique[a] = dict1[cell_unique[a]]

    cell_unique, cell_unique_inverse = np.unique(cell_unique, return_inverse=True)
    cell_inverse = cell_unique_inverse[cell_inverse]



    

    #print (cell_unique[:10])
    #quit()

    matrix1 = np.zeros(( cell_unique.shape[0], position_unique.shape[0] , 2 ), dtype=int)
    matrix1[cell_inverse, position_inverse, 0] = variantReads
    matrix1[cell_inverse, position_inverse, 1] = totalReads

    for a in range(position_unique.shape[0]):
        position_unique[a, 0] = position_unique[a, 0].replace('"', '')
        position_unique[a, 0] = position_unique[a, 0].replace('chr', '')
        

    #print (position_unique.shape)
    #print (matrix1.shape)
    #quit()


    np.savez_compressed('./data/comparison/SNV/' + folder1 + '_SNV.npz', matrix1)
    np.savez_compressed('./data/comparison/SNV/' + folder1 + '_SNV_cell.npz', cell_unique)
    np.savez_compressed('./data/comparison/SNV/' + folder1 + '_SNV_position.npz', position_unique)



#findSNVMatrix()
#quit()




def getTruncalSNVCNA():


    

    folder1 = 'DLP'
    #folder1 = '10x' 


    #method1 = 'signals'
    #folder2 = folder1 + '_' + method1

    SNVMatrix = loadnpz('./data/comparison/SNV/' + folder1 + '_SNV.npz')

    print (SNVMatrix.shape)
    #print (np.sum(SNVMatrix))
    quit()


    #np.savez_compressed('./data/comparison/SNV/DLP_SNV_cell.npz', cell_unique)
    snvPos = loadnpz('./data/comparison/SNV/' + folder1 + '_SNV_position.npz')

    predCNA = loadnpz('./data/comparison/CNA/' + folder1 + '_deep.npz').astype(float)
    #naiveCNA = loadnpz('./data/comparison/CNA/' + folder2 + '_naive.npz').astype(float)
    #copyMatrix = loadnpz('./data/comparison/CNA/' + folder1 + '_there.npz').astype(float)
    cellNames = loadnpz('./data/' + folder1 + '/initial/cellNames.npz')
    


    posToBin = loadnpz('./data/comparison/CNA/' + folder1 + '_posToBin.npz')
    chrAll = loadnpz('./data/' + folder1 + '/initial/allChr_100k.npz')

    
    #useMethod = 'there'



    #if useMethod == 'there':
    #    predCNA = copyMatrix

        
    


    #predCNA = copyMatrix



    subset2 = loadnpz('./data/comparison/CNA/' + folder1 + '_subsetOurs.npz')

    cell_unique = loadnpz('./data/comparison/SNV/' + folder1 + '_SNV_cell.npz')

    cellNames = cellNames[subset2]

    #predCNA = predCNA[np.argsort(cellNames)]
    #cellNames = cellNames[np.argsort(cellNames)]



    SNVMatrix = SNVMatrix[np.isin(cell_unique, cellNames)]
    cell_unique = cell_unique[np.isin(cell_unique, cellNames)]


    predCNA = predCNA[np.isin(cellNames, cell_unique)]
    #copyMatrix = copyMatrix[np.isin(cellNames, cell_unique)]
    cellNames = cellNames[np.isin(cellNames, cell_unique)]

    

    
    #for a in range(cell_unique.shape[0]):
    #    assert cellNames[a] == np.sort(cellNames)[a]
    
    #for a in range(cell_unique.shape[0]):
    #    assert cell_unique[a] == np.sort(cell_unique)[a]


    for a in range(cell_unique.shape[0]):
        assert cellNames[a] == cell_unique[a]

    #quit()

    #print (cellNames[:10])
    #print (cell_unique[:10])
    




    samples = np.copy(cellNames)
    for a in range(samples.shape[0]):
        samples[a] = samples[a].split('-')[0]
    samples_unique, samples_inverse = np.unique(samples, return_inverse=True)

    


    SNVsums = np.zeros(( samples_unique.shape[0], SNVMatrix.shape[1]), dtype=int)
    for a in range(samples_unique.shape[0]):
        #SNVsum1 = np.sum(SNVMatrix[samples_inverse == a], axis=0)
        SNVsums[a] = np.sum(SNVMatrix[samples_inverse == a, :, 0], axis=0)

    #plt.plot(SNVsums.T)
    #plt.show()

    Ncut = 5
    SNVsums[SNVsums < Ncut] = 0
    SNVsums[SNVsums >= Ncut] = 1
    

    

    SNVsum_all = np.sum(SNVsums, axis=0)

    if folder1 == 'DLP':
        argGood = np.argwhere(SNVsum_all >= 3)[:, 0]
    if folder1 == '10x':
        argGood = np.argwhere(SNVsum_all >= 4)[:, 0]

    

    

    print (argGood.shape)
    #quit()

    

    #SNVsum1_both = SNVsum1_both[argGood]
    #SNVsum2_both = SNVsum2_both[argGood]
    #SNVsum3_both = SNVsum3_both[argGood]
    snvPos = snvPos[argGood]
    SNVMatrix = SNVMatrix[:, argGood]

    #print (snvPos[:, 0])
    #quit()

    snvBins = np.zeros(SNVMatrix.shape[1], dtype=int) - 1

    for a in range(22):

        posToBin_mini = posToBin[chrAll == a]

        chr1 = str(a + 1)
        argPos = np.argwhere(snvPos[:, 0] == chr1)[:, 0]
        snvPos_mini = snvPos[ argPos , 1].astype(int)
        snvPos_mini = snvPos_mini // 100000

        #print (snvPos_mini.shape)

        argPos = argPos[snvPos_mini < posToBin_mini.shape[0]]
        snvPos_mini = snvPos_mini[snvPos_mini < posToBin_mini.shape[0]]

        snvPos_mini_bin = posToBin_mini[snvPos_mini]

        argPos = argPos[snvPos_mini_bin != -1]
        snvPos_mini_bin = snvPos_mini_bin[snvPos_mini_bin != -1]

        snvBins[argPos] = snvPos_mini_bin 

        #print (snvPos_mini_bin.shape)

    #quit()

    SNVMatrix = SNVMatrix[:, snvBins!=-1]
    #SNVsum1_both = SNVsum1_both[snvBins!=-1]
    #SNVsum2_both = SNVsum2_both[snvBins!=-1]
    #SNVsum3_both = SNVsum3_both[snvBins!=-1]
    snvBins = snvBins[snvBins!=-1]

    print (snvBins.shape)
    

    #print (SNVsum1_both.shape)
    #quit()

    #SNVsumAll_both = [SNVsum1_both, SNVsum2_both, SNVsum3_both]


    countList = np.zeros((snvBins.shape[0]*8, 2) , dtype=int )
    copyList = np.zeros(countList.shape, dtype=int)
    posList = np.zeros((snvBins.shape[0] * 8, 2), dtype=int )
    count1 = 0

    for a in range(snvBins.shape[0]):

        
        inverseCopy = uniqueValMaker(predCNA[:, snvBins[a]])
        _, index1, counts1 = np.unique(inverseCopy, return_index=True, return_counts=True)

        index1 = index1[counts1 >= 100]

        for b in range(index1.shape[0]):

            #print (inverseCopy.shape)

            index2 = index1[b]
            args1 = np.argwhere(inverseCopy == inverseCopy[index2])[:, 0]


            countList[count1] = np.sum(SNVMatrix[args1, a], axis=0)
            copyList[count1] = predCNA[index2, snvBins[a]]
            posList[count1, 0] = snvBins[a]
            posList[count1, 1] = a
            count1 += 1

    
    countList = countList[:count1]
    copyList = copyList[:count1]
    posList = posList[:count1]

    print (posList.shape)

    
    #if useMethod == 'there':
    #    np.savez_compressed('./data/comparison/SNV/' + folder1 + '_SNVthere_countlist.npz', countList)
    #    np.savez_compressed('./data/comparison/SNV/' + folder1 + '_SNVthere_copylist.npz', copyList)
    #    np.savez_compressed('./data/comparison/SNV/' + folder1 + '_SNVthere_poslist.npz', posList)
    #else:
    print ('hi')
    np.savez_compressed('./data/comparison/SNV/' + folder1 + '_SNV_countlist.npz', countList)
    np.savez_compressed('./data/comparison/SNV/' + folder1 + '_SNV_copylist.npz', copyList)
    np.savez_compressed('./data/comparison/SNV/' + folder1 + '_SNV_poslist.npz', posList)


#getTruncalSNVCNA()
#quit()

#string1 = 'CCTGCCTCCTCTCCTCCTTTTTTCAAGCAGAAGACGGCATACGCGATTGTGCCGTCTCGTGGGCTCGGAGATGTGTATAAGAGACAGTCATTAGGCCCAGCTTATGCCTCACGGTGGCCTTTCCAGGCCTAGCTCCTGCCCCCCCACAGC'
#string2 = 'AACTCCATTCAGGCTCCTTTGAGCCTTCTCCTTGATGAAGCCTCATCCTTGGCCTGCTGAGCTCAGTGCTAGCAAGGAATGCTGCTA'
#string3 = 'GTCTCTCTCTCTCCTCTTGCAATAGTTTTTTTTTTTTTTTAAGAGACAGGGCCTTGCTCTGTCAACCAGGTTGGAGTGC'
#print (len(string1))
#print (len(string2))
#print (len(string3))
#quit()




def getMappedTruncalSNV():


    

    folder1 = 'DLP'
    #folder1 = '10x' 
    #method1 = 'signals'
    #method1 = 'chisel'
    #folder2 = folder1 + '_' + method1


    #folder2 = folder2 + '_mod'

    SNVMatrix = loadnpz('./data/comparison/SNV/' + folder1 + '_SNV.npz')

    
    snvPos = loadnpz('./data/comparison/SNV/' + folder1 + '_SNV_position.npz')

    #inverse1 = uniqueValMaker(snvPos)
    #_, counts = np.unique(inverse1, return_counts=True)
    #print (np.unique(counts, return_counts=True))
    #quit()

    predCNA = loadnpz('./data/comparison/CNA/' + folder1 + '_deep.npz').astype(float)
    #naiveCNA = loadnpz('./data/comparison/CNA/' + folder2 + '_naive.npz').astype(float)
    signals = loadnpz('./data/comparison/CNA/' + folder1 + '_signals.npz').astype(float)
    chisel = loadnpz('./data/comparison/CNA/' + folder1 + '_chisel.npz').astype(float)
    naive = loadnpz('./data/comparison/CNA/' + folder1 + '_naive.npz').astype(float)
    cellNames = loadnpz('./data/' + folder1 + '/initial/cellNames.npz')

    #quit()


    posToBin = loadnpz('./data/comparison/CNA/' + folder1 + '_posToBin.npz')
    chrAll = loadnpz('./data/' + folder1 + '/initial/allChr_100k.npz')

    chr = loadnpz('./data/comparison/CNA/' + folder1 + '_chr.npz')



    
    subset2 = loadnpz('./data/comparison/CNA/' + folder1 + '_subsetOurs.npz')

    cell_unique = loadnpz('./data/comparison/SNV/' + folder1 + '_SNV_cell.npz')

    cellNames = cellNames[subset2]
    SNVMatrix = SNVMatrix[np.isin(cell_unique, cellNames)]
    cell_unique = cell_unique[np.isin(cell_unique, cellNames)]


    predCNA = predCNA[np.isin(cellNames, cell_unique)]
    signals = signals[np.isin(cellNames, cell_unique)]
    chisel = chisel[np.isin(cellNames, cell_unique)]
    naive = naive[np.isin(cellNames, cell_unique)]
    cellNames = cellNames[np.isin(cellNames, cell_unique)]

    #print (cellNames.shape)
    #print (cell_unique.shape)

    #print (predCNA.shape)
    #quit()


    for a in range(cell_unique.shape[0]):
        assert cellNames[a] == cell_unique[a]

    #print (cellNames[:10])
    #print (cell_unique[:10])
    




    samples = np.copy(cellNames)
    for a in range(samples.shape[0]):
        samples[a] = samples[a].split('-')[0]
    samples_unique, samples_inverse = np.unique(samples, return_inverse=True)



    SNVsums = np.zeros(( samples_unique.shape[0], SNVMatrix.shape[1]), dtype=int)
    for a in range(samples_unique.shape[0]):
        #SNVsum1 = np.sum(SNVMatrix[samples_inverse == a], axis=0)
        SNVsums[a] = np.sum(SNVMatrix[samples_inverse == a, :, 0], axis=0)

    #plt.plot(SNVsums.T)
    #plt.show()

    Ncut = 5
    #Ncut = 10
    SNVsums[SNVsums < Ncut] = 0
    SNVsums[SNVsums >= Ncut] = 1
    

    

    SNVsum_all = np.sum(SNVsums, axis=0)
    if folder1 == 'DLP':
        argGood = np.argwhere(SNVsum_all >= 3)[:, 0]
    if folder1 == '10x':
        argGood = np.argwhere(SNVsum_all >= 4)[:, 0]

    
    snvPos = snvPos[argGood]
    SNVMatrix = SNVMatrix[:, argGood]
    snvBins = np.zeros(SNVMatrix.shape[1], dtype=int) - 1

    for a in range(22):

        posToBin_mini = posToBin[chrAll == a]

        chr1 = str(a + 1)
        argPos = np.argwhere(snvPos[:, 0] == chr1)[:, 0]
        snvPos_mini = snvPos[ argPos , 1].astype(int)
        snvPos_mini = snvPos_mini // 100000

        #print (snvPos_mini.shape)

        argPos = argPos[snvPos_mini < posToBin_mini.shape[0]]
        snvPos_mini = snvPos_mini[snvPos_mini < posToBin_mini.shape[0]]

        snvPos_mini_bin = posToBin_mini[snvPos_mini]

        argPos = argPos[snvPos_mini_bin != -1]
        snvPos_mini_bin = snvPos_mini_bin[snvPos_mini_bin != -1]

        snvBins[argPos] = snvPos_mini_bin 

        #print (snvPos_mini_bin.shape)

    

    SNVMatrix = SNVMatrix[:, snvBins!=-1]
    snvPos = snvPos[snvBins!=-1]
    snvBins = snvBins[snvBins!=-1]


    snvBins_unique, snvBins_inverse = np.unique(snvBins, return_inverse=True)

    #print (predCNA.shape)
    #print (chr.shape)
    #quit()

    chisel = chisel[:, snvBins_unique]
    signals = signals[:, snvBins_unique]
    naive = naive[:, snvBins_unique]
    predCNA = predCNA[:, snvBins_unique]
    chr = chr[snvBins_unique]


    np.savez_compressed('./data/comparison/SNV/' + folder1 + '_SNVCNA_deep.npz', predCNA)
    np.savez_compressed('./data/comparison/SNV/' + folder1 + '_SNVCNA_signals.npz', signals)
    np.savez_compressed('./data/comparison/SNV/' + folder1 + '_SNVCNA_chisel.npz', chisel)
    np.savez_compressed('./data/comparison/SNV/' + folder1 + '_SNVCNA_naive.npz', naive)
    np.savez_compressed('./data/comparison/SNV/' + folder1 + '_SNVCNA_SNVmatrix.npz', SNVMatrix)
    np.savez_compressed('./data/comparison/SNV/' + folder1 + '_SNVCNA_mapper.npz', snvBins_inverse)
    np.savez_compressed('./data/comparison/SNV/' + folder1 + '_SNVCNA_chr.npz', chr)


#getMappedTruncalSNV()
#quit()



def analyzeTruncalSNV():


    plotMini = True

    folder1 = '10x'
    #folder1 = 'DLP'
    #method1 = 'signals'
    #folder2 = folder1 + '_' + method1
    countList = loadnpz('./data/comparison/SNV/' + folder1 + '_SNV_countlist.npz')
    copyList = loadnpz('./data/comparison/SNV/' + folder1 + '_SNV_copylist.npz')
    posList = loadnpz('./data/comparison/SNV/' + folder1 + '_SNV_poslist.npz')
    chr = loadnpz('./data/comparison/CNA/' + folder1 + '_chr.npz')



    #countList = countList[np.isin(chr[posList]+1 ,np.array([8]) )==True]
    #copyList = copyList[np.isin(chr[posList]+1 ,np.array([8]) )==True]


    #plt.hist(countList[:, 1], bins=100, range=(0, 200))
    #plt.show()

    copyList = copyList[countList[:, 1] >= 20]
    countList = countList[countList[:, 1] >= 20]

    copyList = np.sort(copyList, axis=1)
    inverse_copy = uniqueValMaker(copyList)
    _, index1, count1 = np.unique(inverse_copy, return_index=True, return_counts=True)
    #index1 = index1[count1 > 5]

    if folder1 == 'DLP':
        index1 = index1[count1 > 150]
    else:
        index1 = index1[count1 > 20]

    #print (copyList[index1], count1)
    #quit()

    #sum1 = 4
    for index0 in index1:
        print (copyList[index0])
        #copyString = str(copyList[index0])
        #print (copyString)
        copyString = str(copyList[index0][0]) + '_' + str(copyList[index0][1])

        balance1 = float(copyList[index0][0]) / float( copyList[index0][0] + copyList[index0][1]  )
        
        args1 = np.argwhere(inverse_copy == inverse_copy[index0])[:, 0]

        print ('number of SNVs', args1.shape[0])

        countList_mini = countList[args1]

        ratio1 = countList_mini[:, 0] / countList_mini[:, 1]

        plt.hist(ratio1, bins=100, range=(0, 1))
        plt.axvline(x=balance1, c='black', lw=1)
        plt.axvline(x=1-balance1, c='black', lw=1)
        plt.xlabel("VAF")
        if plotMini:
            plt.ylabel('#SNVs')
        else:
            plt.ylabel('number of SNVs')
        #plt.hist(ratio1[countList_mini[:, 1] >= 50], bins=100, range=(0, 1))
        #if (copyList[index0][0] == 1) and (copyList[index0][0] == 2):
        miniString = ''
        if plotMini:
            plt.gcf().set_size_inches(4, 1.7)
            miniString = '_mini'
        plt.tight_layout()
        
        plt.savefig('./images/VAF/' + folder1 + '/' + copyString + miniString + '.pdf')
        plt.show()



#analyzeTruncalSNV()
#quit()



def analyzeDiffTruncalSNV():


    def countProbGivenCopyNumber(copyNumbers, SNV_now, ifLOH):

        BAF1 = (copyNumbers[:, 0] + 1e-5) / (np.sum(copyNumbers, axis=1) + 2e-5)
        #BAF1[BAF1<0.1] = 0.1
        #BAF1[BAF1>0.9] = 0.9
        #BAF1[BAF1<0.25] = 0.25
        #BAF1[BAF1>0.75] = 0.75

        if ifLOH == 'withLOH':
            BAF1[BAF1<0.001] = 0.001
            BAF1[BAF1>0.999] = 0.999

        BAF2 = 1.0 - BAF1

        prob1 = np.sum(np.log(BAF1) * SNV_now[:, 0]) + np.sum(np.log(BAF2) * SNV_now[:, 1])
        prob2 = np.sum(np.log(BAF2) * SNV_now[:, 0]) + np.sum(np.log(BAF1) * SNV_now[:, 1])
        prob = max(prob1, prob2)

        return prob
    

    def compareProbGivenCopyNumber(copyNumbers1, copyNumbers2, SNV_now, ifLOH):

        prob1 = countProbGivenCopyNumber(copyNumbers1, SNV_now, ifLOH)
        prob2 = countProbGivenCopyNumber(copyNumbers2, SNV_now, ifLOH)
        
        diff1 = prob1 - prob2 

        return diff1



    

    def countErrorGivenCopyNumber(copyNumbers, SNV_now):

        min1 = np.min(copyNumbers, axis=1)
        copyNumbers2 = np.copy(copyNumbers)
        #copyNumbers2[min1 == 0] = copyNumbers2[min1 == 0].astype(float) + 0.1
        #copyNumbers2 = copyNumbers2.astype(float) + 100.0

        BAF1 = (copyNumbers2[:, 0] + 1e-5) / (np.sum(copyNumbers2, axis=1) + 2e-5)
        BAF1[BAF1<0.001] = 0.001
        BAF1[BAF1>0.999] = 0.999
        #BAF1[BAF1<0.25] = 0.25
        #BAF1[BAF1>0.75] = 0.75

        BAF2 = 1.0 - BAF1

        prob1 = np.sum(np.log(BAF1) * SNV_now[:, 0]) + np.sum(np.log(BAF2) * SNV_now[:, 1])
        prob2 = np.sum(np.log(BAF2) * SNV_now[:, 0]) + np.sum(np.log(BAF1) * SNV_now[:, 1])
        #prob = max(prob1, prob2)

        #prob = logsumexp(np.array([prob1, prob2])) + np.log(0.5)
        prob = max(prob1, prob2)


        return prob
    
    
    def includeLOHsingle(predCNA):

        good1 = np.ones(predCNA.shape[0])
        predCNA_sort = np.sort(predCNA, axis=1)

        good1[np.logical_and( predCNA_sort[:, 0] == 0, predCNA_sort[:, 1] >= 2 )] = 0
        #good1[np.logical_and( predCNA_sort[:, 0] == 0, predCNA_sort[:, 1] == 1 )] = 0
        return good1



    from scipy.special import logsumexp

    folder1 = '10x'
    #folder1 = 'DLP'
    #method1 = 'signals'
    #method1 = 'chisel'
    #folder2 = folder1 + '_' + method1

    #folder2 = folder2 + '_mod'

    #ifLOH = 'noLOH'
    ifLOH = 'withLOH'
    

    predCNA = loadnpz('./data/comparison/SNV/' + folder1 + '_SNVCNA_deep.npz').astype(float)
    signals = loadnpz('./data/comparison/SNV/' + folder1 + '_SNVCNA_signals.npz').astype(float)
    chisel = loadnpz('./data/comparison/SNV/' + folder1 + '_SNVCNA_chisel.npz').astype(float)
    naive = loadnpz('./data/comparison/SNV/' + folder1 + '_SNVCNA_naive.npz').astype(float)
    chr = loadnpz('./data/comparison/SNV/' + folder1 + '_SNVCNA_chr.npz')
    SNVMatrix = loadnpz('./data/comparison/SNV/' + folder1 + '_SNVCNA_SNVmatrix.npz')
    snvBins_inverse = loadnpz('./data/comparison/SNV/' + folder1 + '_SNVCNA_mapper.npz')

    #print (np.unique(chr, return_counts=True))
    #quit()

    #print (np.mean(np.abs( predCNA - copyMatrix )))
    #quit()

    
    SNVMatrix[:, :, 1] = SNVMatrix[:, :, 1] - SNVMatrix[:, :, 0]

    propList_ours = np.zeros(SNVMatrix.shape[1])
    propList_signals = np.zeros(SNVMatrix.shape[1])
    propList_chisel = np.zeros(SNVMatrix.shape[1])
    propList_naive = np.zeros(SNVMatrix.shape[1])

    #probListChr_ours = np.zeros(22)
    #probListChr_there = np.zeros(22)

    #probDiff = []

    #for b in range(10):
    #    print (b)

    #    probDiff.append(0)

    for a in range(SNVMatrix.shape[1]):
        
        #subsetBoot = np.random.choice(predCNA.shape[0], size=predCNA.shape[0], replace=True)
        #subsetBoot = np.arange(predCNA.shape[0])
        

        #SNV_now = SNVMatrix[subsetBoot, a]

        
        #isZero = includeLOHsingle(predCNA[:, snvBins_inverse[a]]) * includeLOHsingle(copyMatrix[:, snvBins_inverse[a]])


        #argNonzero = np.argwhere(isZero != 0)[:, 0]

        #argNonzero = np.argwhere(isZero!=0)[:, 0]
        #argNonzero = np.arange(predCNA.shape[0]) #Trying include zero


        #prob_ours = countProbGivenCopyNumber(predCNA[argNonzero, snvBins_inverse[a]], SNVMatrix[argNonzero, a])
        #prob_there = countProbGivenCopyNumber(copyMatrix[argNonzero, snvBins_inverse[a]], SNVMatrix[argNonzero, a])
        if ifLOH == 'withLOH':
            isZero_signals = np.min(predCNA[:, snvBins_inverse[a]], axis=1) * np.min(signals[:, snvBins_inverse[a]], axis=1)
            isZero_chisel = np.min(predCNA[:, snvBins_inverse[a]], axis=1) * np.min(chisel[:, snvBins_inverse[a]], axis=1)
            isZero_naive = np.min(predCNA[:, snvBins_inverse[a]], axis=1) * np.min(naive[:, snvBins_inverse[a]], axis=1)

            argNonzero_signals = np.argwhere(isZero_signals == 0)[:, 0]
            argNonzero_chisel = np.argwhere(isZero_chisel == 0)[:, 0]
            argNonzero_naive = np.argwhere(isZero_naive == 0)[:, 0]


            prob_signals = compareProbGivenCopyNumber(predCNA[argNonzero_signals, snvBins_inverse[a]], signals[argNonzero_signals, snvBins_inverse[a]], SNVMatrix[argNonzero_signals, a], ifLOH)
            prob_chisel = compareProbGivenCopyNumber(predCNA[argNonzero_chisel, snvBins_inverse[a]], chisel[argNonzero_chisel, snvBins_inverse[a]], SNVMatrix[argNonzero_chisel, a], ifLOH)
            prob_naive = compareProbGivenCopyNumber(predCNA[argNonzero_naive, snvBins_inverse[a]], naive[argNonzero_naive, snvBins_inverse[a]], SNVMatrix[argNonzero_naive, a], ifLOH)

        else:
            isZero_signals = np.min(predCNA[:, snvBins_inverse[a]], axis=1) * np.min(signals[:, snvBins_inverse[a]], axis=1)
            isZero_chisel = np.min(predCNA[:, snvBins_inverse[a]], axis=1) * np.min(chisel[:, snvBins_inverse[a]], axis=1)
            isZero_naive = np.min(predCNA[:, snvBins_inverse[a]], axis=1) * np.min(naive[:, snvBins_inverse[a]], axis=1)

            argNonzero_signals = np.argwhere(isZero_signals != 0)[:, 0]
            argNonzero_chisel = np.argwhere(isZero_chisel != 0)[:, 0]
            argNonzero_naive = np.argwhere(isZero_naive != 0)[:, 0]


            prob_signals = compareProbGivenCopyNumber(predCNA[argNonzero_signals, snvBins_inverse[a]], signals[argNonzero_signals, snvBins_inverse[a]], SNVMatrix[argNonzero_signals, a], ifLOH)
            prob_chisel = compareProbGivenCopyNumber(predCNA[argNonzero_chisel, snvBins_inverse[a]], chisel[argNonzero_chisel, snvBins_inverse[a]], SNVMatrix[argNonzero_chisel, a], ifLOH)
            prob_naive = compareProbGivenCopyNumber(predCNA[argNonzero_naive, snvBins_inverse[a]], naive[argNonzero_naive, snvBins_inverse[a]], SNVMatrix[argNonzero_naive, a], ifLOH)


        #probDiff[a] = probDiff[a] + prob_ours - prob_there

        #print (abs(prob_ours - prob_there))

        #propList_ours[a] = prob_ours
        propList_signals[a] = prob_signals
        propList_chisel[a] = prob_chisel
        propList_naive[a] = prob_naive



    #probDiff_signals = propList_ours - propList_signals
    #probDiff_chisel = propList_ours - propList_chisel
    #probDiff_naive = propList_ours - propList_naive


    #plt.plot(probDiff)
    #plt.show()
    #quit()

    N = 100000
    #N = 10000
    probDiff2_signals = np.zeros(N)
    probDiff2_chisel = np.zeros(N)
    probDiff2_naive = np.zeros(N)

    np.random.seed(4)

    for b in range(N):

        subsetBoot = np.random.choice(propList_ours.shape[0], size=propList_ours.shape[0], replace=True)

        probDiff2_signals[b] = np.sum(propList_signals[subsetBoot])
        probDiff2_chisel[b] = np.sum(propList_chisel[subsetBoot])
        probDiff2_naive[b] = np.sum(propList_naive[subsetBoot])

    #print (np.argwhere(probDiff2_naive < 0).shape[0] / N)
    print (np.argwhere(probDiff2_signals < 0).shape[0] / N)
    print (np.argwhere(probDiff2_chisel < 0).shape[0] / N)

   #quit()

    #np.savez_compressed('./data/temp/1.npz', probDiff2)
    #quit()

    print (probDiff2_chisel.shape)
    print (probDiff2_signals.shape)

    #probDiff1 = loadnpz('./data/temp/1.npz')
    #quit()
    #probDiff2 = loadnpz('./temp/2.png')

    #quit()

    max1 = max(np.max(probDiff2_chisel), np.max(probDiff2_signals))
    min1 = min(np.min(probDiff2_chisel), np.min(probDiff2_signals))

    #plt.hist(probDiff1, bins=100, range=(min1, max1))
    #plt.hist(probDiff2, bins=100, range=(min1, max1))
    plt.hist(probDiff2_signals, bins=100, range=(min1, max1), color='orange')
    plt.hist(probDiff2_chisel, bins=100, range=(min1, max1), color='green')
    #####plt.hist(probDiff2_naive, bins=100, range=(min1, max1), c='green')
    plt.xlabel('truncal SNV support (LLR test)')
    plt.ylabel('bootstrap replicates')
    if (ifLOH == 'noLOH'):
        if folder1 == 'DLP':
            plt.ylim(0, 17000)
        if folder1 == '10x':
            plt.ylim(0, 13200)
    #plt.legend(['SIGNALS vs DeepCopy', 'CHISEL vs DeepCopy'])
    plt.legend(['DeepCopy vs SIGNALS', 'DeepCopy vs CHISEL'])
    if (ifLOH == 'noLOH'):
        plt.gcf().set_size_inches(3.35, 3)
    else:
        plt.gcf().set_size_inches(5, 3)
    plt.tight_layout()
    plt.savefig('./images/SNV/logProb_' + ifLOH + '_' + folder1 + '.pdf')
    plt.show()


    quit()


    #print (np.sum(probDiff))


    plt.scatter(propList_ours, propList_there)
    plt.scatter(propList_there, propList_there)
    plt.show()
    plt.hist(propList_ours - propList_there, bins=100)
    plt.show()

    #plt.plot(np.cumsum(propList_ours - propList_there))
    #plt.plot(propList_ours - propList_there)
    #plt.show()

    #plt.plot(probListChr_ours - probListChr_there)
    #plt.show()
    #quit()

    #plt.hist(probDiff, bins=100)
    #plt.show()



analyzeDiffTruncalSNV()
quit()










def findCloneSizes():



    def getCloneSizesMissing(copyMatrix):

        copyMatrix_clones = np.zeros(copyMatrix.shape, dtype=int)
        copyMatrix_clones[0] = copyMatrix[0]

        cloneSizes = np.ones(copyMatrix.shape[0], dtype=int)
        count1 = 1


        args1 = np.random.choice(copyMatrix.shape[1], size=20, replace=False)
        clones_tiny = np.zeros(copyMatrix[:, args1].shape, dtype=int)
        clones_tiny[0] = copyMatrix[0, args1]

        copyMatrix_tiny = copyMatrix[:, args1]


        args2 = np.random.choice(copyMatrix.shape[1], size=50, replace=False)



        time1_sum = 0
        time2_sum = 0
        time3_sum = 0

        for a in range(1, copyMatrix.shape[0]):

            #print (a, copyMatrix.shape[0])

            #print (time1_sum, time2_sum, time3_sum)

            foundSame = False    

            time1 = time.time()

            CNA1 = copyMatrix_tiny[a:a+1]
            CNA2 = clones_tiny[:count1]
            diff1 = np.abs(CNA1 - CNA2)
            mask1 = np.copy(clones_tiny[:count1])
            mask1[mask1 >= 0] = 1
            mask1[mask1 == -1] = 0
            diff1 = (diff1 * mask1)[:, CNA2[0, :, 0] != -1]
            diff1 = np.sum(diff1, axis=(1, 2))
            

            time1_sum += (time.time() - time1)

            argCheck = np.argwhere(diff1 == 0)[:, 0]

            for b in argCheck:

                if not foundSame:

                    time2 = time.time()

                    
                    
                    CNA1 = copyMatrix[a, args2]
                    CNA2 = copyMatrix_clones[b, args2]
                    diff1 = np.abs(CNA1 - CNA2)
                    diff1 = diff1[np.logical_and(CNA1[:, 0] != -1, CNA2[:, 0] != -1)]

                    if np.sum(diff1) == 0:


                        time3 = time.time()
                        CNA1 = copyMatrix[a]
                        CNA2 = copyMatrix_clones[b]
                        diff1 = np.abs(CNA1 - CNA2)
                        diff1 = diff1[np.logical_and(CNA1[:, 0] != -1, CNA2[:, 0] != -1)]

                        if np.sum(diff1) == 0:
                            foundSame = True
                            cloneSizes[b] += 1

                            args2 = np.argwhere(np.logical_and(  CNA1[:, 0] != -1, CNA2[:, 0] == -1  ))[:, 0]
                            copyMatrix_clones[b, args2] = np.copy(CNA1[args2])

                        time3_sum += (time.time() - time2)
                
                    time2_sum += (time.time() - time2)
            
            if not foundSame:
                clones_tiny[count1] = copyMatrix[a, args1]
                copyMatrix_clones[count1] = copyMatrix[a]
                count1 += 1

        
        cloneSizes = cloneSizes[:count1]

        return cloneSizes
    

    def getCloneSizesEither(copyMatrix):

        if not -1 in copyMatrix:
            copyMatrix = copyMatrix.reshape((copyMatrix.shape[0] , copyMatrix.shape[1]*2 ))
            inverse1 = uniqueValMaker(copyMatrix)
            _, cloneSizes = np.unique(inverse1, return_counts=True)
        else:
            cloneSizes = getCloneSizesMissing(copyMatrix)

        return cloneSizes



    #folder1 = 'DLP'
    folder1 = '10x'
    


    doZoom = '_zoom'

    N = 200
    perm1 = loadnpz('./data/comparison/CNA/zoomedPerm_' + folder1 + '.npz')[:N]

    

    
    
    predCNA = loadnpz('./data/comparison/CNA/' + folder1 + '_deep.npz').astype(float)
    naiveCNA = loadnpz('./data/comparison/CNA/' + folder1 + '_naive.npz').astype(float)
    signals = loadnpz('./data/comparison/CNA/' + folder1 + '_signals_missing.npz').astype(float)
    chisel = loadnpz('./data/comparison/CNA/' + folder1 + '_chisel_missing.npz').astype(float)
    chr = loadnpz('./data/comparison/CNA/' + folder1 + '_chr.npz').astype(float)

    if doZoom != '':
        if folder1 == 'DLP':
            args1 = np.argwhere(chr+1 == 1)[:, 0]
        if folder1 == '10x':
            args1 = np.argwhere(chr+1 == 6)[:, 0]
    



    cloneSizes_deep = getCloneSizesEither(predCNA[perm1][:, args1])
    cloneSizes_naive = getCloneSizesEither(naiveCNA[perm1][:, args1])
    cloneSizes_signals = getCloneSizesEither(signals[perm1][:, args1])
    cloneSizes_chisel = getCloneSizesEither(chisel[perm1][:, args1])

    print (cloneSizes_deep.shape)
    print (cloneSizes_signals.shape)
    print (cloneSizes_chisel.shape)

    if doZoom == '':
        np.savez_compressed('./data/comparison/clones/' + folder1 + '_deep_cloneSizes.npz', cloneSizes_deep)
        np.savez_compressed('./data/comparison/clones/' + folder1 + '_naive_cloneSizes.npz', cloneSizes_naive)
        np.savez_compressed('./data/comparison/clones/' + folder1 + '_signals_cloneSizes.npz', cloneSizes_signals)
        np.savez_compressed('./data/comparison/clones/' + folder1 + '_chisel_cloneSizes.npz', cloneSizes_chisel)




#findCloneSizes()
#quit()



def compareClonesSizes():

    folder1 = 'DLP'
    #folder1 = '10x'
    #method1 = 'signals'
    #method1 = 'chisel'
    #folder2 = folder1 + '_' + method1

    

    cloneSizes_deep = loadnpz('./data/comparison/clones/' + folder1 + '_deep_cloneSizes.npz')
    cloneSizes_naive = loadnpz('./data/comparison/clones/' + folder1 + '_naive_cloneSizes.npz')
    cloneSizes_signals = loadnpz('./data/comparison/clones/' + folder1 + '_signals_cloneSizes.npz')
    cloneSizes_chisel = loadnpz('./data/comparison/clones/' + folder1 + '_chisel_cloneSizes.npz')


    cloneSizes_deep = np.sort(cloneSizes_deep)[-1::-1]
    cloneSizes_naive = np.sort(cloneSizes_naive)[-1::-1]
    cloneSizes_signals = np.sort(cloneSizes_signals)[-1::-1]
    cloneSizes_chisel = np.sort(cloneSizes_chisel)[-1::-1]

    print (np.max(cloneSizes_deep))
    print (np.max(cloneSizes_chisel))
    quit()

    vals = np.array([  np.sum(cloneSizes_deep), np.sum(cloneSizes_naive), np.sum(cloneSizes_signals), np.sum(cloneSizes_chisel) ])

    assert np.sum(np.abs(vals - np.mean(vals) )) == 0

    

    if folder1 == 'DLP':
        N = 10
        loc = 'lower right'
    if folder1 == '10x':
        N = 100
        loc = 'upper right'
    
    arange1 = np.arange(N) + 1


    
    plt.plot(arange1, cloneSizes_deep[:N], color='blue')#, alpha=0.5)
    plt.plot(arange1, cloneSizes_naive[:N], color='tab:blue', alpha=0.5)
    plt.plot(arange1, cloneSizes_signals[:N], color='orange')#, alpha=0.5)
    plt.plot(arange1, cloneSizes_chisel[:N], color='green')#, alpha=0.5)

    if N <= 10:
         plt.xticks(arange1)
    
    plt.yscale('log')
    plt.xlabel("clone number")
    plt.ylabel('clone size (number of cells)')
    plt.legend(['DeepCopy', 'NaiveCopy', 'SIGNALS', "CHISEL"], loc=loc)
    #plt.gcf().set_size_inches(4, 3)
    plt.gcf().set_size_inches(3, 3)
    plt.tight_layout()
    #plt.title(folder2)
    plt.savefig('./images/clones/' + folder1 + '.pdf')
    plt.show()




#compareClonesSizes()
#quit()



def analyzeLargeCopyNumber():

    folder1 = '10x'
    #folder1 = 'DLP'
    method1 = 'chisel'
    #method1 = 'signals'
    folder2 = folder1 + '_' + method1

    predCNA = loadnpz('./data/comparison/CNA/' + folder2 + '_deep.npz').astype(float)
    copyMatrix = loadnpz('./data/comparison/CNA/' + folder2 + '_there.npz').astype(float)

    mean_ours = np.mean(predCNA, axis=(1, 2)) * 2
    mean_there = np.mean(copyMatrix, axis=(1, 2)) * 2

    mean_there[mean_there > 20] = 20


    plt.scatter(mean_ours, mean_there)
    plt.show()

    #argBad = np.argwhere(mean_there > 100)[:, 0]

    #plt.plot(  np.sum(copyMatrix[argBad[0]], axis=1) )
    #plt.show()
    #quit() 

    #plt.hist(mean_ours, histtype='step')
    #plt.hist(mean_there, histtype='step', bins=100)
    #plt.yscale('log')
    #plt.show()


#analyzeLargeCopyNumber()
#quit()




def checkMeasurementError():

    def getPredictedRDR(copyNumbers):

        return np.mean(copyNumbers, axis=2) / np.mean(copyNumbers, axis=(1, 2)).reshape((-1, 1))
    

    def getPredictedBAF(copyNumbers):

        return (np.min(copyNumbers, axis=2) + 1e-5) / (np.sum(copyNumbers, axis=2) + 2e-5)
    

    

    print ("A")


    folder1 = '10x'
    #folder1 = 'DLP'
    #method1 = 'chisel'
    #method1 = 'signals'
    #folder2 = folder1 + '_' + method1


    

    #BAFMatrix = loadnpz('./data/comparison/CNA/' + folder1 + '_BAF.npz')

    #subset2 = loadnpz('./data/comparison/CNA/' + folder1 + '_subsetOurs.npz')

    #chr = loadnpz('./data/comparison/CNA/' + folder1 + '_chr.npz')


    cloneSizes_deep = loadnpz('./data/comparison/clones/' + folder1 + '_deep_cloneSizes.npz')
    cloneSizes_naive = loadnpz('./data/comparison/clones/' + folder1 + '_naive_cloneSizes.npz')
    cloneSizes_signals = loadnpz('./data/comparison/clones/' + folder1 + '_signals_cloneSizes.npz')
    cloneSizes_chisel = loadnpz('./data/comparison/clones/' + folder1 + '_chisel_cloneSizes.npz')

    print (np.max(cloneSizes_chisel))
    quit()


    #cloneSizes = loadnpz('./data/comparison/clones/' + folder1 + '_cloneSizes.npz')
    #cloneSizes_ours = loadnpz('./data/comparison/clones/' + folder1 + '_ours_cloneSizes.npz')

    print ("B")
    #unique_profile = [np.max(cloneSizes_deep), np.max(cloneSizes_naive), np.max(cloneSizes_signals), np.max(cloneSizes_chisel) ] #1000 is just temporary until I actually load this.
    unique_profile = [cloneSizes_deep.shape[0], cloneSizes_naive.shape[0], cloneSizes_signals.shape[0], cloneSizes_chisel.shape[0] ]
    #unique_profile = np.array(unique_profile).astype(float)

    #print (unique_profile)
    #quit()

    if False:
        predCNA = loadnpz('./data/comparison/CNA/' + folder1 + '_deep.npz').astype(float)
        naiveCNA = loadnpz('./data/comparison/CNA/' + folder1 + '_naive.npz').astype(float)
        signals = loadnpz('./data/comparison/CNA/' + folder1 + '_signals.npz').astype(float)
        chisel = loadnpz('./data/comparison/CNA/' + folder1 + '_chisel.npz').astype(float)
        RDR_full = loadnpz('./data/comparison/CNA/' + folder1 + '_RDR.npz')
        HAP_full = loadnpz('./data/comparison/CNA/' + folder1 + '_HAP.npz').astype(float)


        RDR_ours = getPredictedRDR(predCNA)
        RDR_signals = getPredictedRDR(signals)
        RDR_chisel = getPredictedRDR(chisel)
        RDR_naive = getPredictedRDR(naiveCNA)

        error_RDR_ours = np.mean(np.abs(RDR_full - RDR_ours))#, axis=1)
        error_RDR_signals = np.mean(np.abs(RDR_full - RDR_signals))#, axis=1)
        error_RDR_chisel = np.mean(np.abs(RDR_full - RDR_chisel))
        error_RDR_naive = np.mean(np.abs(RDR_full - RDR_naive))#, axis=1)

        error_BAF_ours = np.mean(np.abs(   getPredictedBAF(predCNA) -  getPredictedBAF(HAP_full)   ))#, axis=1)
        error_BAF_signals = np.mean(np.abs(   getPredictedBAF(signals) -  getPredictedBAF(HAP_full)   ))#, axis=1)
        error_BAF_chisel = np.mean(np.abs(   getPredictedBAF(chisel) -  getPredictedBAF(HAP_full)   ))#, axis=1)
        error_BAF_naive = np.mean(np.abs(   getPredictedBAF(naiveCNA) -  getPredictedBAF(HAP_full)   ))#, axis=1)


        error_RDR = [np.mean(error_RDR_ours), np.mean(error_RDR_naive), np.mean(error_RDR_signals), np.mean(error_RDR_chisel)]
        error_BAF = [np.mean(error_BAF_ours), np.mean(error_BAF_naive), np.mean(error_BAF_signals), np.mean(error_BAF_chisel)]

        np.savez_compressed('./data/comparison/error/error_RDR_' + folder1 + '.npz', np.array(error_RDR))
        np.savez_compressed('./data/comparison/error/error_BAF_' + folder1 + '.npz', np.array(error_BAF))
        print ('done1')
        quit()

    else:
        error_RDR = loadnpz('./data/comparison/error/error_RDR_' + folder1 + '.npz')
        error_BAF = loadnpz('./data/comparison/error/error_BAF_' + folder1 + '.npz')



    #np.savez_compressed('./temp/error_RDR', np.array(error_RDR))

    #print (error_RDR)
    #quit()


    x = ['DeepCopy', 'NaiveCopy', 'SIGNALS', 'CHISEL']

    


    plotData = {}
    plotData['index'] = x
    plotData['error_RDR'] = error_RDR
    plotData['error_BAF'] = error_BAF
    plotData['unique_profile'] = unique_profile
    plotData['label1'] = x

    import pandas as pd

    palette = ['blue', 'lightblue', 'orange', 'green']

    df = pd.DataFrame(data=plotData)


    ax = sns.barplot(df, x='index', y='unique_profile', label='label1', palette=palette)
    #ax = sns.barplot(x=x, y=unique_profile, hue=['blue', 'green', 'orange', 'red']) #label='label1',

    
    #ax.set_palette(palette)


    #ax = sns.lineplot(data=df, x="index", y="error_RDR", color="red", lw=3)

    #plt.show()
    #quit()
    plt.xticks(rotation = 90)
    ax.set_ylabel("Number of Unique Profiles")
    ax.set_xlabel("")
    ax2 = ax.twinx()
    #sns.lineplot(ax=ax2, data=df, x="index", y="error_RDR", color="red", lw=3)
    sns.lineplot(x=x, y=error_RDR, ax=ax2, lw=3, label='R Error', color='black')
    sns.lineplot(x=x, y=error_BAF, ax=ax2, lw=3, label='B Error', color='black', linestyle='dotted')

    
    #ax2.set(ylim=[0.035, 0.07]) #0.035, 0.07
    ax2.set_ylabel("Error")
    ax2.set_xlabel("")
    

    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(labels=labels, handles=handles, loc='upper center')

    plt.gcf().set_size_inches(4, 3)

    
    saveFile = './images/plots/error_' + folder1 + '.pdf'

    plt.tight_layout()
    plt.savefig(saveFile)
    plt.show()


    quit()

    
    



    

    
    x = ['Ours', 'CHISEL', 'Naive']



    quit()

    plotData = {}
    plotData['index'] = x
    plotData['error_RDR'] = error_RDR
    plotData['error_BAF'] = error_BAF
    plotData['unique_profile'] = unique_profile

    import pandas as pd


    df = pd.DataFrame(data=plotData)


    #ax = sns.barplot(df, x='index', y='unique_profile', label='label1')
    ax = sns.barplot(x=x, y=unique_profile, label='label1')

    #ax = sns.lineplot(data=df, x="index", y="error_RDR", color="red", lw=3)

    #plt.show()
    #quit()
    plt.xticks(rotation = 90)
    ax.set_ylabel("Number of Unique Profiles")
    ax.set_xlabel("")
    ax2 = ax.twinx()
    #sns.lineplot(ax=ax2, data=df, x="index", y="error_RDR", color="red", lw=3)
    sns.lineplot(x=x, y=error_RDR, ax=ax2, color="red", lw=3, label='R Error')
    sns.lineplot(x=x, y=error_BAF, ax=ax2, color="blue", lw=3, label='B Error')

    
    #ax2.set(ylim=[0.035, 0.07]) #0.035, 0.07
    ax2.set_ylabel("Error")
    ax2.set_xlabel("")
    

    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(labels=labels, handles=handles)

    plt.gcf().set_size_inches(6, 5)

    
    #saveFile = "./images/realError_S" + patientNum0 + ".pdf"

    plt.tight_layout()
    #plt.savefig(saveFile)
    plt.show()




#checkMeasurementError()
#quit()


def plotReadCounts():

    folder1 = 'DLP'
    #folder1 = '10x'

    subset2 = loadnpz('./data/comparison/CNA/' + folder1 + '_subsetOurs.npz')

    totalRead_file = './data/' + folder1 + '/initial/totalReads.npz'

    totalRead = loadnpz(totalRead_file)

    if folder1 == 'DLP':
        readLength = 150.0
    if folder1 == '10x':
        readLength = (87.0 + 79.0) / 2.0

    totalLength = 3.4 * 1e9

    coverage = (totalRead * readLength) /  totalLength

    max1 = np.max(coverage)

    
    plt.hist(coverage, bins=100, range=(0, max1))
    plt.hist(coverage[subset2], bins=100, range=(0, max1))
    plt.legend(['all cells', 'included cells'])
    plt.xlabel('coverage')
    plt.ylabel('number of cells')
    plt.tight_layout()
    plt.savefig('./images/plots/coverage_' + folder1 + '.pdf')
    plt.show()



#plotReadCounts()
#quit()









