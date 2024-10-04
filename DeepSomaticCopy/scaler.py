#CNA.py

import numpy as np


#import matplotlib.pyplot as plt
import time
import scipy
from scipy import stats

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.optim import Optimizer


#import seaborn as sns

if __name__ == "__main__":
    from shared import *
else:
    from .shared import *

from tqdm import tqdm








def estimateHapNoise(HAP):

    #This is varience! Not standard deviation! 

    lastIndex = len(HAP.shape) - 1

    HAP = np.swapaxes(HAP, 0, lastIndex)
    ratio1 = (np.min(HAP, axis=0) + 1).astype(float) / (np.sum(HAP, axis=0) + 2).astype(float)
    
    varMajority = (ratio1 * 2) ** 2
    varMinority = (2 - (ratio1 * 2)) ** 2
    varAvg = (varMajority * (1-ratio1)) + (varMinority * ratio1)

    HAPtotal = np.sum(HAP, axis=0)

    varAll = varAvg / (HAPtotal + 0.001)

    varAll = varAll * 0.25 #Since it's BAF 0 to 1, rather than bias -1 to 1

    #print (varAll)
    #quit()

    return varAll





def estimateRDRnoise(noise_mini):

    noise_mini_fft = np.fft.fft(noise_mini, axis=1) 
    noise_mini_fft = noise_mini_fft / (noise_mini.shape[1] ** 0.5) #This just corrects for the scaling included in np.fft.fft
    noise_mini_fft = np.abs(noise_mini_fft)[:, 1:] ** 2
    noise_mini_fft = noise_mini_fft / noise_mini.shape[1] #To make it an average not a sum

    arange1 = np.arange(noise_mini_fft.shape[1]) + 2#1
    arange2 = np.min(  np.array([arange1, arange1[-1::-1] ]), axis=0 )
    arange3 = arange2
    
    arange3 = arange3.reshape((1, -1))
    
    noise_mini_val = noise_mini_fft / arange3
    noise_mini_val = np.sum(noise_mini_val, axis=1) #** 0.5

    return noise_mini_val


    

def multiHapNoise(HAP1):

    weight1 = np.sum(HAP1, axis=2)
    weight_sum = np.sum(weight1, axis=1)

    argValid = np.argwhere(weight_sum != 0)[:, 0]
    
    std3 = np.zeros(weight_sum.shape[0]) + 1e10
    if argValid.shape[0] != 0:

        
        
        weight1 = weight1[argValid]
        HAP1 = HAP1[argValid]
        weight_sum = weight_sum[argValid]


        HAP_total = np.mean(HAP1, axis=1)
        mean2 = HAP_total[:, 1] / np.sum(HAP_total, axis=1)


        
        BAF1 = HAP1[:, :, 1] / (weight1 + 1e-5)
        BAF1_diff = BAF1 - mean2.reshape((-1, 1))

        
        std2 = np.sum(  (weight1 ** 2) * (BAF1_diff ** 2) , axis=1 ) / (weight_sum ** 2)
        


        assert np.min(std2) >= 0

        std2 = std2 + (0.25 / weight_sum) #Adding the intrinsic noise in the BAF

        #if np.sum(HAP1[0]) > 100:
        #    print (std2[0])

        std3[argValid] = std2 #Note: Varience not std.

        #print (HAP1[:10])
        #print (std2[:10])
        
        
    return std3







def findVariableBins(RDR_file, bins_file, chr_file, totalRead_file, doBAF, BAF_file=''):



    def getRDRBreakEvidence(RDR, N0):
        

        diffList = np.zeros(RDR.shape[1])
        
        for a in range( RDR.shape[1] - (N0*2 - 1)  ):

            RDR_now = np.copy(RDR[:, a:a+(N0*2)])
            mean1 = np.mean(RDR_now[:, :N0], axis=1)
            mean2 = np.mean(RDR_now[:, N0:], axis=1)
            RDR_now[:, :N0] = RDR_now[:, :N0] - mean1.reshape((-1, 1))
            RDR_now[:, N0:] = RDR_now[:, N0:] - mean2.reshape((-1, 1)) 

            RDR_noise1 = estimateRDRnoise(RDR_now[:, :N0]) + 1e-5
            RDR_noise2 = estimateRDRnoise(RDR_now[:, N0:]) + 1e-5
            RDR_noise = (RDR_noise1 + RDR_noise2) ** 0.5

            #if 0 in RDR_noise1:
            #    arg1 = np.argwhere(RDR_noise1 == 0)[0, 0]
            #    print (RDR_now[arg1, :N0])
            #    quit()

            diff1 = np.abs(mean1 - mean2) 

            #print (np.mean(diff1))

            
            diff1 = diff1 / RDR_noise

            diff2 = np.mean(diff1 ** 2)


            diffList[a + N0] = diff2
    
        return diffList

    
    def getBAFBreakEvidence(HAP, N0):

        diffList = np.zeros(HAP.shape[1])
        
        for a in range( HAP.shape[1] - (N0*2 - 1)  ):


            HAP1 = np.copy(HAP[:, a:a+(N0*2)])

            if np.sum(HAP1) > 0:
                
                HAP_sum1 = np.sum(HAP1[:, :N0], axis=1)
                HAP_sum2 = np.sum(HAP1[:, N0:], axis=1)

                argValid = np.argwhere( np.logical_and(  np.sum(HAP_sum1, axis=1) != 0 , np.sum(HAP_sum2, axis=1) != 0 )  )[:, 0]

                BAF_diff_all = np.ones(HAP1.shape[0])

                if argValid.shape[0] > 0:
                    
                    HAP_sum1 = HAP_sum1[argValid]
                    HAP_sum2 = HAP_sum2[argValid]

                    if False:
                        BAF_var1 = estimateHapNoise(HAP_sum1)
                        BAF_var2 = estimateHapNoise(HAP_sum2)

                    else:
                        BAF_var1 = multiHapNoise(HAP1[argValid, :N0])
                        BAF_var2 = multiHapNoise(HAP1[argValid, N0:])

                    

                    BAF_noise = (BAF_var1 + BAF_var2) ** 0.5

                    BAF1 = HAP_sum1[:, 1] / np.sum(HAP_sum1, axis=1)
                    BAF2 = HAP_sum2[:, 1] / np.sum(HAP_sum2, axis=1)

                    BAF_diff1 = np.abs(BAF1 - BAF2)

                    BAF_diff1 = BAF_diff1 / BAF_noise

                    BAF_diff_all[argValid] = np.copy(BAF_diff1)

                
                BAF_diff2 = np.mean(BAF_diff_all)

                diffList[a+N0] = BAF_diff2

        return diffList


    def coreFindBins(RDR, doBAF, HAP):


        #if np.sum(HAP) > 0:

        #print (HAP.shape)
        #print (RDR.shape)
        #quit()

        bigScale = 200
        scaleList = [10, 20, 40]#, 80]
        
        RDR_diff = np.zeros(( len(scaleList), RDR.shape[1] ))
        BAF_diff = np.zeros(( len(scaleList), RDR.shape[1] ))

        splitPoints = np.zeros(RDR.shape[1])
        #mask1 = np.zeros(RDR.shape[1])
        mask1 = np.zeros(( 2, len(scaleList), RDR.shape[1] ))
        mask2 = np.zeros(RDR.shape[1])

        for a in range(len(scaleList)):
            scale1 = scaleList[a]
            RDR_diff1 = getRDRBreakEvidence(RDR, scale1)
            RDR_diff[a] = np.copy(RDR_diff1)

        for a in range(len(scaleList)):
            scale1 = scaleList[a]
            BAF_diff1 = getBAFBreakEvidence(HAP, scale1)
            BAF_diff[a] = np.copy(BAF_diff1)



        
        if False:

            plt.plot(RDR_diff.T)
            plt.plot(BAF_diff.T)
            plt.show()


            plt.plot(RDR_diff.T)
            plt.show()

            plt.plot(BAF_diff.T)
            plt.show()

            for a in range(RDR_diff.shape[0]):
                print (a)
                plt.plot(RDR_diff[a])
                plt.show()
            
            for a in range(BAF_diff.shape[0]):
                print (a)
                plt.plot(BAF_diff[a])
                plt.show()
            quit()

        
        both_diff = np.array([RDR_diff, BAF_diff])



        #print ('max RDR diff', np.max(RDR_diff))
        #print ('max BAF diff', np.max(BAF_diff))

        #print ('max both diff', np.max(both_diff))
        #quit()


        while np.max( both_diff[mask1==0] ) > 3:
            
            #max1 = np.max( both_diff[:, :, mask1==0] ) 
            max1 = np.max( both_diff[mask1 == 0] ) 
            #argMask = np.argwhere(mask1 == 0)[:, 0]
            #arg1 = np.argwhere( both_diff[:, :, argMask] == max1 )[0]
            arg1 = np.argwhere( np.logical_and( both_diff == max1, mask1 == 0  ) )[0]
            #arg1[2] = argMask[arg1[2]]

            #print (arg1)

            scaleSizeNow = scaleList[arg1[1]]

            for b in range(len(scaleList)):
                scaleSize = scaleList[b]
                scaleSize = min(scaleSize, scaleSizeNow)

                #print (b, mask1.shape)
                mask1[:, b, arg1[2]-scaleSize: arg1[2]+scaleSize] = 1
            #mask1[arg1[2]-scaleSize: arg1[2]+scaleSize] = 1
            
            splitPoints[arg1[2]] = 1

            start2, end2 = max(0, arg1[2]-bigScale), min(RDR.shape[1], arg1[2]+bigScale )
            mask2[start2:end2] = 1

            #print (np.sum(mask2))


        mask2[:bigScale] = 1
        mask2[-bigScale:] = 1


        #print ("A")
        #argSplit = np.argwhere(splitPoints == 1)[:, 0]
        #print (np.min(  argSplit[1:] - argSplit[:-1] ))

        while np.min(mask2) == 0:

            #print (both_diff.shape)
            
            max1 = np.max( both_diff[:, :, mask2==0] ) 
            argMask = np.argwhere(mask2 == 0)[:, 0]
            #print (max1, np.max(both_diff[:, :, argMask]) )
            arg1 = np.argwhere( both_diff[:, :, argMask] == max1 )[0]
            arg1[2] = argMask[arg1[2]]

            #assert mask2[arg1[2]] == 0

            scaleSize = scaleList[arg1[1]]

            splitPoints[arg1[2]] = 1

            #print (mask2[arg1[2]])

            start2, end2 = max(0, arg1[2]-bigScale), min(RDR.shape[1], arg1[2]+bigScale )
            mask2[start2:end2] = 1

            #print (mask2[arg1[2]])

            #print (np.sum(mask2))


        #splitPoints = np.concatenate( (np.zeros(1),  splitPoints[:-1]))

        argSplit = np.argwhere(splitPoints == 1)[:, 0]

        #print (np.min(  argSplit[1:] - argSplit[:-1] ))

        bins = np.cumsum(splitPoints)

        

        return bins



    #N0 = 200
    #N1 = 1000
    #N2 = 2000


    #N0 = 20

    N0 = 5
    




    #N0 = 50
    N1 = 100
    N2 = 200

    #argGood = loadnpz(goodSubset_file)

    RDR = loadnpz(RDR_file)
    chr = loadnpz(chr_file)

    if doBAF:
        BAF = loadnpz(BAF_file)
    



    #print (chr.shape)
    #print (RDR.shape)
    #quit()

    #plt.plot(np.mean(RDR[:, 208:560], axis=0))
    #plt.show()

    #RDR = RDR[:, argGood]
    #chr = chr[argGood]

    #adjust = loadnpz(adjustment_file)
    #adjust = adjust[argGood]

    #adjust = adjust[:RDR.shape[1]]

    #RDR = RDR / adjust.reshape((1, -1))


    #mean1 = np.mean(RDR, axis=0)
    #std1 = np.mean(  (RDR - mean1.reshape((1, -1))) ** 2.0 , axis=0) ** 0.5

    #plt.plot(mean1)
    #plt.plot(mean1 + std1, color='red')
    #plt.plot(mean1 - std1, color='red')
    #plt.show()
    #quit()

    #print (RDR.shape)
    #quit()

    #totalReads = loadnpz(totalRead_file)

    #RDR = RDR * totalReads.reshape((-1, 1))



    unique1 = np.unique(chr)

    bins = np.zeros(chr.shape[0], dtype=int) - 1

    for a in range(unique1.shape[0]):
        #for a in range(1):

        #print (a)


        args1 = np.argwhere(chr == unique1[a])[:, 0]
        RDR_mini = RDR[:, args1]

        if doBAF:
            BAF_mini = BAF[:, args1]
        else:
            BAF_mini = ''
        #print (args1)
        #quit()

        #print (RDR_mini.shape)
        #quit()

        #print (RDR.shape)
        #print (RDR_mini.shape)
        #quit()

        #print (RDR_mini.shape)
        #print (BAF_mini.shape)
        
        bins_mini = coreFindBins(RDR_mini, doBAF, BAF_mini)

        #quit()


        _, index1 = np.unique(bins_mini, return_index=True)
        #print (index1)



        bins_mini = bins_mini + np.max(bins) + 1

        #_, counts = np.unique(bins_mini, return_counts=True)
        #print (np.min(counts))
        #quit()


        bins[args1] = bins_mini

    
    np.savez_compressed(bins_file, bins)





#chr_file = './data/DLP/initial/chr_100k.npz'
#RDR_file = './data/DLP/initial/RDR_100k.npz'
#bins_file = './data/DLP/binScale/bins.npz'
#totalRead_file = './data/DLP/initial/totalReads.npz'
#findVariableBins(RDR_file, bins_file, chr_file, totalRead_file)
#quit()

#folder1 = 'DLP'
#folder1 = '10x'
#folder1 = 'ACT10x'
folder1 = 'TN3'
chr_file = './data/' + folder1 + '/initial/chr_100k.npz'
RDR_file = './data/' + folder1 + '/initial/RDR_100k.npz'
BAF_file = './data/' + folder1 + '/initial/HAP_100k.npz'
bins_file = './data/' + folder1 + '/binScale/bins.npz'
totalRead_file = './data/' + folder1 + '/initial/totalReads.npz'
doBAF = True


#bins = loadnpz(bins_file)
#_, index1 = np.unique(bins, return_index=True)
#print (index1[:10])
#print (index1.shape)
#quit()




#findVariableBins(RDR_file, bins_file, chr_file, totalRead_file, doBAF, BAF_file=BAF_file)
#quit()





def saveVairableBinPosition(chr_file, bins_file, chr_file_many, goodSubset_file, positionList_file):


    chr_100k = loadnpz(chr_file)
    chr_10k = loadnpz(chr_file_many)
    bins = loadnpz(bins_file)
    goodSubset = loadnpz(goodSubset_file)

    uniqueChr = np.unique(chr_100k)

    _, chr_10k_index = np.unique(chr_10k, return_index=True)

    uniqueBins = np.unique(bins)

    positionList = np.zeros((uniqueBins.shape[0], 3), dtype=int)

    for a in range(uniqueChr.shape[0]):

        goodSubset_local = goodSubset[chr_10k[goodSubset] == uniqueChr[a]]
        goodSubset_local = goodSubset_local - chr_10k_index[a]

        binsNow = bins[chr_100k == uniqueChr[a]]

        binsNow_unique, indexFirst = np.unique(binsNow, return_index=True)
        _, indexLast = np.unique(binsNow[-1::-1], return_index=True)
        indexLast = binsNow.shape[0] - 1 - indexLast

        indexFirst = indexFirst * 10
        indexLast = (indexLast * 10) + 9
        
        indexFirst = goodSubset_local[indexFirst]
        indexLast = goodSubset_local[indexLast]

        indexFirst = indexFirst * 10000
        indexLast = ((indexLast+1) * 10000) - 1

        

        positionList[binsNow_unique, 0] = a
        positionList[binsNow_unique, 1] = np.copy(indexFirst)
        positionList[binsNow_unique, 2] = np.copy(indexLast)

        
    
    np.savez_compressed(positionList_file, positionList)




#folder1 = 'DLP'
#folder1 = '10x'
#folder1 = 'ACT10x'
#folder1 = 'TN3'
#chr_file = './data/' + folder1 + '/initial/chr_100k.npz'
#bins_file = './data/' + folder1 + '/binScale/bins.npz'
#chr_file_many = './data/' + folder1 + '/initial/allChr_10k.npz'
#goodSubset_file = './data/' + folder1 + '/initial/subset.npz'
#positionList_file = './data/' + folder1 + '/initial/binPositions.npz'
#saveVairableBinPosition(chr_file, bins_file, chr_file_many, goodSubset_file, positionList_file)
#quit()






def applyVariableBins(RDR_file, bins_file, chr_file, RDR_file2, noise_file, chr_file2, doBAF, BAF_file='', BAF_file2='', BAF_noise_file=''):


    data = loadnpz(RDR_file)

    bins = loadnpz(bins_file)
    #call = loadnpz(call_file)
    chr = loadnpz(chr_file)

    #adjustment = loadnpz(adjustment_file)

    unique1 = np.unique(bins)
    Nbin = unique1.shape[0]


    data_avg = np.zeros((data.shape[0], Nbin))
    noise1 = np.zeros((data.shape[0], Nbin))

    #adjust_avg = np.zeros(Nbin)

    BAF_noise = np.zeros((data.shape[0], Nbin))

    #call_avg = np.zeros((data.shape[0], Nbin), dtype=int)
    chr_avg = np.zeros(Nbin, dtype=int)

    if doBAF:
        HAP = loadnpz(BAF_file)
        HAP_sum = np.zeros((data.shape[0], Nbin, 2))
    

    for count1 in range(unique1.shape[0]):

        args1 = np.argwhere(bins == unique1[count1])[:, 0]

        HAP_sum[:, count1] = np.sum(HAP[:, args1], axis=1)
        
        #data_avg[:, count1] = np.mean(data[:, args1], axis=1)
        data_avg[:, count1] = np.median(data[:, args1], axis=1)

        #adjust_avg[count1] = np.mean(adjustment[args1])


        HAP_totals = np.sum(HAP[:, args1], axis=0)
        BAF_chunk = np.sum(HAP_totals, axis=0)
        #print (BAF_chunk)
        BAF_chunk = BAF_chunk[0] / (np.sum(BAF_chunk) + 1e-6)

        BAF_totals = HAP_totals[:, 0] / (np.sum(HAP_totals, axis=1) + 1e-6)


        BAF_var = multiHapNoise(HAP[:, args1]) ** 0.5

        BAF_noise[:, count1] = np.copy(BAF_var)

        
    
        
        noise_mini = data[:, args1] - data_avg[:, count1].reshape((-1, 1))

        
        noise_mini_val = estimateRDRnoise(noise_mini) ** 0.5


        noise1[:, count1] =  noise_mini_val

        
        chr_avg[count1] = chr[args1[0]]



    

    for a in range(data.shape[0]):
        noise1[a] = noise1[a] / np.mean(data_avg[a])
        data_avg[a] = data_avg[a] / np.mean(data_avg[a])


    np.savez_compressed(RDR_file2, data_avg)
    np.savez_compressed(chr_file2, chr_avg)
    np.savez_compressed(noise_file, noise1)


    if doBAF:
        np.savez_compressed(BAF_file2, HAP_sum)
        np.savez_compressed(BAF_noise_file, BAF_noise)
        








bins_file = './data/' + folder1 + '/binScale/bins.npz'
chr_file = './data/' + folder1 + '/initial/chr_100k.npz'
RDR_file = './data/' + folder1 + '/initial/RDR_100k.npz'
RDR_file2 = './data/' + folder1 + '/binScale/filtered_RDR_avg.npz'
noise_file = './data/' + folder1 + '/binScale/filtered_RDR_noise.npz'
chr_file2 = './data/' + folder1 + '/binScale/chr_avg.npz'
doBAF = True
BAF_file = './data/' + folder1 + '/initial/HAP_100k.npz'
BAF_file2 = './data/' + folder1 + '/binScale/filtered_HAP_avg.npz'
BAF_noise_file = './data/' + folder1 + '/binScale/BAF_noise.npz'

#applyVariableBins(RDR_file, bins_file, chr_file, RDR_file2, noise_file, chr_file2, doBAF, BAF_file=BAF_file, BAF_file2=BAF_file2, BAF_noise_file=BAF_noise_file)
#quit()





def mapBAF(x):

    print ("mapBAF is no longer used!")
    error1 = intentionalError
    quit()
    return (x * 0.8) + 0.05




#RDR = loadnpz('./data/input/S' + '1' + '_RDR.npz')
#print (RDR.shape)
#quit()









def findRegions(RDR_file, BAF_file, chr_File, region_file):

    def giveMode(ar):
        unique1, count1 = np.unique(ar, return_counts=True)
        maxArg = np.argmax(count1)
        return unique1[maxArg]


    def findBestRegion(RDR, HAP):#, N):



        RDR = RDR - np.mean(RDR)
        RDR_cumsum = paddedCumSum(RDR)
        RDR_sq_cumsum = paddedCumSum(RDR**2)

       

        if type(HAP) != type(''):
            BAF = HAP[:, 1] / (np.sum(HAP, axis=1) + 1e-5) #Just removing division by zero
            weight = np.sum(HAP, axis=1)
            
            

            BAF_cumsum = paddedCumSum(BAF*weight)
            BAF_sq_cumsum = paddedCumSum( (weight * BAF)**2)
            weight_cumsum = paddedCumSum(weight)
            weight_sq_cumsum = paddedCumSum(weight ** 2)


        size1 = RDR.shape[0]


        N = 1
        sizeRound = ((RDR.shape[0] - 1) // N) + 1
        

        data1 = np.zeros((  sizeRound ** 2, 3))
        count1 = 0
        for a0 in range(sizeRound):
            for b0 in range(sizeRound):
                a = (a0 * N)
                b = (b0 * N)


                if b >= (a + 5):
                    

                    if True:

                        length0 = b - a
                        error1_sq = (RDR_sq_cumsum[b] - RDR_sq_cumsum[a]) / length0
                        error1_me = (RDR_cumsum[b] - RDR_cumsum[a]) / length0

                        error1 = error1_sq - (error1_me ** 2)

                        error1 = (error1 / length0) ** 0.5



                        if type(BAF) != type(''):

                            
                            weight_sum = weight_cumsum[b] - weight_cumsum[a]

                            if weight_sum == 0:
                                error2 = 1e5 #infinity for all practical purposes
                            else:
                                weight_sq = weight_sq_cumsum[b] - weight_sq_cumsum[a]
                                error2_sq = (BAF_sq_cumsum[b] - BAF_sq_cumsum[a]) / (weight_sum ** 2)
                                error2_me = (BAF_cumsum[b] - BAF_cumsum[a]) / weight_sum


                                error2 = error2_sq - ((error2_me ** 2) * weight_sq / (  weight_sum ** 2  ) )

                                if error2 < 0:
                                    error2 = 0


                                error2 = error2 + (0.25 / (weight_sum) ) #Adding the intrinsic noise in the BAF

                               


                                error2 = error2 ** 0.5
                        

                        if type(BAF) != type(''):
                            error1 = 1/((1 / error1) + (1 / error2))

                        data1[count1, 0] = a
                        data1[count1, 1] = b
                        data1[count1, 2] = error1

                        count1 += 1




        data1 = data1[:count1]





        min1 = np.argmin(data1[:, 2])

        start1, end1 = int(data1[min1, 0]), int(data1[min1, 1])


        return start1, end1


    def findMultiRegion(RDR, BAF):#, N):

        regionToCheck = np.zeros((1, 2), dtype=int)
        regionToCheck[0, 1] = RDR.shape[0]

        regionDone = np.zeros((1000, 2), dtype=int)

        

        doneCount1 = 0

        while regionToCheck.shape[0] > 0:

            regionToCheck_new = np.zeros((regionToCheck.shape[0] * 2, 2), dtype=int)
            checkCount1 = 0

            for a in range(regionToCheck.shape[0]):
                start1, end1 = regionToCheck[a, 0], regionToCheck[a, 1]
                RDR1 = RDR[start1:end1]
                if type(BAF) == type(''):
                    BAF1 = ''
                else:
                    BAF1 = BAF[start1:end1]




                start2, end2 = findBestRegion(RDR1, BAF1)#, N)

                regionDone[doneCount1, 0] = start1 + start2
                regionDone[doneCount1, 1] = start1 + end2
                doneCount1 += 1

                if start2 > 5:
                    regionToCheck_new[checkCount1, 0] = start1
                    regionToCheck_new[checkCount1, 1] = start1 + start2
                    checkCount1 += 1
                if end2 < RDR1.shape[0] - 5:
                    regionToCheck_new[checkCount1, 0] = start1 + end2
                    regionToCheck_new[checkCount1, 1] = end1
                    checkCount1 += 1

            regionToCheck_new = regionToCheck_new[:checkCount1]
            regionToCheck = regionToCheck_new


        regionDone = regionDone[:doneCount1]

        

        return regionDone


    def findAllRegions(RDR, BAF, start1, end1, chr):

        RDR_change = np.mean(np.abs(RDR[1:]-RDR[:-1]))
        

        regions = np.zeros((10000, 2), dtype=int)

        count1 = 0
        for a in range(start1.shape[0]):
            subset1 = np.arange(end1[a] - start1[a]) + start1[a]
            if type(BAF) != type(''):
                regionDone = findMultiRegion(RDR[subset1], BAF[subset1])#, N)
            else:
                regionDone = findMultiRegion(RDR[subset1], '')#, N)
            regionDone = regionDone + start1[a]
            size1 = regionDone.shape[0]

            
            regions[count1:count1+size1] = regionDone
            
            count1 += size1
            



        regions = regions[:count1]

        
        sizes = regions[:, 1] - regions[:, 0]
        regions = regions[sizes > 6]
        

        regions = regions[np.argsort(regions[:, 0])]

        

        return regions


    

    


    RDR_all = loadnpz(RDR_file)

    


    if BAF_file == '':
        BAF_all = ''
    else:
        BAF_all = loadnpz(BAF_file)

    

    chr = loadnpz(chr_File)


    _, start1 = np.unique(chr, return_index=True)
    end1 = np.concatenate((start1[1:], np.zeros(1) + chr.shape[0])).astype(int)

    regionList = np.zeros((RDR_all.shape[0]*RDR_all.shape[1], 3), dtype=int)

    count1 = 0

    created1 = False

    perm1 = np.random.permutation(RDR_all.shape[0])

    #54

    
    for a0 in tqdm(range(0, RDR_all.shape[0])):

        #a = perm1[a0]
        a = a0



        #print (a, RDR_all.shape[0])

        
        RDR = RDR_all[a]
        if type(BAF_all) == type(''):
            BAF = ''
            BAF1 = ''
        else:
            BAF1 = BAF_all[a]
            
            

        regions = findAllRegions(RDR, BAF1, start1, end1, chr)

        


        size1 = regions.shape[0]
        count2 = count1 + size1

        regionList[count1:count2, 0] = a
        regionList[count1:count2, 1:] = regions






        count1 = count2


    regionList = regionList[:count1]

    np.savez_compressed(region_file, regionList)







#chr_file = outFolder + '/initial/chr_1M.npz'
#RDR_file = outFolder + '/initial/RDR_1M.npz'
#region_file = outFolder + '/binScale/regions.npz'
#HAP_file = outFolder + '/initial/HAP_1M.npz'
#findRegions(RDR_file, HAP_file, chr_file, region_file)
#quit()








def findDividers(RDR_file, HAP_file, chr_File, divider_file, error_file, dividerList_file, region_file, precise=True, naive=False, maxPloidy=10):



    def calculateRegionMeans(RDR, HAP, regions):

        if type(HAP) == type(''):
            means1 = np.zeros((regions.shape[0], 1))
            vars1 = np.zeros((regions.shape[0], 1))
        else:
            means1 = np.zeros((regions.shape[0], 2))
            vars1 = np.zeros((regions.shape[0], 2))

        for a in range(regions.shape[0]):
            start2, end2 = regions[a, 0], regions[a, 1]
            subset1 = np.arange(end2 - start2) + start2


            RDR1 = RDR[subset1]
            mean1 = np.mean(RDR1)
            means1[a, 0] = mean1

            #std1 = estimateRDRnoise(RDR1.reshape((1, -1)) - mean1)
            #std1 = std1[0]
            
            if True:
                std1 = np.mean( (RDR1 - mean1) ** 2 )
                length1 = float(RDR1.shape[0] - 1)
                std1 = std1 / (length1 )
            
            std1 = std1 ** 0.5
            vars1[a, 0] = std1


            if type(HAP) != type(''):
                HAP1 = HAP[subset1]

                #HAP1 = np.array([  [400, 0], [0, 400], [400, 0], [0, 400], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0] ])

                weight1 = np.sum(HAP1, axis=1)
                weight_sum = np.sum(weight1)
                if weight_sum == 0:
                    mean2, std2 = 0.5, 1e5 #1e5 is approximately infinity
                else:
                    
                    #print (HAP1.shape)
                    #print (multiHapNoise(HAP1.reshape((1, HAP1.shape[0], HAP1.shape[1]))  ))
                    #quit()

                    BAF1 = HAP1[:, 0] / (weight1 + 1e-5)


                    mean2 = np.mean(BAF1 * weight1) / np.mean(weight1)

                    #print (mean2)

                    weight_sq = np.sum( weight1 ** 2 )
                    error2_sq = np.sum(  (weight1 ** 2) * (BAF1 ** 2)  ) / (weight_sum ** 2)

                    #print (error2_sq)
                    #error2_sq = (BAF_sq_cumsum[b] - BAF_sq_cumsum[a]) / (weight_sum ** 2)
                    #error2_me = (BAF_cumsum[b] - BAF_cumsum[a]) / weight_sum

                    std2 = error2_sq - ((mean2 ** 2) * weight_sq / (  weight_sum ** 2  ) )

                    if std2 < 0:
                        std2 = 0

                    
                    std2 = std2 + (0.25 / weight_sum ) #Adding the intrinsic noise in the BAF

                    #print (std2)
                    #quit()
                    
                    std2 = std2 ** 0.5


                    if False:#weight_sum > 200:
                        #print (HAP1)
                        #print (BAF1)
                        print (std1)
                        print (std2)

                        quit()

                means1[a, 1] = mean2
                vars1[a, 1] = std2


        return means1, vars1


    def addNonRegion(means1, vars1, RDR, HAP, regions):


        bool1 = np.zeros(RDR.shape[0], dtype=int)
        for a in range(regions.shape[0]):
            bool1[regions[a, 0]:regions[a, 1]] = 1
        argAll = np.argwhere(bool1 == 0)[:, 0]

        if type(HAP) == (''):
            means2, vars2 = np.zeros((argAll.shape[0], 1)), np.zeros((argAll.shape[0], 1))
        else:
            means2, vars2 = np.zeros((argAll.shape[0], 2)), np.zeros((argAll.shape[0], 2))


        means2[:, 0] = RDR[argAll]


        #var1 = np.mean(RDR ** 2)
        #var2 = np.mean(BAF ** 2)
        #vars2[:, 0] = var1
        #vars2[:, 1] = var2

        var1 = np.mean(np.abs(RDR[1:] - RDR[:-1])) #** 2
        vars2[:, 0] = var1

        if type(HAP) != (''):
            
            HAP1 = HAP[argAll]
            BAF = HAP1[:, 1] / (np.sum(HAP1, axis=1) + 1e-3)
            print (BAF.shape)
            means2[:, 1] = BAF


            #var2 = np.mean(np.abs(BAF[1:] - BAF[:-1])) #** 2
            var2 = estimateHapNoise(HAP1)
            var2 = var2 ** 0.5
            vars2[:, 1] = var2

        means1 = np.concatenate((means1, means2), axis=0)
        vars1 = np.concatenate((vars1, vars2), axis=0)

        #quit()
        return means1, vars1


    def checkDivider(means1, vars1, divider, noises1, doPrint=False):

        intChr = means1[:, 0] / divider
        intChr = np.floor(intChr + 0.5).astype(float)

        error1 = means1[:, 0] - (intChr * divider)
        error1 = error1 / (vars1[:, 0] + 0.001)

        #print (np.sum(noises1))
        #quit()


        if noises1 != '':
            error1 = (error1 + np.sum(noises1[:, 0] * 0.2)) / divider
        else:
            #Original Jan 12
            #error1 = error1 / divider
            #After
            error1 = (error1 / divider) ** 2.0




        error1 = np.abs(error1)
        #print (vars1[:, 0])
        #print (error1)
        error1 = np.sum(error1)

        if means1.shape[1] == 2:
            Bnum = []

            error2 = 0
            for a in range(means1.shape[0]):

                error3_list = []
                int1 = int(intChr[a])

                for b in range( (int1 // 2) + 1 ):
                    int2 = float(int1) + 0.001
                    #ratio1 = (float(b) / int2) * 0.9
                    
                    #ratio1 = mapBAF(float(b) / int2)
                    ratio1 = float(b) / int2

                    error3 = np.abs(ratio1 - means1[a, 1])
                    error3 = error3 / (vars1[a, 1] + 0.01)

                    #Original Jan 12
                    #error3 = error3 / (divider ** 0.5)
                    #After
                    error3 = (error3 / divider)**2
                    #error3 = (error3) / divider

                    error3_list.append(error3)

                #print ('error3_list', len(error3_list))


                error3_list = np.array(error3_list)

                error3 = np.min(error3_list)
                error2 += error3

                argMin = np.argmin(error3_list)
                #Bnum.append(argMin)




            error1 = error1 + error2 #+ np.log((1 / divider)+1)

        #return error1, Bnum

        return error1


    def findBestDivider(means1, vars1, noises1, precise, maxPloidy=10):

        #print ('noise')
        #print (np.sum(noises1[:, 0]))

        errorList = []
        dividerList = []
        #BnumAll = np.zeros((50, means1.shape[0]))
        #for b in range(50):
        #    divider = 0.2 + (b * 0.01)

        


        Ncheck = 100
        tickSize = 0.02

        #Ncheck = 400
        #tickSize = 0.005

        for b in range(Ncheck):
            divider = 0.1 * np.exp(b * tickSize) #0.1

            if (1.0 / divider) < maxPloidy:

                #for b in range(200):
                #    divider = 0.1 * np.exp(b * 0.01)

                #divider = 0.15 * np.exp(b * 0.02)

                dividerList.append(divider)

                #print (divider)
                #plt.plot(means1 / divider)
                #plt.plot(np.floor(  (means1 / divider) + 0.5))
                #plt.show()

                #print (means1.shape)

                error1 = checkDivider(means1, vars1, divider, noises1)
                #BnumAll[b] = Bnum
                errorList.append(error1)

        #plt.plot(dividerList, errorList)
        #plt.show()

        errorList = np.array(errorList)
        dividerList = np.array(dividerList)
        argMin = np.argmin(errorList)
        divideGood = dividerList[argMin]


        if precise:

            Nprecise = 20

            isLimited = False

            if argMin >= dividerList.shape[0] - 1:
                argMin = dividerList.shape[0] - 2
                isLimited = True

            #print (argMin)

            try:
                dividerList[argMin + 1]
            except:
                print ("dividerList[argMin + 1] failed in scaler.py")
                print ('argMin', argMin)
                print ('dividerList', dividerList.shape)
                print ('errorList', errorList.shape)
                print ('isLimited', isLimited)



            dividerBefore = dividerList[argMin - 1]
            dividerAfter = dividerList[argMin + 1]
            dividerTick = (dividerAfter - dividerBefore) / float(Nprecise)

            errorList2 = []
            dividerList2 = []
            for c in range(Nprecise):
                divider = dividerBefore + (c * dividerTick)
                error1 = checkDivider(means1, vars1, divider, noises1)
                errorList2.append(error1)
                dividerList2.append(divider)

            errorList2 = np.array(errorList2)
            dividerList2 = np.array(dividerList2)
            argMin2 = np.argmin(errorList2)
            divideGood = dividerList2[argMin2]



        return divideGood, dividerList, errorList


    def calculateMultiRegionMean(RDR, BAF, regions):

        #print ('banana')
        #print (regions)

        bool1 = np.zeros(RDR.shape[0], dtype=int)
        for b in range(regions.shape[0]):
            start2, end2 = regions[b, 0], regions[b, 1]
            bool1[start2:end2] = 1
        subset1 = np.argwhere(bool1 == 1)[:, 0]

        RDR1 = RDR[subset1]
        BAF1 = BAF[subset1]

        mean1 = np.mean(RDR1)
        mean2 = np.mean(BAF1)

        #means1[a, 0] = mean1
        #means1[a, 1] = mean2

        noise1, noise2 = np.sum( (RDR1 - mean1) ** 2 ), np.sum( (BAF1 - mean2) ** 2 )

        std1 = np.mean( (RDR1 - mean1) ** 2 )
        std2 = np.mean( (BAF1 - mean2) ** 2 )
        length1 = float(RDR1.shape[0] - 1)
        std1 = std1 / (length1 )
        std2 = std2 / (length1 )

        std1 = std1 ** 0.5
        std2 = std2 ** 0.5

        meanBoth = np.array([mean1, mean2])
        stdBoth = np.array([std1, std2])
        noiseBoth = np.array([noise1, noise2])

        return meanBoth, stdBoth, noiseBoth


    def findAllMultiMean(RDR, BAF, regions):

        unique1 = np.unique(regions[:, 0])

        means1 = np.zeros((unique1.shape[0], 2))
        stds1 = np.zeros((unique1.shape[0], 2))
        noises1 = np.zeros((unique1.shape[0], 2))

        for a in range(unique1.shape[0]):
            subset1 = np.argwhere(regions[:, 0] == unique1[a])[:, 0]

            meanBoth, stdBoth, noiseBoth = calculateMultiRegionMean(RDR, BAF, regions[subset1, 1:])

            means1[a] = np.copy(meanBoth)
            stds1[a] = np.copy(stdBoth)
            noises1[a] = np.copy(noiseBoth)

        return means1, stds1, noises1




    def giveMultiRegionError(RDR, BAF, regions):

        bool1 = np.zeros(RDR.shape[0], dtype=int)
        for b in range(regions.shape[0]):
            start2, end2 = regions[b, 0], regions[b, 1]
            bool1[start2:end2] = 1
        subset1 = np.argwhere(bool1 == 1)[:, 0]


        RDR1 = RDR[subset1]
        BAF1 = BAF[subset1]

        size1 = -100

        mean1 = np.mean(RDR1)
        error1 = np.sum( np.abs(  RDR1 - mean1  ) )

        mean2 = np.mean(BAF1)
        error2 = np.sum( np.abs(  BAF1 - mean2  ) )
        error2 = error2 * 0.1

        error1 = error1 + error2


        error1 = error1 + 0.8

        length1 = subset1.shape[0]
        length1 = length1 - 1

        error1 = error1 / length1

        return error1


    def doRegionReduction(RDR, BAF, regions):


        cutOff = 0.02

        size1 = 0

        while np.unique(regions[:, 0]).shape[0] != size1:

            means1, vars1, noises1 = findAllMultiMean(RDR, BAF, regions)
            _, inverse1 = np.unique(regions[:, 0], return_inverse=True)
            regions = regions[np.argsort(vars1[inverse1, 0])]

            unique1 = np.unique(regions[:, 0])

            size1 = unique1.shape[0]

            for a in range(unique1.shape[0]):
                for b in range(unique1.shape[0]):
                    subset1 = np.argwhere(regions[:, 0] == unique1[a])[:, 0]
                    subset2 = np.argwhere(regions[:, 0] == unique1[b])[:, 0]

                    if (subset1.shape[0] > 0) and (subset2.shape[0] > 0):

                        regions1 = regions[subset1, 1:]
                        regions2 = regions[subset2, 1:]

                        mean1, std1, noise1 = calculateMultiRegionMean(RDR, BAF, regions1)
                        mean2, std2, noise2 = calculateMultiRegionMean(RDR, BAF, regions2)

                        if abs(mean1[0] - mean2[0]) < cutOff:

                            regions3 = np.concatenate((regions1, regions2), axis=0)

                            error1 = giveMultiRegionError(RDR, BAF, regions1)
                            error2 = giveMultiRegionError(RDR, BAF, regions2)
                            error3 = giveMultiRegionError(RDR, BAF, regions3)

                            if error3 < min(error1, error2):

                                regions[subset2, 0] = unique1[a]

            if (size1 == np.unique(regions[:, 0]).shape[0]) and (cutOff in [0.02]):
                cutOff = 0.05
                size1 = 0

            #print ('')
            #print (regions)

        regions = regions[np.argsort(regions[:, 1])]

        return regions


    patientNum0 = '1'


    #x = loadnpz('./data/input/S0.npz')
    #x = loadnpz('./data/input/filtered_S' + patientNum0 + '.npz')
    RDR_all = loadnpz(RDR_file)

    ##RDR_all = RDR_all[:100] #TODO remove
    if HAP_file == '':
        HAP_all = ''
    else:
        HAP_all = loadnpz(HAP_file)


    #x = x[:500]

    #chr = loadnpz('./data/input/chr_S' + patientNum0 + '.npz')

    #argGood = loadnpz('./data/input/argFilter_S0.npz')
    #dataCall = loadnpz('./data/input/call_S0.npz')[argGood]


    #rand1 = np.random.permutation(x.shape[0])
    #np.savez_compressed('./data/input/random.npz', rand1)
    #rand1 = loadnpz('./data/input/random.npz')
    #rand1 = rand1[:100]
    #xSample = x[rand1]
    #calculateCorBoth(xSample, xSample)


    #_, start1 = np.unique(chr, return_index=True)
    #end1 = np.concatenate((start1[1:], np.zeros(1) + chr.shape[0])).astype(int)

    #end1 = np.concatenate((start1[1:] + 1, np.zeros(1) + chr.shape[0])).astype(int)


    #170, 341

    dividerNums = []

    #divideAll = []
    #divideError = []

    created1 = False

    count1 = 0
    count2 = 0

    if naive:
        regionList = np.zeros((1, 3)) - 1
    else:
        regionList = loadnpz(region_file)

    perm1 = np.random.permutation(RDR_all.shape[0])



    #for rand1 in np.random.randint(x.shape[0], size=100):
    for a in tqdm(range(RDR_all.shape[0])):

        #a = perm1[a]

        RDR = RDR_all[a]
        if type(HAP_all) == type(''):
            HAP = ''
            HAP1 = ''
        else:
            HAP = HAP_all[a]
            #HAP1 = np.min(np.array([BAF, 1 - BAF]), axis=0)
            HAP1 = HAP

        time1 = time.time()

        count2 = count1
        while regionList[count2 % regionList.shape[0], 0] == a:
            count2 += 1
        

        regions = regionList[count1:count2, 1:]
        count1 = count2



        if True:
            #print (regions.shape)

            means1, vars1 = calculateRegionMeans(RDR, HAP, regions)

            vars1[np.isnan(vars1)] = 1e5
            vars1[:, 1] = 1e5

            

            
            
            #means1, vars1 = addNonRegion(means1, vars1, RDR, HAP, regions) 

            noises1 = ''

            divideGood, dividerList, errorList = findBestDivider(means1, vars1, noises1, precise, maxPloidy=maxPloidy)


            

            #The code within the False if statement is for debugging during developement 
            if False:
                print (divideGood)
                meansPlot = np.zeros(RDR.shape[0])
                bool1 = np.zeros(RDR.shape[0])
                for b in range(regions.shape[0]):
                    #print (regions[b, 0], regions[b, 1])
                    bool1[regions[b, 0]:regions[b, 1]] = 1 + (b%2)
                    meansPlot[regions[b, 0]:regions[b, 1]] = np.mean(RDR[regions[b, 0]:regions[b, 1]]) / divideGood

                for b in range(8):
                    plt.plot(np.zeros(RDR.shape[0])+b, c='grey')
                plt.plot(bool1)
                #plt.plot(RDR)
                plt.plot(RDR / divideGood)
                plt.plot(meansPlot)
                #plt.plot(BAF1)
                plt.show()


        dividerNums.append(divideGood)


        if not created1:
            divideAll = np.zeros((RDR_all.shape[0], dividerList.shape[0]))
            divideError = np.zeros((RDR_all.shape[0], errorList.shape[0]))
            created1 = True

        divideAll[a] = dividerList
        divideError[a] = errorList




    dividerNums = np.array(dividerNums)

    #quit()

    #print ("done1")

    np.savez_compressed(divider_file, dividerNums)
    np.savez_compressed(error_file, divideError)

    np.savez_compressed(dividerList_file, divideAll)





#folder1 = 'DLP'
folder1 = '10x'
#folder1 = 'ACT10x'
#folder1 = 'TN3'

RDR_file = './data/' + folder1 + '/initial/RDR_1M.npz'
chr_file = './data/' + folder1 + '/initial/chr_1M.npz'
region_file = './data/' + folder1 + '/binScale/regions.npz'
divider_file = './data/' + folder1 + '/binScale/dividers.npz'
error_file = './data/' + folder1 + '/binScale/dividerError.npz'
dividerList_file = './data/' + folder1 + '/binScale/dividerAll.npz'
HAP_file = './data/' + folder1 + '/initial/HAP_1M.npz'
#findDividers(RDR_file, HAP_file, chr_file, divider_file, error_file, dividerList_file, region_file)
#quit()





def newFindDividers(bins_file, RDR_file, noise_file, BAF_file, BAF_noise_file, divider_file, error_file, dividerList_file):

 
   
    def checkDivider(means1, vars1, divider, noises1, doPrint=False):

        intChr = means1[:, 0] / divider
        intChr = np.floor(intChr + 0.5).astype(float)

        error1 = means1[:, 0] - (intChr * divider)
        error1 = error1 / (vars1[:, 0] + 0.001)

        #print (np.sum(noises1))
        #quit()


        if noises1 != '':
            error1 = (error1 + np.sum(noises1[:, 0] * 0.2)) / divider
        else:
            #Original Jan 12
            #error1 = error1 / divider
            #After
            error1 = (error1 / divider) ** 2.0




        error1 = np.abs(error1)
        #print (vars1[:, 0])
        #print (error1)
        error1 = np.sum(error1)

        if means1.shape[1] == 2:
            Bnum = []

            error2 = 0
            for a in range(means1.shape[0]):

                error3_list = []
                int1 = int(intChr[a])

                for b in range( (int1 // 2) + 1 ):
                    int2 = float(int1) + 0.001
                    #ratio1 = (float(b) / int2) * 0.9
                    
                    #ratio1 = mapBAF(float(b) / int2)
                    ratio1 = float(b) / int2

                    error3 = np.abs(ratio1 - means1[a, 1])
                    error3 = error3 / (vars1[a, 1] + 0.01)

                    #Original Jan 12
                    #error3 = error3 / (divider ** 0.5)
                    #After
                    error3 = (error3 / divider)**2
                    #error3 = (error3) / divider

                    error3_list.append(error3)

                #print ('error3_list', len(error3_list))


                error3_list = np.array(error3_list)

                error3 = np.min(error3_list)
                error2 += error3

                argMin = np.argmin(error3_list)
                #Bnum.append(argMin)




            error1 = error1 + error2 #+ np.log((1 / divider)+1)

        #return error1, Bnum

        return error1


    def findBestDivider(means1, vars1, noises1, precise):

        #print ('noise')
        #print (np.sum(noises1[:, 0]))

        errorList = []
        dividerList = []
        #BnumAll = np.zeros((50, means1.shape[0]))
        #for b in range(50):
        #    divider = 0.2 + (b * 0.01)


        Ncheck = 100
        tickSize = 0.02

        #Ncheck = 400
        #tickSize = 0.005

        for b in range(Ncheck):
            divider = 0.1 * np.exp(b * tickSize) #0.1

            #for b in range(200):
            #    divider = 0.1 * np.exp(b * 0.01)

            #divider = 0.15 * np.exp(b * 0.02)

            dividerList.append(divider)

            #print (divider)
            #plt.plot(means1 / divider)
            #plt.plot(np.floor(  (means1 / divider) + 0.5))
            #plt.show()

            #print (means1.shape)

            error1 = checkDivider(means1, vars1, divider, noises1)
            #BnumAll[b] = Bnum
            errorList.append(error1)

        #plt.plot(dividerList, errorList)
        #plt.show()

        errorList = np.array(errorList)
        dividerList = np.array(dividerList)
        argMin = np.argmin(errorList)
        divideGood = dividerList[argMin]


        if precise:

            Nprecise = 20

            if argMin == Ncheck - 1:
                argMin = Ncheck - 2

            #print (argMin)

            dividerBefore = dividerList[argMin - 1]
            dividerAfter = dividerList[argMin + 1]
            dividerTick = (dividerAfter - dividerBefore) / float(Nprecise)

            errorList2 = []
            dividerList2 = []
            for c in range(Nprecise):
                divider = dividerBefore + (c * dividerTick)
                error1 = checkDivider(means1, vars1, divider, noises1)
                errorList2.append(error1)
                dividerList2.append(divider)

            errorList2 = np.array(errorList2)
            dividerList2 = np.array(dividerList2)
            argMin2 = np.argmin(errorList2)
            divideGood = dividerList2[argMin2]



        return divideGood, dividerList, errorList

    
    
    RDR_all = loadnpz(RDR_file)

    ##RDR_all = RDR_all[:100] #TODO remove
    if HAP_file == '':
        HAP_all = ''
    else:
        HAP_all = loadnpz(HAP_file)


    

    dividerNums = []
    

    created1 = False

    
    RDR = loadnpz(RDR_file)
    noiseRDR = loadnpz(noise_file)
    BAF = loadnpz(BAF_file).astype(float)
    noiseBAF = loadnpz(BAF_noise_file)

    BAF = (BAF[:, :, 1] + 1e-5) / (np.sum(BAF, axis=2) + 1e-5)



    perm1 = np.random.permutation(RDR_all.shape[0])

    precise = True


    bins = loadnpz(bins_file)

    #for rand1 in np.random.randint(x.shape[0], size=100):
    for a in range(0, RDR_all.shape[0]):

        print (a)

        time1 = time.time()

        print (RDR.shape)
        print (BAF.shape)
        
        means1 = np.array([ RDR[a], BAF[a] ]).T 
        vars1 = np.array([ noiseRDR[a], noiseBAF[a] ]).T 


        if True:
            #print (regions.shape)

            noises1 = ''

            divideGood, dividerList, errorList = findBestDivider(means1, vars1, noises1, precise)


            #print ('done1')
            #quit()
            


            #plt.plot(errorList / (chr.shape[0] * 2))
            #plt.show()


            if True:#a >= 100:#abs(divideGood - 0.27) > 0.04:#a > 80:#a in [0, 18, 66]:#[0, 18]:
                print (divideGood)
                meansPlot = np.zeros(RDR[a].shape[0])
                bool1 = np.zeros(RDR[a].shape[0])
                #for b in range(regions.shape[0]):
                #    #print (regions[b, 0], regions[b, 1])
                #    bool1[regions[b, 0]:regions[b, 1]] = 1 + (b%2)
                #    meansPlot[regions[b, 0]:regions[b, 1]] = np.mean(RDR[regions[b, 0]:regions[b, 1]]) / divideGood

                #for b in range(8):
                #    plt.plot(np.zeros(RDR.shape[0])+b, c='grey')
                #plt.plot(bool1)
                #plt.plot(RDR)
                plt.plot(RDR[a][bins] / divideGood)
                #plt.plot(meansPlot)


                #plt.plot(BAF1)
                plt.show()

        

        dividerNums.append(divideGood)

        #quit()

        if not created1:
            divideAll = np.zeros((RDR_all.shape[0], dividerList.shape[0]))
            divideError = np.zeros((RDR_all.shape[0], errorList.shape[0]))
            created1 = True

        divideAll[a] = dividerList
        divideError[a] = errorList




    dividerNums = np.array(dividerNums)

    #quit()

    print ("done1")

    np.savez_compressed(divider_file, dividerNums)
    np.savez_compressed(error_file, divideError)

    np.savez_compressed(dividerList_file, divideAll)










def findInitialCNA(RDR_file, noise_file, BAF_file, BAF_noise_file, chr_file, divider_file, error_file, dividerList_file, initialCNA_file, initialUniqueCNA_file, initialUniqueIndex_file):


    def findCurrentCNA(RDR, noise, HAP, BAF_noise):


        
        
        int1 = np.floor(RDR)
        int1 = np.array([int1 - 1, int1, int1+1, int1+2])


        numTry = int1.shape[0]


        maxIntFull = int(np.max(int1))

        errors = np.zeros((maxIntFull+1, numTry, RDR.shape[0]))
        errors[:] = -1


        errors_BAF = np.zeros((maxIntFull+1, numTry, RDR.shape[0]))
        errors_BAF[:] = -1
        errors_RDR = np.zeros((maxIntFull+1, numTry, RDR.shape[0]))
        errors_RDR[:] = -1

        

        for b in range(numTry):

            int2 = int1[b]
            #int2 = int2.reshape((1, int2.shape[0]))

            #print (int2[:10])

            maxInt = int(np.max(int2))
            #range1 = (maxInt // 2) + 1
            #range2 = (int2 // 2) + 1
            range1 = maxInt + 1
            range2 = int2 + 1

            RDR_error = np.abs(RDR - int2)

            for a in range(range1):
                argValid = np.argwhere(range2 > a)[:, 0]


                #if 1 in argValid:
                #    print (a)

                BAF_now = (int2[argValid]-float(a)) / (int2[argValid] + 1e-04) 
                #BAF_error = np.abs(BAF_now - BAF[argValid])

                #BAF_now_adjusted = (BAF_now * 0.9) + 0.05
                #BAF_now_adjusted = (BAF_now * 0.95) + 0.025
                #BAF_now_adjusted = (BAF_now * 0.99) + 0.005
                BAF_now_adjusted = (BAF_now * 0.999) + 0.0005
                #BAF_now_adjusted = tweakBAF(BAF_now)


                HAP_now = HAP[argValid] 
                #HAP_mod_now = HAP_mod[argValid] 

                #if False:
                #    BAF_error = (np.log(BAF_now_adjusted) * HAP_mod_now[:, 1]) + (np.log(1 - BAF_now_adjusted) * HAP_mod_now[:, 0] )
                #    BAF_error = BAF_error * -1
                #else:
                BAF_measure = HAP_now[:, 1] / (np.sum(HAP_now, axis=1) + 1e-5)
                BAF_error = ((BAF_measure - BAF_now) / BAF_noise[argValid]) ** 2
                


                errors_BAF[a, b, argValid] = np.copy(BAF_error)


                #shift1 =  HAP_now[:, 1] -  HAP_now[:, 0]
                #BAF_error = BAF_error + ((shift1 ** 2) / (np.sum(HAP_now, axis=1) + 1e-5)  )
                

                errorSum = (RDR_error[argValid] / noise[argValid]) ** 2
                #errorSum = errorSum + (BAF_error ** 2)

                errors_RDR[a, b, argValid] = np.copy(errorSum)

                errorSum = errorSum + BAF_error



                #errors[a, b, argValid] = RDR_error[argValid] + (0.05 * BAF_error)

                #errors[a, b, argValid] = (RDR_error[argValid] / backgroundError1) + (BAF_error / backgroundError2)

                errors[a, b, argValid] = errorSum



        #argNoHap = np.argwhere (  np.sum(HAP, axis=1) == 0 )[:, 0]

        

        #print (errors[:,:, argNoHap[-3]]  )

        #print (errorsRDR[:,:, argNoHap[-3]]  )


        errors[errors == -1] = np.max(errors) * 2

        
        

        errors = errors.reshape(( (maxIntFull+1)*numTry, RDR.shape[0]))
        bestFit = np.argmin(errors, axis=0)



        
        #print (bestFit)
        maternalChoice = bestFit // numTry
        #print (maternalChoice)
        intChoice = bestFit % numTry

        
        #print (intChoice[argNoHap[-3]])
        #print (maternalChoice[argNoHap[-3]])

        intChoice = int1[intChoice, np.arange(intChoice.shape[0])]

        #print (intChoice[argNoHap[-3]])

        CNA = np.zeros((intChoice.shape[0], 2), dtype=int)
        CNA[:, 0] = maternalChoice
        CNA[:, 1] = intChoice - maternalChoice


        #argWeird = np.argwhere( np.abs(intChoice - RDR) > 0.9 )[:, 0]

        if False:#argWeird.shape[0] > 0:
            argWeird1 = argWeird[0]
            print (argWeird1)
            print (intChoice[argWeird1])
            print (RDR[argWeird1])
            print (noise[argWeird1])
            print (HAP[argWeird1])
            print (errors[:, argWeird1].reshape( (maxIntFull+1, numTry) )  ) 
            print (errors_BAF[:, :, argWeird1]  ) 
            print (errors_RDR[:, :, argWeird1] )  
            quit()

        #print (bestFit[argNoHap[-3]]  )

        #print (RDR[argNoHap])
        #print (intChoice[argNoHap])

        #print (errors[:, argNoHap[-3]].reshape(  (maxIntFull+1, numTry) )   )

        #ar1 = [RDR[argNoHap], intChoice[argNoHap]]
        #ar1 = np.array(ar1).T
        #print (ar1[-3])
        #quit()

        

        if np.min(CNA[:, 1]) < 0:
            print (CNA.shape)
            #    quit()
            argIssue = np.argwhere( CNA[:, 1] < 0 )[0]

            print (errors[:, argIssue[0]])
            quit()

        #print (np.min(CNA[:, 1]))
        #quit()
        assert np.min(CNA[:, 1]) >= 0

        return CNA

    

  


    def doInitialPart(dividerNums, RDR_all, noise_all, BAF_all, BAF_noise_all, chr):

        


        CNAfull = np.zeros((RDR_all.shape[0], RDR_all.shape[1], 2), dtype=int)

        for a in range(0, dividerNums.shape[0]):

            print (a, dividerNums.shape[0])

            RDR = RDR_all[a]

            #print (RDR.shape)
            #quit()

            BAF = BAF_all[a]
            #HAP_mod = HAP_mod_all[a]
            noise = noise_all[a]
            BAF_noise = BAF_noise_all[a]
            #BAF = np.min(np.array([BAF, 1-BAF]), axis=0)

            #RDR, BAF = x[a, :, 0], x[a, :, 1]
            #RDR, BAF = RDR_all[a], BAF_all[a]
            RDR = RDR / dividerNums[a]

            #CNA = findCurrentCNA(RDR, noise, BAF, HAP_mod, BAF_noise)
            CNA = findCurrentCNA(RDR, noise, BAF, BAF_noise)


            CNAfull[a] = CNA



            if False:
                argAbove = np.argwhere((CNA[:, 0] + CNA[:, 1]) > 2)[:, 0]

                if True:#argAbove.shape[0] > 50:

                    plt.plot(RDR)
                    plt.plot(BAF)
                    plt.plot(CNA[:, 0] + CNA[:, 1])
                    #plt.plot(CNA1[:, 0] + CNA1[:, 1] + 1)
                    #plt.plot(CNA[:, 0])
                    #plt.plot(CNA[:, 1])
                    plt.show()

        return CNAfull






    dividerNums = loadnpz(divider_file)
    divideError = loadnpz(error_file)
    divideAll = loadnpz(dividerList_file)

    RDR_all = loadnpz(RDR_file)
    noise = loadnpz(noise_file) + 1e-5
    if BAF_file == '':
        BAF_all = ''
        BAF_noise_all = ''
    else:
        BAF_all = loadnpz(BAF_file)[:dividerNums.shape[0]]
        BAF_noise_all = loadnpz(BAF_noise_file)[:dividerNums.shape[0]] + 1e-5
        #HAP_mod = loadnpz(HAP_mod_file)
    

    #chr = loadnpz('./data/input/chr_S' + patientNum0 + '.npz')
    chr = loadnpz(chr_file)

    _, start1 = np.unique(chr, return_index=True)
    end1 = np.concatenate((start1[1:], np.zeros(1) + chr.shape[0])).astype(int)

    

    argList = []
    dividerNums = []

    for a in range(RDR_all.shape[0]):
        errorList = divideError[a]
        errorList = errorList / (2 * chr.shape[0])

        minList = []
        arange1 = np.arange(errorList.shape[0])
        min1 = np.min(errorList)
        while np.min(errorList) < min1 + 1:
            argMin1 = np.argmin(errorList)
            errorList[np.abs(arange1 - argMin1) <= 5] = min1 + 10
            minList.append(argMin1)

            argList.append(a)
            dividerNums.append( divideAll[a][argMin1]  )


    argList = np.array(argList).astype(int)
    dividerNums = np.array(dividerNums)
    _, indexFirst, indexCounts = np.unique(argList, return_index=True, return_counts=True)

    subsetSingle = np.argwhere(indexCounts==1)[:, 0]
    


    
    if type(BAF_all) == type(''):
        CNAfull = np.round(RDR_all[argList] / dividerNums.reshape((-1, 1)) ).astype(int)
    else:
        #CNAfull = doInitialPart(dividerNums, RDR_all[argList], noise[argList], BAF_all[argList], HAP_mod[argList],  BAF_noise_all[argList], chr)
        CNAfull = doInitialPart(dividerNums, RDR_all[argList], noise[argList], BAF_all[argList],  BAF_noise_all[argList], chr)
        

    
    np.savez_compressed(initialCNA_file, CNAfull[indexFirst])

    if type(BAF_all) != type(''):
        CNAfull = CNAfull.reshape((CNAfull.shape[0], CNAfull.shape[1]*2))

    inverse1 = uniqueValMaker(CNAfull)
    _, index1 = np.unique(inverse1, return_index=True)
    CNAfull = CNAfull[index1]
    if type(BAF_all) != type(''):
        CNAfull = CNAfull.reshape((CNAfull.shape[0], CNAfull.shape[1]//2, 2))
    relevantIndex = inverse1[indexFirst]


    #plt.imshow(np.sum(CNAfull, axis=2))
    #plt.show()

    CNAfull_total = np.sum(CNAfull, axis=2)
    highNoise = np.sum(np.abs( CNAfull_total[:, 1:] - CNAfull_total[:, :-1] ), axis=1)
    #print (highNoise.shape)
    #print (CNAfull.shape)
    #print (relevantIndex.shape)
    argGood = np.argwhere(highNoise <= 500)[:, 0]
    CNAfull = CNAfull[argGood]
    #relevantIndex = relevantIndex[argGood]

    np.savez_compressed(initialUniqueCNA_file, CNAfull)
    #np.savez_compressed(initialUniqueIndex_file, relevantIndex)







RDR_file = './data/' + folder1 + '/binScale/filtered_RDR_avg.npz'
noise_file = './data/' + folder1 + '/binScale/filtered_RDR_noise.npz'
chr_file = './data/' + folder1 + '/binScale/chr_avg.npz'
divider_file = './data/' + folder1 + '/binScale/dividers.npz'
error_file = './data/' + folder1 + '/binScale/dividerError.npz'
dividerList_file = './data/' + folder1 + '/binScale/dividerAll.npz'
BAF_file = './data/' + folder1 + '/binScale/filtered_HAP_avg.npz'
BAF_noise_file = './data/' + folder1 + '/binScale/BAF_noise.npz'
initialCNA_file = './data/' + folder1 + '/binScale/initialCNA.npz'
initialUniqueCNA_file = './data/' + folder1 + '/binScale/initialUniqueCNA.npz'
initialUniqueIndex_file = './data/' + folder1 + '/binScale/initialIndex.npz'

#findInitialCNA(RDR_file, noise_file, BAF_file, BAF_noise_file, chr_file, divider_file, error_file, dividerList_file, initialCNA_file, initialUniqueCNA_file, initialUniqueIndex_file)
#quit()



def saveReformatCSV(outLoc, isNaive=False):

    if isNaive:
        pred1 = loadnpz(outLoc + '/binScale/initialCNA.npz')
    else:
        pred1 = loadnpz(outLoc + '/model/pred_now.npz')
    
    goodSubset = loadnpz(outLoc + '/initial/subset.npz')
    chr1 = loadnpz(outLoc + '/initial/chr_100k.npz')
    chrAll = loadnpz(outLoc + '/initial/allChr_100k.npz')
    bins = loadnpz(outLoc + '/binScale/bins.npz')
    cellNames = loadnpz(outLoc + '/initial/cellNames.npz')

    _, chrStarts = np.unique(chrAll, return_index=True)
    goodSubset = goodSubset - chrStarts[chr1]

    

    _, index_start = np.unique(bins, return_index=True)
    _, index_end = np.unique(bins[-1::-1], return_index=True)
    index_end = bins.shape[0] - 1 - index_end

    k100 = 100000

    posIndexing = []
    for a in range(index_start.shape[0]):
        chrome = chr1[index_start[a]] + 1
        startPos = (goodSubset[index_start[a]] * k100) + 1
        endPos = ((goodSubset[index_end[a]] + 1) * k100)

        posIndexing.append([chrome, startPos, endPos])
        #print (posIndexing[-1]) 

    dataAll = [['Cell barcode', 'Chromosome', 'Start', 'End', 'Haplotype 1', 'Haplotype 2']]

    for a in range(pred1.shape[0]):
        for b in range(pred1.shape[1]):
            dataAll.append( [ cellNames[a], posIndexing[b][0], posIndexing[b][1], posIndexing[b][2], int(pred1[a][b][0]), int(pred1[a][b][1])  ] )
    
    dataAll = np.array(dataAll)
    if isNaive:
        outFile = outLoc + '/finalPrediction/NaiveCopyPrediction.csv'
    else:
        outFile = outLoc + '/finalPrediction/DeepCopyPrediction.csv'
    np.savetxt(outFile, dataAll, delimiter=",", fmt='%s')

    #naiveAll = [['Cell barcode', 'Chromosome', 'Start', 'End', 'Haplotype 1', 'Haplotype 2']]
    #for a in range(naive1.shape[0]):
    #    for b in range(naive1.shape[1]):
    #        naiveAll.append( [ cellNames[a], posIndexing[b][0], posIndexing[b][1], posIndexing[b][2], int(naive1[a][b][0]), int(naive1[a][b][1])  ] )
    #naiveAll = np.array(naiveAll)
    #naiveFile = outLoc + '/finalPrediction/NaiveCopyPrediction.csv'
    #np.savetxt(naiveFile, naiveAll, delimiter=",", fmt='%s')
    True











def scalorRunBins(outLoc):

    
    numSteps = '9'
    stepName = 8


    stepName += 1
    stepString = str(stepName) + '/' + numSteps
    print ('Data processing — Step ' + stepString + ': Creating segements... ', end='')

    chr_file = outLoc + '/initial/chr_100k.npz'
    RDR_file = outLoc + '/initial/RDR_100k.npz'
    BAF_file = outLoc + '/initial/HAP_100k.npz'
    bins_file = outLoc + '/binScale/bins.npz'
    totalRead_file = outLoc + '/initial/totalReads.npz'
    doBAF = True
    findVariableBins(RDR_file, bins_file, chr_file, totalRead_file, doBAF, BAF_file=BAF_file)

    
    bins_file = outLoc + '/binScale/bins.npz'
    chr_file = outLoc + '/initial/chr_100k.npz'
    RDR_file = outLoc + '/initial/RDR_100k.npz'
    RDR_file2 = outLoc + '/binScale/filtered_RDR_avg.npz'
    noise_file = outLoc + '/binScale/filtered_RDR_noise.npz'
    chr_file2 = outLoc + '/binScale/chr_avg.npz'
    doBAF = True
    BAF_file = outLoc + '/initial/HAP_100k.npz'
    BAF_file2 = outLoc + '/binScale/filtered_HAP_avg.npz'
    BAF_noise_file = outLoc + '/binScale/BAF_noise.npz'
    applyVariableBins(RDR_file, bins_file, chr_file, RDR_file2, noise_file, chr_file2, doBAF, BAF_file=BAF_file, BAF_file2=BAF_file2, BAF_noise_file=BAF_noise_file)
    print ("Done")





def runNaiveCopy(outLoc, maxPloidy=10):
    


    print ('NaiveCopy — Step 1/3: Finding low variance regions... ')
    chr_file = outLoc + '/initial/chr_1M.npz'
    RDR_file = outLoc + '/initial/RDR_1M.npz'
    region_file = outLoc + '/binScale/regions.npz'
    HAP_file = outLoc + '/initial/HAP_1M.npz'
    findRegions(RDR_file, HAP_file, chr_file, region_file)
    #print ('Done')



    print ('NaiveCopy — Step 2/3: Finding cell specific scaling factors... ')
    RDR_file = outLoc + '/initial/RDR_1M.npz'
    chr_file = outLoc + '/initial/chr_1M.npz'
    region_file = outLoc + '/binScale/regions.npz'
    divider_file = outLoc + '/binScale/dividers.npz'
    error_file = outLoc + '/binScale/dividerError.npz'
    dividerList_file = outLoc + '/binScale/dividerAll.npz'
    HAP_file = outLoc + '/initial/HAP_1M.npz'
    findDividers(RDR_file, HAP_file, chr_file, divider_file, error_file, dividerList_file, region_file, maxPloidy=maxPloidy)
    #print ('Done')


    print ('NaiveCopy — Step 3/3: Estimating copy number profiles... ')
    RDR_file = outLoc + '/binScale/filtered_RDR_avg.npz'
    noise_file = outLoc + '/binScale/filtered_RDR_noise.npz'
    chr_file = outLoc + '/binScale/chr_avg.npz'
    divider_file = outLoc + '/binScale/dividers.npz'
    error_file = outLoc + '/binScale/dividerError.npz'
    dividerList_file = outLoc + '/binScale/dividerAll.npz'
    BAF_file = outLoc + '/binScale/filtered_HAP_avg.npz'
    BAF_noise_file = outLoc + '/binScale/BAF_noise.npz'
    initialCNA_file = outLoc + '/binScale/initialCNA.npz'
    initialUniqueCNA_file = outLoc + '/binScale/initialUniqueCNA.npz'
    initialUniqueIndex_file = outLoc + '/binScale/initialIndex.npz'
    #HAP_mod_file = outLoc + '/binScale/HAP_mod.npz'
    findInitialCNA(RDR_file, noise_file, BAF_file, BAF_noise_file, chr_file, divider_file, error_file, dividerList_file, initialCNA_file, initialUniqueCNA_file, initialUniqueIndex_file)
    
    

    saveReformatCSV(outLoc, isNaive=True)
    print ('Done')


def scalorRunAll(outLoc, maxPloidy=10):
    scalorRunBins(outLoc)
    runNaiveCopy(outLoc, maxPloidy=maxPloidy)


#outLoc = './data/newTN3'
#scalorRunAll(outLoc)


#saveReformatCSV(outLoc, isNaive=True)
#saveReformatCSV(outLoc, isNaive=False)
#quit()