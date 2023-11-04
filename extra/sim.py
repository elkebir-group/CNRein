#CNA.py

import numpy as np


import seaborn as sns
import matplotlib.pyplot as plt
import time
import scipy
from scipy import stats
from scipy.special import logsumexp

from scaler import uniqueProfileMaker




def rebin(data, M, doPytorch=False):

    if len(data.shape) == 1:
        M = 10
        N = data.shape[0] // M
        data = data[:(N*M)]
        data = data.reshape( (N, M) )
        if doPytorch:
            data = torch.mean(data, axis=1)
        else:
            data = np.mean(data, axis=1)
        return data

    if len(data.shape) == 2:
        M = 10
        N = data.shape[1] // M
        data = data[:, :(N*M)]
        data = data.reshape( (data.shape[0], N, M) )
        if doPytorch:
            data = torch.mean(data, axis=2)
        else:
            data = np.mean(data, axis=2)
        return data
    
    if len(data.shape) == 3:
        M = 10
        N = data.shape[1] // M
        data = data[:, :(N*M)]
        data = data.reshape( (data.shape[0], N, M, data.shape[2]) )
        if doPytorch:
            data = torch.mean(data, axis=2)
        else:
            data = np.mean(data, axis=2)
        return data



def loadnpz(name, allow_pickle=False):

    #This simple function more easily loads in compressed numpy files.

    if allow_pickle:
        data = np.load(name, allow_pickle=True)
    else:
        data = np.load(name)
    data = data.f.arr_0
    return data


def saveNoises():

    folder1 = '10x'

    RDR_file = './data/' + folder1 + '/initial/RDR_100k.npz'
    HAP_file = './data/' + folder1 + '/initial/HAP_100k.npz'
    chr_file = './data/' + folder1 + '/initial/chr_100k.npz'
    initialCNA_file = './data/' + folder1 + '/binScale/initialCNA.npz'
    bins_file = './data/' + folder1 + '/binScale/bins.npz'

    #print (loadnpz(chr_file).shape)
    #quit()

    RDR = loadnpz(RDR_file)



    if False:
        RDR_mini = RDR[:10]
        #print (np.mean(np.abs(RDR_mini) ** 2))

        noise_mini_fft = np.fft.fft(RDR_mini, axis=1) / (RDR_mini.shape[1] ** 0.5)

        #print (np.mean(np.abs(noise_mini_fft) ** 2))
        #quit()

        RDR_mini2 = np.fft.ifft(noise_mini_fft, axis=1)

        RDR_mini2 = RDR_mini2 / np.mean(np.abs(RDR_mini2))

        print (np.mean(np.abs(RDR_mini - RDR_mini2)))

        plt.plot(RDR_mini[0, :100] + 0.01)
        plt.plot(RDR_mini2[0, :100])
        plt.show()
        quit()


    HAP = loadnpz(HAP_file)
    chr = loadnpz(chr_file)
    CNAfull = loadnpz(initialCNA_file)
    bins = loadnpz(bins_file)

    CNAfull = CNAfull[:, bins]






    #_, start1 = np.unique(chr, return_index=True)
    #end1 = np.concatenate((start1[1:], np.zeros(1) + chr.shape[0])).astype(int)


    pred_RDR = np.sum(CNAfull, axis=2)
    #pred_BAF = np.min(CNAfull, axis=2) / (pred_RDR + 1e-6)
    #pred_BAF = mapBAF(pred_BAF)

    pred_RDR = pred_RDR / np.mean(pred_RDR, axis=1).reshape((-1, 1))


    #error_RDR = np.abs(pred_RDR - RDR)

    #RDR_mean = rebin(RDR, 10)



    diff1 = pred_RDR - RDR


    noise_fft = np.zeros(diff1.shape[1])

    #count1 = 0

    for a in range(22):

        args1 = np.argwhere(chr == a)[:, 0]
        #cell_noises = np.mean( diff1 ** 2, axis=1) ** 0.5
        

        noise_mini_fft = np.fft.fft(diff1[:, args1], axis=1) #/ (RDR.shape[1] ** 0.5)

        noise_mini_fft = np.mean(np.abs(noise_mini_fft) ** 2, axis=0) ** 0.5

        noise_fft[args1] = noise_mini_fft

    

    HAP = np.mean(HAP, axis=(0, 2)) * 2
    
    #plt.hist(cell_noises, bins=100)
    #plt.show()

    #noise1 = np.random.normal(size=noise_fft.shape) * noise_fft.reshape(1, -1) 
    noise1 = np.random.normal(size=noise_fft.shape) * noise_fft
            #RDR = true_RDR + (np.random.normal(size=true_RDR.shape) * error_RDR.reshape(1, -1) 
            
    noise_fft2 = np.zeros(noise_fft.shape)
    
    for a in range(22):
        args1 = np.argwhere(chr == a)[:, 0]
        noise_fft_mini = np.fft.ifft(noise1[args1], axis=0)
        noise_fft_mini = np.real(noise_fft_mini)
        noise_fft2[args1] = noise_fft_mini

    plt.plot(noise_fft2)
    plt.show()

    quit()
    


    #np.savez_compressed('./data/simulation/HAPsum_' +  folder1 + '.npz', HAP)
    #np.savez_compressed('./data/simulation/FFT_RDRerror_' +  folder1 + '.npz', noise_fft)
    

#saveNoises()
#quit()


def makeSimulation():

    #from numpy import fft

    
    folder1 = '10x'

    
    HAP_sums = loadnpz('./data/simulation/HAPsum_' +  folder1 + '.npz').astype(int)
    HAP_sums[HAP_sums < 5] = 5

    #error_RDR = loadnpz('./data/simulation/RDRerror_' +  folder1 + '.npz')

    noise_fft0 = loadnpz('./data/simulation/FFT_RDRerror_' +  folder1 + '.npz')
    #cell_noises = loadnpz('./data/simulation/cell_RDRerror_' +  folder1 + '.npz')

    
    
    chr = loadnpz('./data/' + folder1 + '/initial/chr_100k.npz')

    #TEMP!!!
    #chr = chr[0::100]



    _, start1 = np.unique(chr, return_index=True)
    end1 = np.concatenate((start1[1:], np.zeros(1) + chr.shape[0])).astype(int)
    chrSizes = end1 - start1
    chrSizes = chrSizes / np.sum(chrSizes)


    
    
    



    #cellNum = 2000
    cellNum = 1000

    #maxIter = 50
    #maxIter = 200
    #maxIter = 500
    #maxIter = 1000
    #maxIter = 2000
    maxIter = 4000


    cutOffList = [6, 8, 10, 12, 14, 16, 20, 50, 100, 1000]

    #Nsim = 1
    #for simNum in range(10, 30):
    for simNum in range(10, 30):

        print ('sim', simNum)

        fitnessModifier = int(simNum % 10) 

        cutOff = cutOffList[fitnessModifier]

        #print (fitnessModifier)

        #cutOff = np.log(5) + (np.log(200) * (fitnessModifier / 9.0) )
        #cutOff = np.exp(cutOff)

        print (cutOff)

    
        #if True:


        withWGD = False
        if simNum >= 20:
            withWGD = True


        profiles = np.ones(( maxIter,  chr.shape[0], 2 ), dtype=int) 
        
        if withWGD:
            profiles = profiles * 2 #Adding WGD!
        

        fitness = np.zeros(maxIter)

        for a in range(maxIter-1):

            

            prob1 = fitness[:a+1]
            prob1 = prob1 - np.max(prob1)
            prob1 = np.exp(prob1)
            prob1 = prob1 / np.sum(prob1)

            cloneChoice = np.random.choice(np.arange(a+1), size=1, p=prob1)[0]


            fitChange = np.random.random()
            #cutoff = 10 ** (fitnessModifier / 3.0)
            #cutoff = 3 ** (2.0 +  (fitnessModifier/ 2.0))

            


            if fitChange <= (1 / cutOff):
                fitChange = 1
            else:
                fitChange = 0
            #fitChange = np.random.choice(2) * np.log(1.5)
            #fitChange = np.random.choice(5)
            #fitChange = np.random.choice(10) * (4.0 / 3.0)
            #fitChange = np.random.choice(1000)
            #if (fitChange >= 2) and (fitChange <= 7):
            #    fitChange = -1
            #if fitChange > 1:
            #    fitChange = 0


            #fitChange = fitChange * np.log(2.8)
            fitChange = fitChange * np.log(2) #* np.log(1.01)

            #print (fitness)
            fitness[a+1] = fitness[cloneChoice] + fitChange

            #chrChoice = np.random.choice(start1.shape[0], p=chrSizes)

            chrChoice = np.random.choice(start1.shape[0], p=chrSizes)


            #pair1 = np.random.choice(2)
            isFullStart = np.random.choice(2)
            isFullEnd = np.random.choice(2)
            addDelete = (np.random.choice(2) * 2) - 1
            pairChoice = np.random.choice(2)

            startChoice = start1[chrChoice]
            endChoice = end1[chrChoice]
            size1 = endChoice - startChoice

            #print (startChoice, endChoice)

            if isFullStart == 1:
                startChoice = startChoice + np.random.choice(size1)

            if isFullEnd == 1:
                endChoice = startChoice + np.random.choice(endChoice - startChoice) + 1


            profiles[a+1] = np.copy(profiles[cloneChoice])
            boolVec = np.copy(profiles[a+1, startChoice:endChoice, pairChoice])
            boolVec[boolVec!=0] = 1
            #oldCopy = np.copy(profiles[a+1, startChoice:endChoice, pairChoice])
            profiles[a+1, startChoice:endChoice, pairChoice] = profiles[a+1, startChoice:endChoice, pairChoice] + (addDelete * boolVec)
            #argBad = np.argwhere(np.sum(profiles, axis=2) == 0)
            #profiles[argBad[:, 0], argBad[:, 1], 0] = 

            profiles[profiles<0] = 0

            

            if 0 in np.sum(profiles[a+1], axis=1):
                
                #plt.plot(np.sum(profiles[a+1], axis=1))
                #plt.show()

                fitness[a+1] = -1 * np.log(1000)

        #print ('B')


        prob1 = fitness[:a+1]
        prob1 = prob1 - np.max(prob1)
        prob1 = np.exp(prob1)
        prob1 = prob1 / np.sum(prob1)
        
        cellNums = np.random.choice(np.arange(prob1.shape[0]), size=cellNum, p=prob1)

        profiles_full = profiles[cellNums]


        inverse1 = uniqueProfileMaker(profiles_full)
        print ('number of unique profiles', np.unique(inverse1).shape)

        #sns.clustermap( np.sum(profiles_full[:, 0::10], axis=2), col_cluster=False, row_cluster=True, linewidths=0.0)
        #plt.show()
        #quit()

        
        #quit()

        #plt.imshow(np.sum( profiles_full  ))

        #For Sim 2:
        #argAddNormal = np.random.permutation(profiles_full.shape[0])[: (profiles_full.shape[0] // 4) ]
        #profiles_full[argAddNormal] = profiles_full[argAddNormal] + 1
        #print (profiles_full.shape)
        #quit()


        true_RDR = np.sum(profiles_full, axis=2).astype(float)
        true_BAF = (profiles_full[:, :, 0] + 1e-5) / (true_RDR + 2e-5)

        from scaler import haplotypePlotter
        from scipy.cluster.hierarchy import linkage

        #print ("A")
        #linkage_matrix = linkage(profiles_full.reshape((profiles_full.shape[0],  profiles_full.shape[1] * 2 ))  , method='ward', metric='euclidean')
        #print (profiles_full.shape)
        #haplotypePlotter(profiles_full.astype(int), doCluster=True, chr=[chr], withLinkage=[linkage_matrix])

        #quit()

        #plt.imshow(true_RDR, cmap='bwr')
        #plt.show()
        


        for a in range(true_RDR.shape[0]):
            true_RDR[a] = true_RDR[a] / np.mean(true_RDR[a])


        if True:

            noise1 = np.random.normal(size=true_RDR.shape) * noise_fft0.reshape(1, -1) 
            #RDR = true_RDR + (np.random.normal(size=true_RDR.shape) * error_RDR.reshape(1, -1) )
            
            noise_fft = np.zeros(true_RDR.shape)
            
            for a in range(22):
                args1 = np.argwhere(chr == a)[:, 0]
                noise_fft_mini = np.fft.ifft(noise1[:, args1], axis=1)
                noise_fft_mini = np.real(noise_fft_mini)
                noise_fft[:, args1] = noise_fft_mini


            #cell_noises2 = cell_noises[np.random.choice(cell_noises.shape[0], size=noise_fft_mini.shape[0],  replace=True )]

            #for b in range(noise_fft_mini.shape[0]):
            #    noise_fft_mini[b] = noise_fft_mini[b] / (np.mean(noise_fft_mini[b] ** 2) ** 0.5)
            #    noise_fft_mini[b] = noise_fft_mini[b] * cell_noises2[b]

            #plt.plot(noise_fft_mini[0])
            #plt.plot(true_RDR[0])
            #plt.show()
            #quit()
            RDR = true_RDR + noise_fft
            RDR[true_RDR == 0] = 0

            #RDR_mean = rebin(RDR, 10)
            #true_RDR_mean = rebin(true_RDR, 10)
            #for a in range(100):
            #    plt.plot(RDR_mean[a])# - RDR[a])
            #   plt.plot(true_RDR_mean[a])
            #    plt.show()


            #HAP = true_BAF + (np.random.normal(size=true_BAF.shape) * error_BAF.reshape(1, -1) )

            #HAP_sums

            HAP = np.zeros(profiles_full.shape)

            for b in range(true_BAF.shape[0]):

                #HAP1 = np.random.binomial(HAP_sums, true_BAF[b])
                #print (HAP1.shape)
                #quit()
                HAP[b, :, 1] = np.random.binomial(HAP_sums, true_BAF[b])
                HAP[b, :, 0] = HAP_sums - HAP[b, :, 1]
                


            BAF_new = HAP[:, :, 1] / np.sum(HAP, axis=2)

            



            RDR[RDR < 0] = 0


            #plt.plot(RDR[100])
            #plt.show()


            #Correcting noise effect on normalization
            for a in range(RDR.shape[0]):
                RDR[a] = RDR[a] / np.mean(RDR[a])

            #BAF = np.min(np.array([  BAF, 1 - BAF   ]), axis=0)

            #print (np.min(RDR))


            #quit()

            
            np.savez_compressed('./data/simulation/' + folder1 + '/profiles_unique_sim' + str(simNum) + '.npz',  profiles)
            np.savez_compressed('./data/simulation/' + folder1 + '/profiles_sim' + str(simNum) + '.npz',  profiles_full)
            np.savez_compressed('./data/simulation/' + folder1 + '/RDR_sim' + str(simNum) + '.npz',  RDR)
            np.savez_compressed('./data/simulation/' + folder1 + '/HAP_sim' + str(simNum) + '.npz',  HAP)


        



#makeSimulation()
#quit()



def rescaleSim():

    folder1 = '10x'

    for simNum in range(10, 30):

        print (simNum)

        RDR = loadnpz('./data/simulation/' + folder1 + '/RDR_sim' + str(simNum) + '.npz')
        HAP = loadnpz('./data/simulation/' + folder1 + '/HAP_sim' + str(simNum) + '.npz')
        chr = loadnpz('./data/' + folder1 + '/initial/chr_100k.npz')

        RDR_new = np.zeros(RDR.shape)
        HAP_new = np.zeros(HAP.shape)
        chr_new = np.zeros(HAP.shape[1], dtype=int)

        count1 = 0

        for a in range(22):

            RDR1 = RDR[:, chr==a]
            HAP1 = HAP[:, chr==a]

            N = RDR1.shape[1] // 10
            RDR1 = RDR1[:, :N*10]
            HAP1 = HAP1[:, :N*10]
            
            RDR1 = RDR1.reshape(( RDR1.shape[0], N, 10 ))
            HAP1 = HAP1.reshape(( HAP1.shape[0], N, 10 , 2))

            RDR1 = np.mean(RDR1, axis=2)
            HAP1 = np.sum(HAP1, axis=2)

            size1 = N

            RDR_new[:, count1:count1+size1] = RDR1
            HAP_new[:, count1:count1+size1] = HAP1
            chr_new[count1:count1+size1] = a

            count1 += size1

        RDR_new = RDR_new[:, :count1]
        HAP_new = HAP_new[:, :count1]
        chr_new = chr_new[:count1]

        #for a in range(RDR_new.shape[0]):
        #    plt.plot(RDR_new[a])
        #    plt.show()
        #quit()

        np.savez_compressed('./data/simulation/' + folder1 + '/chr_1M_sim' + str(simNum) + '.npz',  chr_new)
        np.savez_compressed('./data/simulation/' + folder1 + '/RDR_1M_sim' + str(simNum) + '.npz',  RDR_new)
        np.savez_compressed('./data/simulation/' + folder1 + '/HAP_1M_sim' + str(simNum) + '.npz',  HAP_new)

        #quit()


#rescaleSim()
#quit()

from scaler import *
from RLCNA import simpleTrain

def doPipelineSim():


    
    

    #simNum = 0
    #simNum = 1
    
    #simNum = 0
    #quit()
    for simNum in range(25, 30):
        
        timeVector = []
        timeVector.append(time.time())
        

        #folder1 = '10x'
        folder1 = '10x_2'
        folder2 = '10x'

        if True:#simNum != 20:
            
            RDR_file = './data/simulation/' + folder1 + '/RDR_sim' + str(simNum) + '.npz'
            BAF_file = './data/simulation/' + folder1 + '/HAP_sim' + str(simNum) + '.npz'
            totalRead_file = ''
            chr_file = './data/' + folder2 + '/initial/chr_100k.npz'
            doBAF = True
            bins_file = './data/simulation/' + folder1 + '/bins_sim' + str(simNum) + '.npz'
            findVariableBins(RDR_file, bins_file, chr_file, totalRead_file, doBAF, BAF_file=BAF_file)


            timeVector.append(time.time())



            RDR_file2 = './data/simulation/' + folder1 + '/RDR_avg_sim' + str(simNum) + '.npz'
            noise_file = './data/simulation/' + folder1 + '/RDR_noise_sim' + str(simNum) + '.npz'
            chr_file2 = './data/simulation/' + folder1 + '/chr_avg_sim' + str(simNum) + '.npz'
            BAF_file2 = './data/simulation/' + folder1 + '/HAP_avg_sim' + str(simNum) + '.npz'
            BAF_noise_file = './data/simulation/' + folder1 + '/BAF_sim' + str(simNum) + '.npz'


            applyVariableBins(RDR_file, bins_file, chr_file, RDR_file2, noise_file, chr_file2, doBAF, BAF_file=BAF_file, BAF_file2=BAF_file2, BAF_noise_file=BAF_noise_file)
            #quit()

            timeVector.append(time.time())


            chr_file = './data/simulation/' + folder1 + '/chr_1M_sim' + str(simNum) + '.npz'
            RDR_file = './data/simulation/' + folder1 + '/RDR_1M_sim' + str(simNum) + '.npz'
            region_file = './data/simulation/' + folder1 + '/regions_sim' + str(simNum) + '.npz'
            HAP_file = './data/simulation/' + folder1 + '/HAP_1M_sim' + str(simNum) + '.npz'
            findRegions(RDR_file, HAP_file, chr_file, region_file)
            #quit()


            #profiles_full = loadnpz('./data/simulation/' + folder1 + '/profiles_sim' + str(simNum) + '.npz')
            #RDR1 = np.sum(profiles_full, axis=2).astype(float)
            #RDR1 = RDR1 / np.mean(RDR1, axis=1).reshape((-1, 1))

            #DR = loadnpz('./data/simulation/' + folder1 + '/RDR_1M_sim' + str(simNum) + '.npz')[0]
            #RDR = rebin(RDR, 10)
            #plt.plot(RDR * np.mean(RDR1[0]))
            #plt.plot(RDR1[0][0::100])
            #plt.show()
            #quit()

            timeVector.append(time.time())

            #RDR_file2 = './data/simulation/' + folder1 + '/RDR_avg_sim' + str(simNum) + '.npz'
            #noise_file = './data/simulation/' + folder1 + '/RDR_noise_sim' + str(simNum) + '.npz'
            #BAF_file2 = './data/simulation/' + folder1 + '/HAP_avg_sim' + str(simNum) + '.npz'
            #BAF_noise_file = './data/simulation/' + folder1 + '/BAF_sim' + str(simNum) + '.npz'

            RDR_file = './data/simulation/' + folder1 + '/RDR_1M_sim' + str(simNum) + '.npz'
            region_file = './data/simulation/' + folder1 + '/regions_sim' + str(simNum) + '.npz'
            chr_file = './data/simulation/' + folder1 + '/chr_1M_sim' + str(simNum) + '.npz'
            HAP_file = './data/simulation/' + folder1 + '/HAP_1M_sim' + str(simNum) + '.npz'

            divider_file = './data/simulation/' + folder1 + '/dividers_sim' + str(simNum) + '.npz'
            error_file = './data/simulation/' + folder1 + '/dividerError_sim' + str(simNum) + '.npz'
            dividerList_file = './data/simulation/' + folder1 + '/dividerAll_sim' + str(simNum) + '.npz'
            findDividers(RDR_file, HAP_file, chr_file, divider_file, error_file, dividerList_file, region_file, precise=True, naive=False)
            #quit()



            #RDR_file2 = './data/simulation/' + folder1 + '/RDR_avg_sim' + str(simNum) + '.npz'
            #noise_file = './data/simulation/' + folder1 + '/RDR_noise_sim' + str(simNum) + '.npz'
            #BAF_file2 = './data/simulation/' + folder1 + '/HAP_avg_sim' + str(simNum) + '.npz'
            #BAF_noise_file = './data/simulation/' + folder1 + '/BAF_sim' + str(simNum) + '.npz'

            #divider_file = './data/simulation/' + folder1 + '/dividers_sim' + str(simNum) + '.npz'
            #error_file = './data/simulation/' + folder1 + '/dividerError_sim' + str(simNum) + '.npz'
            #dividerList_file = './data/simulation/' + folder1 + '/dividerAll_sim' + str(simNum) + '.npz'
            #newFindDividers(bins_file, RDR_file2, noise_file, BAF_file2, BAF_noise_file, divider_file, error_file, dividerList_file)
            #quit()

            timeVector.append(time.time())


            RDR_file = './data/simulation/' + folder1 + '/RDR_avg_sim' + str(simNum) + '.npz'
            noise_file = './data/simulation/' + folder1 + '/RDR_noise_sim' + str(simNum) + '.npz'
            chr_file = './data/simulation/' + folder1 + '/chr_avg_sim' + str(simNum) + '.npz'
            BAF_file = './data/simulation/' + folder1 + '/HAP_avg_sim' + str(simNum) + '.npz'
            BAF_noise_file = './data/simulation/' + folder1 + '/BAF_sim' + str(simNum) + '.npz'
            divider_file = './data/simulation/' + folder1 + '/dividers_sim' + str(simNum) + '.npz'
            error_file = './data/simulation/' + folder1 + '/dividerError_sim' + str(simNum) + '.npz'
            dividerList_file = './data/simulation/' + folder1 + '/dividerAll_sim' + str(simNum) + '.npz'
            initialCNA_file = './data/simulation/' + folder1 + '/initialCNA_sim' + str(simNum) + '.npz'
            initialUniqueCNA_file = './data/simulation/' + folder1 + '/initialUniqueCNA_sim' + str(simNum) + '.npz'
            initialUniqueIndex_file = './data/simulation/' + folder1 + '/initialIndex_sim' + str(simNum) + '.npz'
            findInitialCNA(RDR_file, noise_file, BAF_file, BAF_noise_file, chr_file, divider_file, error_file, dividerList_file, initialCNA_file, initialUniqueCNA_file, initialUniqueIndex_file)
            #quit()

            timeVector.append(time.time())

            #quit()




        RDR_file = './data/simulation/' + folder1 + '/RDR_avg_sim' + str(simNum) + '.npz'
        noise_file = './data/simulation/' + folder1 + '/RDR_noise_sim' + str(simNum) + '.npz'
        chr_file = './data/simulation/' + folder1 + '/chr_avg_sim' + str(simNum) + '.npz'
        HAP_file = './data/simulation/' + folder1 + '/HAP_avg_sim' + str(simNum) + '.npz'
        BAF_noise_file = './data/simulation/' + folder1 + '/BAF_sim' + str(simNum) + '.npz'
        initialCNA_file = './data/simulation/' + folder1 + '/initialCNA_sim' + str(simNum) + '.npz'
        initialUniqueCNA_file = './data/simulation/' + folder1 + '/initialUniqueCNA_sim' + str(simNum) + '.npz'

        modelName =  './data/simulation/' + folder1 + '/model_sim' + str(simNum) + '_new.pt'
        predict_file = './data/simulation/' + folder1 + '/pred_sim' + str(simNum) + '_new.npz'
        originalError_file = ''

        Ncall = 20
        withAdjust = True
        balance = 1.0
        simpleTrain(RDR_file, HAP_file, chr_file, initialCNA_file, initialUniqueCNA_file, originalError_file, modelName, predict_file, Ncall, noise_file, BAF_noise_file, balance, withAdjust, stopIter=True)

        timeVector.append(time.time())

        timeVector = np.array(timeVector)
        np.savez_compressed('./data/simulation/' + folder1 + '/timeVector_' + str(simNum) + '.npz', timeVector)

#doPipelineSim()
quit()


from scaler import haplotypePlotter

simNum = 2
folder1 = '10x'
predict_file = './data/simulation/' + folder1 + '/pred_sim' + str(simNum) + '.npz'

profiles_full = loadnpz('./data/simulation/' + folder1 + '/profiles_sim' + str(simNum) + '.npz')

bins = loadnpz('./data/simulation/' + folder1 + '/bins_sim' + str(simNum) + '.npz')

_, index1 = np.unique(bins, return_index=True)

pred1 = loadnpz(predict_file)
naive1 = loadnpz('./data/simulation/' + folder1 + '/initialCNA_sim' + str(simNum) + '.npz')

#RDR = loadnpz('./data/simulation/' + folder1 + '/RDR_sim' + str(simNum) + '.npz')
#RDR = loadnpz('./data/simulation/' + folder1 + '/RDR_avg_sim' + str(simNum) + '.npz')



pred1 = pred1[:, bins]
#naive1 = naive1[:, bins]
#profiles_full = profiles_full[:, index1]

print (pred1.shape)
print (profiles_full.shape)

naive_ploidy = np.mean(naive1, axis=(1, 2)) * 2


args1 = np.argwhere(naive_ploidy > 4)[:, 0]



if False:
    #plt.hist( naive_ploidy, bins=100 )
    #plt.show()


    for a in args1:
        naive_now = naive1[a]
        naive_now = np.sum(naive_now, axis=1)
        naive_now =naive_now / np.mean(naive_now)

        


        plt.plot(RDR[a])
        plt.plot(naive_now)
        plt.show()
    quit()


#profiles_full = np.sort(profiles_full, axis=2)
#pred1 = np.sort(pred1, axis=2)
profiles_full = profiles_full[:, :, -1::-1]
#pred1 = pred1[:, :, -1::-1]


error1 = np.sum(np.abs(profiles_full - pred1), axis=2)
error1[error1!=0] = 1
print (np.mean(error1.astype(float)))


inverse1 = uniqueProfileMaker(profiles_full)

_, counts1 = np.unique(inverse1, return_counts=True)

counts1 = np.sort(counts1)[-1::-1]

plt.plot(counts1)
plt.show()





quit()

#diff = np.sum(pred1, axis=2) - np.sum(profiles_full, axis=2)
#diff = np.sort(pred1, axis=2) - np.sort(profiles_full, axis=2)

#diff = pred1 - profiles_full
#diff = naive1[:, :, -1::-1] - profiles_full[:, :, -1::-1]


profiles_full = profiles_full[:, 0::10]
pred1 = pred1[:, 0::10]


from scipy.cluster.hierarchy import linkage
linkage_matrix = linkage(profiles_full.reshape((profiles_full.shape[0],  profiles_full.shape[1] * 2 ))  , method='ward', metric='euclidean')


#haplotypePlotter(profiles_full.astype(int), doCluster=True, chr=[], withLinkage=[linkage_matrix], saveFile='./images/temp1.png')
#haplotypePlotter(pred1.astype(int), doCluster=True, chr=[], withLinkage=[linkage_matrix], saveFile='./images/temp2.png')

#plt.imshow(np.sum(profiles_full, axis=2))
#plt.show()#
#quit()
#print (np.mean(np.abs(diff)))




