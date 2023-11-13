
#CNA.py


import pandas as pd
import numpy as np

import statsmodels.api as sm
import time
import pysam
import os



def loadnpz(name, allow_pickle=False):

    #This simple function more easily loads in compressed numpy files.

    if allow_pickle:
        data = np.load(name, allow_pickle=True)
    else:
        data = np.load(name)
    data = data.f.arr_0
    return data



#ar = np.arange(10000000)
#np.savez_compressed('./testArray.npz', ar)
#ar = loadnpz('./testArray.npz')
#quit()




def dictInverter(names):

    dict = {}
    count1 = 0
    inverse1 = np.zeros(len(names), dtype=int)
    names_unique = []
    for a in range(len(names)):
        name = names[a]
        if name in dict:
            count2 = dict[name]
            inverse1[a] = count2
        else:
            names_unique.append(name)
            dict[name] = count1
            count1 += 1

    names_unique = np.array(names_unique)

    return inverse1, names_unique


def averageMap():


    #data = pd.read_csv("./data/DLP/initial/k100.umap.bedgraph", sep='\t', low_memory=False)

    #print (data)

    #['chr1', '180190517', '180271567', '1.0']

    file1 = open("./data/DLP/initial/k100.umap.bedgraph", 'r')
    Lines = file1.readlines()

    N = 1000

    #data = np.zeros((2 * len(Lines) // N, 3 ))
    vals = np.zeros(len(Lines))
    chr = np.zeros(vals.shape[0], dtype=int)

    #print (len(Lines))
    #quit()

    chrCount = 0
    chrLast = ''

    posRound2 = 0

    count1 = 0
    for line in Lines:

        if count1 != 0:
            line = line.replace('\n', '')
            list1 = line.split('\t')

            posRound2_last = posRound2

            pos = int(list1[1])
            posRound = pos // N

            pos2 = int(list1[2])
            posRound2 = (pos2 - 1) // N
            #weight1 = (pos2 - pos1)

            value = float(list1[3])

            chr1 = int(list1[0][3:])

            if chr1 == 2:
                print (list1)

            #print (value, posRound, chr1)


            if list1[0] != chrLast:
                chrLast = list1[0]
                if count1 != 1:
                    print ('update')
                    print (chrCount, posRound2_last )
                    chrCount = chrCount+(posRound2_last + 1)


                    print (chrCount)

            #data[chrCount+posRound]

            pos1 = pos
            weight1 = (pos2 - pos1)

            #if posRound2 >= 249240:
            #    print (list1)
            #    print (posRound2)


            #print (chr)
            if not chr[chrCount+posRound] in [0, chr1]:
                print (list1)
                print ('chrCount', chrCount, chrCount+posRound)
                print (chr[chrCount+posRound], chr1)
                quit()

            if weight1 == 1:
                vals[chrCount+posRound] += (value / float(N))


                chr[chrCount+posRound] = chr1


            else:

                #print (list1)

                assert value == 1.0

                if posRound == posRound2:
                    vals[chrCount+posRound] += ((value / float(N)) * weight1)
                    chr[chrCount+posRound] = chr1

                else:

                    weight2 = (((pos1 // N) + 1) * N) - pos1
                    vals[chrCount+posRound] += ((value / float(N)) * weight2)
                    chr[chrCount+posRound] = chr1

                    weight3 = pos2 - ((pos2 // N) * N)
                    vals[chrCount+posRound2] += ((value / float(N)) * weight3)
                    chr[chrCount+posRound2] = chr1

                    if (posRound2 - posRound) >= 2:
                        vals[chrCount+posRound+1:chrCount+posRound2] = value
                        chr[chrCount+posRound+1:chrCount+posRound2] = chr1



            #if count1 == 10:
            #    quit()

        count1 += 1

    last1 = 1
    for a0 in range(chr.shape[0]):
        a = chr.shape[0] - a0 - 1
        if chr[a] == 0:
            chr[a] = last1
        else:
            last1 = chr[a]


    vals = vals[:chrCount+posRound2]
    chr = chr[:chrCount+posRound2]

    info = np.array([chr, vals]).T

    np.savez_compressed('./data/DLP/initial/originalMappability.npz', info)

#averageMap()
#quit()



def patientDoSmallBinning(read_folder, cell_folder, patientFile, hist_file, chr_file, uniqueCell_file):



    M = 10000

    dict = {}

    for a0 in range(22):
        a1 = a0 + 1
        read_folder2 = read_folder + '/' + str(a1)
        cell_folder2 = cell_folder + '/' + str(a1)

        print (a1)

        read_file = read_folder2 + '/' + patientFile
        cell_file = cell_folder2 + '/' + patientFile

        cell_names = loadnpz(cell_file)
        read_pos = loadnpz(read_file)[:, 0]

        print ("A")

        inverse1, names_unique = dictInverter(cell_names)

        print (read_pos.shape)
        print (inverse1.shape)

        Ncell = names_unique.shape[0]

        np.savez_compressed(uniqueCell_file, names_unique)

        max1 = np.max(read_pos)

        max2 = (max1 // M) #+ 1 #Remove the last one to avoid partial bins
        #histChr = np.zeros((Ncell, Nbin), dtype=int)

        read_pos = read_pos // M

        print (read_pos.shape)
        print (inverse1.shape)

        inverse1 = inverse1[read_pos < max2]
        read_pos = read_pos[read_pos < max2]

        #vals = read_pos + (max2 * inverse1)

        #print ("B")

        #Nbin = Ncell * max2
        #histChr, bins = np.histogram(vals[:vals.shape[0] // 20], bins=np.arange(Nbin+1)-0.5)
        #histChr = histChr.reshape((Ncell, max2))

        histChr = np.zeros((Ncell, max2), dtype=int)

        for b in range(read_pos.shape[0]):
            histChr[inverse1[b], read_pos[b]] += 1
            if b % 10000 == 0:
                print (b//1000000, read_pos.shape[0] // 1000000)


        #print (histChr.shape)

        #plt.imshow(histChr)
        #plt.show()
        #quit()

        chrNow = np.zeros(max2, dtype=int) + a0

        if a0 == 0:
            chrAll = chrNow
            histAll = histChr
        else:
            chrAll = np.concatenate((chrAll, chrNow), axis=0)
            histAll = np.concatenate((histAll, histChr), axis=1)


    np.savez_compressed(hist_file, histAll)
    np.savez_compressed(chr_file, chrAll)




#folder1 = 'ACT/P2'
#read_folder = './AWS/data/ACT/processed/pos'
#cell_folder = './AWS/data/ACT/processed/cell'
#patientFile = 'ACT.patient1_merged.rg.bam.npz'
#patientFile = 'ACT.patient2_merged.rg.bam.npz'
#hist_file = './data/' + folder1 + '/initial/allHistBam_10k.npz'
#chr_file = './data/' + folder1 + '/initial/allChr_10k.npz'
#uniqueCell_file = './data/' + folder1 + '/initial/cellNames.npz'
#patientDoSmallBinning(read_folder, cell_folder, patientFile, hist_file, chr_file, uniqueCell_file)
#quit()




def cellDoSmallBinning(read_folder, hist_file, chr_file, uniqueCell_file):



    #M = 10000
    M = 100000

    dict = {}

    for a0 in range(22):
        a1 = a0 + 1
        read_folder2 = read_folder + '/' + str(a1)

        #print (a1)

        if a0 == 0:
            fnames = os.listdir(read_folder2)

            Ncell = len(fnames)
            for b in range(len(fnames)):
                dict[fnames[b]] = b


            cellNames = []
            for b in range(len(fnames)):
                name = fnames[b]
                name = name.replace('.npz', '')
                name = name.replace('.bam', '')
                cellNames.append(name)
            cellNames = np.array(cellNames)

            np.savez_compressed(uniqueCell_file, cellNames)

        max1 = 0
        for b in range(len(fnames)):
            fname2 = read_folder2 + '/' + fnames[b]

            data = loadnpz(fname2)
            if np.max(data) > max1:
                max1 = np.max(data)



        Nbin = (max1 // M) #+ 1 #Remove the last one to avoid partial bins
        histChr = np.zeros((Ncell, Nbin), dtype=int)

        #print (histChr.shape)
        #quit()

        for b in range(len(fnames)):
            fname2 = read_folder2 + '/' + fnames[b]

            data = loadnpz(fname2) // M
            if len(data.shape) == 2:
                data = data[:, 0] 
            #print (data.shape)
            #quit()

            #print (b)


            #time1 = time.time()
            hist1, bins = np.histogram(data, bins=np.arange(Nbin+1)-0.5)

            #print (histChr.shape)
            #print (Nbin)
            #print (hist1.shape)


            histChr[b] = hist1

            #plt.plot(ar)
            #plt.show()
            #quit()


            #print (Nbin)
            #quit()

            #for c in range(data.shape[0]):
            #    histChr[b, data[c] // 1000] += 1

            #print (time.time() - time1)
            #quit()

        chrNow = np.zeros(Nbin, dtype=int) + a0

        if a0 == 0:
            chrAll = chrNow
            histAll = histChr
        else:
            chrAll = np.concatenate((chrAll, chrNow), axis=0)
            histAll = np.concatenate((histAll, histChr), axis=1)


    np.savez_compressed(hist_file, histAll)
    np.savez_compressed(chr_file, chrAll)





#folder1 = '10x'
#folder1 = 'DLP'
folder1 = 'TN3'
#folder1 = 'ACT10x'
#read_folder = './AWS/data/' + folder1 + '/processed/pos'
#name_folder = './AWS/data/' + folder1 + '/processed/cell'
read_folder = './data/' + folder1 + '/readCounts/pos'
name_folder = './data/' + folder1 + '/readCounts/cell'
hist_file = './data/' + folder1 + '/initial/allHistBam_100k.npz' #originaly 10k
chr_file = './data/' + folder1 + '/initial/allChr_100k.npz' #originaly 10k
uniqueCell_file = './data/' + folder1 + '/initial/cellNames.npz'
#cellDoSmallBinning(read_folder, hist_file, chr_file, uniqueCell_file)
#quit()





def doHapSmallBinning(name_file, uniqueCell_file, nameOld_folder, hap_folder, chr_file, rawHAP_file):


    fnames = loadnpz(uniqueCell_file)


    #cellNames = np.loadtxt(nameOld_folder, dtype=str)
    #for a in range(cellNames.shape[0]):
    #    name = cellNames[a].split('/')[-1]
    #    name = name.replace('.bam', '')
    #    cellNames[a] = name

    cellNames = loadnpz(nameOld_folder)
    #print (cellNames)
    #print (fnames)

    #print (np.intersect1d(cellNames, fnames).shape)
    #quit()
    
    perm1 = np.zeros(len(fnames), dtype=int)
    for a in range(len(fnames)):
        arg1 = np.argwhere(cellNames == fnames[a])[0, 0]
        perm1[a] = arg1
    



    #M = 10000
    M = 100000


    

    chrAll = loadnpz(chr_file)
    histAll = np.zeros((len(fnames), chrAll.shape[0], 2), dtype=int)


    _, index = np.unique(chrAll, return_index=True)


    #chrA = 4
    chrList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    chrList = np.array(chrList) - 1

    for chrA in chrList:
        
        
        chrNum = str(chrA + 1)

        #print (chrNum)

        hap_file = hap_folder + 'chr_' + chrNum + '.npz'
        position_file = hap_folder + 'positions_chr' + chrNum + '.npz'

        hapCounts = loadnpz(hap_file)
        positions = loadnpz(position_file)


        indexStart = index[chrA]

        #print (np.max(positions[:, 0] // M) + indexStart)

        excludeSum = 0

        for b in range(positions.shape[0]):
            pos1 = positions[b, 0] // M

            if chrAll[indexStart] == chrAll[(pos1+indexStart)% chrAll.shape[0] ]:
                #print (histAll[pos1 + indexStart].shape)
                #print (hapCounts[b].shape)
                #quit()
                #print (pos1 + indexStart)
                histAll[:, pos1 + indexStart] = histAll[:, pos1 + indexStart] + hapCounts[b]
            else:
                excludeSum += 1
        
        #print (positions.shape)
        #print ('excluded', excludeSum)



    

    histAll = histAll[perm1]

    np.savez_compressed(rawHAP_file, histAll)
    





#folder1 = '10x'
#folder1 = 'ACT10x'
#folder1 = 'TN3'
#folder1 = 'DLP'

read_folder = './data/' + folder1 + '/readCounts/pos'
#nameOld_folder = './data/' + folder1 + '/phased/DLPbams_890.txt'
#nameOld_folder = './data/' + folder1 + '/phased/bamsList.txt'
nameOld_folder = './data/' + folder1 + '/info/bamAll.txt'
name_file = './data/' + folder1 + '/initial/cellNames.npz'

hap_folder = './data/' + folder1 + '/phasedCounts/' #positions_chr' + a1 + '.npz' 
chr_file = './data/' + folder1 + '/initial/allChr_100k.npz' #used to be 10k
rawHAP_file = './data/' + folder1 + '/initial/allRawHAP_100k.npz' #used to be 10k
uniqueCell_file = './data/' + folder1 + '/initial/cellNames.npz'

#doHapSmallBinning(name_file, uniqueCell_file, nameOld_folder, hap_folder, chr_file, rawHAP_file)
#quit()







def findGCadjustment(refGenome, refLoc, hist_file, adjustment_file, chr_file, lowHapDoImbalance, rawHAP_file, hapHist_file,  chr_file2, goodSubset_file, RDR_file, totalRead_file):



    M = 100000

    chr = loadnpz(chr_file)
    #chr_diff = np.argwhere((chr[1:] - chr[:-1]) != 0)[:, 0]
    #chr_diff = chr_diff + 1
    #chr_diff = np.concatenate((chr_diff))


    #print (chr.shape)




    data = loadnpz(hist_file)


    totalRead = np.sum(data, axis=1)
    np.savez_compressed(totalRead_file, totalRead)
    #quit()
    




    sum1 = np.sum(data, axis=0)


    #sum2 = np.sum(data, axis=1)
    #plt.plot(sum1)
    #plt.show()
    #quit()


    #plt.plot(sum1[chr == 1])
    #plt.show()
    #quit()




    sum1_sort = np.sort(sum1)
    removeTop = sum1_sort[:-sum1_sort.shape[0] // 100]
    mean1 = np.mean(removeTop)
    std1 = np.mean( (removeTop - mean1) ** 2 ) ** 0.5
    cutoff = mean1 + (std1 * 3.0)#(std1 * 10.0)

    argGood = np.argwhere(sum1 < cutoff)[:, 0]
    #argGood = argGood[sum1[argGood] > 10]
    argGood = argGood[sum1[argGood] > 100] #since 100k not 10k


    #plt.plot(sum1[argGood])
    #plt.show()
    #quit()

    if refGenome == 'hg38':
        #df_gc = pd.read_csv("./data/gc/hg38_1.gc.bed", sep='\t', low_memory=False)
        df_gc = pd.read_csv(refLoc + "/hg38_1.gc.bed", sep='\t', low_memory=False)
        
    else:
        #df_gc = pd.read_csv("./data/gc/b37_1.gc.bed", sep='\t', low_memory=False)
        df_gc = pd.read_csv(refLoc + "/b37_1.gc.bed", sep='\t', low_memory=False)


    
    chr_name = df_gc['#1_usercol'].to_numpy()
    gc_num_initial = df_gc['5_pct_gc'].to_numpy()
    pos_end = df_gc['3_usercol'].to_numpy()

    #print (chr_name)
    #quit()

    #print (gc_num_initial.shape)
    #print (data.shape)
    #quit()

    #if True:
    #    gc_num_initial = gc_num_initial[:(gc_num_initial.shape[0]//10)*10]
    #    gc_num_initial = gc_num_initial.reshape((gc_num_initial.shape[0]//10, 10))
    #    gc_num_initial = np.mean(gc_num_initial, axis=1)
    #    chr_name = chr_name[0::10][:gc_num_initial.shape[0]]
    #    pos_end = pos_end[0::10][:gc_num_initial.shape[0]]



    gc_num = np.zeros(data.shape[1])
    #map_num = np.zeros(data.shape[1])

    #dataMap = loadnpz('./data/DLP/initial/originalMappability.npz')


    #print (gc_num.shape)
    #print (chr.shape)
    #print (data.shape)
    #quit()
    #print (np.unique(chr, return_counts=True))
    #print (np.unique(dataMap[:, 0], return_counts=True))

    #for a in range(22):
    #    a1 = str(a+1)
    #    gc_num_chr = gc_num_initial[chr_name==a1]
    #    print (a1, gc_num_chr.shape[0] // 1000)
    #quit()

    M2 = M // 1000

    for a in range(22):
        a1 = str(a+1)
        if refGenome == 'hg38':
            a1 = 'chr' + a1

        args1 = np.argwhere(chr == int(a))[:, 0]

        #print (args1.shape)
        #quit()
        gc_num_chr = gc_num_initial[chr_name==a1]
        
        #print (gc_num_chr.shape)
        #quit()

        #print (args1.shape, gc_num_chr.shape)

        gc_num_chr = gc_num_chr[:M2*(gc_num_chr.shape[0] // M2)]
        gc_num_chr = gc_num_chr.reshape((gc_num_chr.shape[0] // M2, M2))
        gc_num_chr = np.mean(gc_num_chr, axis=1)


        #print (gc_num_chr.shape, args1.shape)
        #print (args1.shape)
        #quit()


        #map_chr = dataMap[dataMap[:, 0].astype(int) == int(a1), 1]
        #print (map_chr.shape)

        if True:
            gc_num[args1] = np.copy(gc_num_chr[:args1.shape[0]])
        #map_num[args1] = np.copy(map_chr[:args1.shape[0]])

    



    gc_num = gc_num[argGood]
    #map_num = map_num[argGood]
    data = data[:, argGood]
    sum1 = sum1[argGood]
    chr = chr[argGood]



    argGood2 = np.argwhere(gc_num > 0.01)[:, 0]
    gc_num = gc_num[argGood2]
    data = data[:, argGood2]
    sum1 = sum1[argGood2]
    chr = chr[argGood2]
    #map_num = map_num[argGood2]


    np.savez_compressed(goodSubset_file, argGood[argGood2])
    np.savez_compressed(chr_file2, chr)


    def adjust_lowess(x, y, f=.5):
        jlow = sm.nonparametric.lowess(np.log(y), x, frac=f)
        jz = np.interp(x, jlow[:,0], jlow[:,1])
        return np.log(y)-jz, jz


    

    # _, dist_gc = adjust_lowess(gc_num, sum1+1, f=0.05)# +0.01)
    _, dist_gc = adjust_lowess(gc_num, sum1+1, f=0.2)# +0.01) #Updates 0.05 to 0.1 due to 100k

    if False:
        import matplotlib.pyplot as plt
        plt.scatter(gc_num[0::100], np.log(sum1+1)[0::100])
        plt.scatter(gc_num[0::100], dist_gc[0::100])
        #plt.show()
        plt.savefig('./images/temp.png')




    dist_gc = np.exp(dist_gc) + 1

    np.savez_compressed(adjustment_file, dist_gc)


    data = data / dist_gc.reshape((1, -1))
    mean1 = np.mean(data, axis=1)
    data = data / mean1.reshape((-1, 1))

    np.savez_compressed(RDR_file, data)


    hapAll = loadnpz(rawHAP_file)

    hapAll = hapAll[:, argGood[argGood2]]

    if lowHapDoImbalance:
        HAPsum = np.mean(hapAll.astype(float), axis=(0, 2)) * 2
        HAPsum_log = np.log(HAPsum + 1e-8)

        hapAll[:, HAPsum_log < 0, 0] = hapAll[:, HAPsum_log < 0, 0] + 5


    np.savez_compressed(hapHist_file, hapAll)







#folder1 = '10x'
#folder1 = 'ACT10x'
#folder1 = 'TN3'
folder1 = 'DLP'

hist_file = './data/' + folder1 + '/initial/allHistBam_100k.npz' #used to be 10k
rawHAP_file = './data/' + folder1 + '/initial/allRawHAP_100k.npz' 
chr_file = './data/' + folder1 + '/initial/allChr_100k.npz' #used to be 10k
adjustment_file = './data/' + folder1 + '/initial/gc_adjustment.npz'
goodSubset_file = './data/' + folder1 + '/initial/subset.npz'
RDR_file = './data/' + folder1 + '/initial/RDR_100k.npz'
chr_file2 = './data/' + folder1 + '/initial/chr_100k.npz'
hapHist_file = './data/' + folder1 + '/initial/HAP_100k.npz'
totalRead_file = './data/' + folder1 + '/initial/totalReads.npz'

useHG38 = True
if folder1 == 'DLP':
    useHG38 = False

lowHapDoImbalance = False
if folder1 == 'DLP':
    lowHapDoImbalance = True

#findGCadjustment(useHG38, hist_file, adjustment_file, chr_file, lowHapDoImbalance, rawHAP_file, hapHist_file,  chr_file2, goodSubset_file, RDR_file, totalRead_file)
#quit()





def saveRDR(RDR_file, adjustment_file, chr_file, goodSubset_file, chr_file_2, RDR_file_2, cellGood_file, doBAF, hapHist_file='', BAF_file_2=''):

    data = loadnpz(RDR_file)
    if doBAF:
        hapData = loadnpz(hapHist_file)
    chr = loadnpz(chr_file)
    adjustment = loadnpz(adjustment_file)
    argGood = loadnpz(goodSubset_file)

    
    N = 10

    #chr = chr[argGood]
    #data = data[:, argGood] #Already handled in RDR
    #if doBAF:
    #    hapData = hapData[:, argGood]

    

    
    if False:
        totalReads = np.sum(data, axis=1)
        cellGood = np.argwhere(totalReads > 20000)[:, 0]
        np.savez_compressed(totalRead_file, totalReads[cellGood])
        np.savez_compressed(cellGood_file, cellGood)

        data = data[cellGood]
        if doBAF:
            hapData = hapData[cellGood]
        

    
        data = data / adjustment.reshape((1, -1))
        data = data / np.mean(data, axis=1).reshape((-1, 1))
        adjustment_new = np.zeros(2 * data.shape[1] // N )

    RDR_new = np.zeros((data.shape[0],  2 * data.shape[1] // N ))
    chr_new = np.zeros(RDR_new.shape[1], dtype=int)

    #RDR2 = np.zeros((data.shape[0],  2 * data.shape[1] // N2 ))
    #chr_new2 = np.zeros(RDR.shape[1], dtype=int)

    if doBAF:
        #BAF = np.zeros((data.shape[0],  2 * data.shape[1] // N, 2 ))
        #BAF2 = np.zeros((data.shape[0],  2 * data.shape[1] // N2, 2 ))
        BAF_new = np.zeros((data.shape[0],  2 * data.shape[1] // N, 2 ))

    count1 = 0
    chr_unique = np.unique(chr)
    for a in range(chr_unique.shape[0]):
        #print (a)
        args1 = np.argwhere(chr == chr_unique[a])[:, 0]

        args1 = args1[:N * (args1.shape[0] // N)]
        #args2 = args1[:N2 * (args1.shape[0] // N2)]

        RDR_chr = data[:, args1]
        RDR_chr = RDR_chr.reshape((RDR_chr.shape[0], RDR_chr.shape[1] // N, N ))
        RDR_chr = np.mean(RDR_chr, axis=2)

        #RDR_chr2 = data[:, args2]
        #RDR_chr2 = RDR_chr2.reshape((RDR_chr2.shape[0], RDR_chr2.shape[1] // N2, N2 ))
        #RDR_chr2 = np.mean(RDR_chr2, axis=2)

        #adjustment_chr = adjustment[args1]
        #adjustment_chr = adjustment_chr.reshape(( adjustment_chr.shape[0] // N, N ))
        #adjustment_chr = np.mean(adjustment_chr, axis=1)

        size1 = RDR_chr.shape[1]
        #size2 = RDR_chr2.shape[1]

        RDR_new[:, count1:count1+size1] = np.copy(RDR_chr)
        chr_new[count1:count1+size1] = chr_unique[a]


        #RDR2[:, count2:count2+size2] = np.copy(RDR_chr2)
        #chr_new2[count2:count2+size2] = chr_unique[a]

        #adjustment_new[count1:count1+size1] = np.copy(adjustment_chr)


        if doBAF:

            BAF_chr = hapData[:, args1]
            BAF_chr = BAF_chr.reshape((BAF_chr.shape[0], BAF_chr.shape[1] // N, N, 2 ))
            BAF_chr = np.sum(BAF_chr, axis=2)#.astype(float)

            
            #quit()

            #BAF_chr2 = hapData[:, args2]
            #BAF_chr2 = BAF_chr2.reshape((BAF_chr2.shape[0], BAF_chr2.shape[1] // N2, N2, 2 ))
            #BAF_chr2 = np.sum(BAF_chr2, axis=2).astype(float)




            

            BAF_new[:, count1:count1+size1] = np.copy(BAF_chr)
            #BAF2[:, count2:count2+size2] = np.copy(BAF_chr2)
        

        

        
        count1 +=  size1
        #count2 +=  size2
    

    #print (count1)

    RDR_new = RDR_new[:, :count1]
    chr_new = chr_new[:count1]

    #RDR2 = RDR2[:, :count2]
    #chr_new2 = chr_new2[:count2]

    #adjustment_new = adjustment_new[:count1]

    #print (RDR_new.shape)
    #print (chr_new.shape)

    #print (RDR2.shape)
    #print (chr_new2.shape)


    np.savez_compressed(chr_file_2, chr_new)
    np.savez_compressed(RDR_file_2, RDR_new)

    #np.savez_compressed(chr_file_2, chr_new2)
    #np.savez_compressed(RDR_file_2, RDR2)

    if doBAF:

        BAF_new = BAF_new[:, :count1]
        #BAF2 = BAF2[:, :count2]

        np.savez_compressed(BAF_file_2, BAF_new)
        #np.savez_compressed(BAF_file_2, BAF2)










#folder1 = 'DLP'
#folder1 = '10x'
#folder1 = 'ACT10x'
#folder1 = 'TN3'
hapHist_file = './data/' + folder1 + '/initial/HAP_100k.npz' #used to be 10k
RDR_file = './data/' + folder1 + '/initial/RDR_100k.npz' #used to be 10k
chr_file = './data/' + folder1 + '/initial/chr_100k.npz'
adjustment_file = './data/' + folder1 + '/initial/gc_adjustment.npz'
goodSubset_file = './data/' + folder1 + '/initial/subset.npz'
chr_file_2 = './data/' + folder1 + '/initial/chr_1M.npz'
RDR_file_2 = './data/' + folder1 + '/initial/RDR_1M.npz'
BAF_file_2 = './data/' + folder1 + '/initial/HAP_1M.npz'
cellGood_file = './data/' + folder1 + '/initial/cellGood.npz'
doBAF = True
#saveRDR(RDR_file, adjustment_file, chr_file, goodSubset_file, chr_file_2, RDR_file_2, cellGood_file, doBAF, hapHist_file=hapHist_file, BAF_file_2=BAF_file_2)
#quit()



def runProcessFull(outLoc, refLoc, refGenome):

    numSteps = '9'
    stepName = 6


    stepName += 1
    stepString = str(stepName) + '/' + numSteps
    print ('Data processing — Step ' + stepString + ': Creating bins... ', end='')
    read_folder = outLoc + '/readCounts/pos'
    name_folder = outLoc + '/readCounts/cell'
    hist_file = outLoc + '/initial/allHistBam_100k.npz' #originaly 10k
    chr_file = outLoc + '/initial/allChr_100k.npz' #originaly 10k
    uniqueCell_file = outLoc + '/initial/cellNames.npz'
    cellDoSmallBinning(read_folder, hist_file, chr_file, uniqueCell_file)




    read_folder = outLoc + '/readCounts/pos'
    #nameOld_folder = outLoc + '/info/bamAll.txt'
    nameOld_folder = outLoc + '/phasedCounts/barcodes_chr1.npz'
    name_file = outLoc + '/initial/cellNames.npz'
    hap_folder = outLoc + '/phasedCounts/'
    chr_file = outLoc + '/initial/allChr_100k.npz'
    rawHAP_file = outLoc + '/initial/allRawHAP_100k.npz'
    uniqueCell_file = outLoc + '/initial/cellNames.npz'
    doHapSmallBinning(name_file, uniqueCell_file, nameOld_folder, hap_folder, chr_file, rawHAP_file)
    print ('Done')



    hist_file = outLoc + '/initial/allHistBam_100k.npz' #used to be 10k
    rawHAP_file = outLoc + '/initial/allRawHAP_100k.npz' 
    chr_file = outLoc + '/initial/allChr_100k.npz' #used to be 10k
    adjustment_file = outLoc + '/initial/gc_adjustment.npz'
    goodSubset_file = outLoc + '/initial/subset.npz'
    RDR_file = outLoc + '/initial/RDR_100k.npz'
    chr_file2 = outLoc + '/initial/chr_100k.npz'
    hapHist_file = outLoc + '/initial/HAP_100k.npz'
    totalRead_file = outLoc + '/initial/totalReads.npz'
    #useHG38 = True
    #if folder1 == 'DLP':
    #    useHG38 = False
    lowHapDoImbalance = True
    #if folder1 == 'DLP':
    #    lowHapDoImbalance = True

    stepName += 1
    stepString = str(stepName) + '/' + numSteps
    print ('Data processing — Step ' + stepString + ': GC bias correction... ', end='')

    findGCadjustment(refGenome, refLoc, hist_file, adjustment_file, chr_file, lowHapDoImbalance, rawHAP_file, hapHist_file,  chr_file2, goodSubset_file, RDR_file, totalRead_file)


    
    hapHist_file = outLoc + '/initial/HAP_100k.npz' #used to be 10k
    RDR_file = outLoc + '/initial/RDR_100k.npz' #used to be 10k
    chr_file = outLoc + '/initial/chr_100k.npz'
    adjustment_file = outLoc + '/initial/gc_adjustment.npz'
    goodSubset_file = outLoc + '/initial/subset.npz'
    chr_file_2 = outLoc + '/initial/chr_1M.npz'
    RDR_file_2 = outLoc + '/initial/RDR_1M.npz'
    BAF_file_2 = outLoc + '/initial/HAP_1M.npz'
    cellGood_file = outLoc + '/initial/cellGood.npz'
    doBAF = True
    saveRDR(RDR_file, adjustment_file, chr_file, goodSubset_file, chr_file_2, RDR_file_2, cellGood_file, doBAF, hapHist_file=hapHist_file, BAF_file_2=BAF_file_2)
    print ('Done')


refLoc = './data/refNew'
outLoc = './data/newTN3'
refGenome = 'hg38'
#runProcessFull(outLoc, refLoc, refGenome)





#quit()

