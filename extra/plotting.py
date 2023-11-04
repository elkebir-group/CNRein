
#CNA.py

import numpy as np


import matplotlib.pyplot as plt
import time
import scipy
from scipy import stats
from scipy.special import logsumexp



import pandas as pd
#%matplotlib inline
#%config InlineBackend.figure_format = 'svg'
import matplotlib as mpl
import seaborn as sns

#sns.set_style('whitegrid')
#mpl.rc('text', usetex=True)
#sns.set_context("notebook", font_scale=1.4)


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.optim import Optimizer



#from scaler import *
#from RLCNA import *


#for fancy plot Chisel
from matplotlib.colors import LinearSegmentedColormap
from itertools import cycle

from shared import *







def loadnpz(name, allow_pickle=False):

    #This simple function more easily loads in compressed numpy files.

    if allow_pickle:
        data = np.load(name, allow_pickle=True)
    else:
        data = np.load(name)
    data = data.f.arr_0
    return data



def uniqueValMaker(X):

    _, vals1 = np.unique(X[:, 0], return_inverse=True)

    for a in range(1, X.shape[1]):

        #vals2 = np.copy(X[:, a])
        #vals2_unique, vals2 = np.unique(vals2, return_inverse=True)
        vals2_unique, vals2 = np.unique(X[:, a], return_inverse=True)

        vals1 = (vals1 * vals2_unique.shape[0]) + vals2
        _, vals1 = np.unique(vals1, return_inverse=True)

    return vals1


def OLD_testDLP():




    #df_ov2295_cn_origin = pd.read_csv("./fromServer/ov2295_cell_cn.csv.gz")
    #ar = df_ov2295_cn_origin.to_numpy()

    #print (df_ov2295_cn_origin.keys())
    #print (ar[:10, :10])

    #quit()


    folder1 = '10x'


    Ncall = 20
    RDR_file = './data/' + folder1 + '/binScale/filtered_RDR_avg.npz'
    #RDR_file = './data/' + folder1 + '/binScale/RDR_adjusted.npz'

    chr_file = './data/' + folder1 + '/binScale/chr_avg.npz'
    region_file = './data/' + folder1 + '/binScale/regions.npz'
    HAP_file = './data/' + folder1 + '/binScale/filtered_HAP_avg.npz'
    initialCNA_file = './data/' + folder1 + '/binScale/initialCNA.npz'
    initialUniqueCNA_file = './data/' + folder1 + '/binScale/initialUniqueCNA.npz'
    originalError_file = './data/' + folder1 + '/originalError.npz' #2
    #modelName =  './data/DLP/model4.pt'
    #predict_file = './data/DLP/pred4.npz'

    modelName =  './data/' + folder1 + '/model1.pt'
    predict_file = './data/' + folder1 + '/pred1.npz'
    ##############call_file = './data/DLP/calls_avg2.npz'
    noise_file = './data/' + folder1 + '/binScale/filtered_RDR_noise.npz'
    BAF_noise_file = './data/' + folder1 + '/binScale/BAF_noise.npz'
    bins_file = './data/' + folder1 + '/binScale/bins.npz'
    

    #135, 138


    

    

    model = torch.load(modelName)
    adjustment = model.biasAdjuster()
    adjustment = adjustment.data.numpy()


    


    

    noise1 = loadnpz(noise_file)
    noiseBAF = loadnpz(BAF_noise_file)

    naiveCNA = loadnpz(initialCNA_file)
    predCNA = loadnpz(predict_file)#[:, :, 0]
    #callCNA = loadnpz(call_file)



    

    


    RDR  = loadnpz(RDR_file)
    HAP_all = loadnpz(HAP_file)
    bins = loadnpz(bins_file)
    chr = loadnpz(chr_file)


    #print (RDR.shape)
    #quit()

    predBAF = predCNA[:, :, 1] / ( np.sum(predCNA, axis=2) + 1e-5 )
    naiveBAF = naiveCNA[:, :, 1] / ( np.sum(naiveCNA, axis=2) + 1e-5 )

    measureBAF = HAP_all[:, :, 1] / (np.sum(HAP_all, axis=2) + 1e-5)


    measureBAF[np.sum(HAP_all, axis=2) < 10] = measureBAF[np.sum(HAP_all, axis=2) < 10] / 0


    predTotal = np.sum(predCNA, axis=2)
    naiveTotal = np.sum(naiveCNA, axis=2)

    inverse1 = uniqueValMaker(predTotal)
    print (np.unique(inverse1).shape)

    #sns.clustermap( predTotal  , col_cluster=False, row_cluster=True,linewidths=0.0) #, cmap=sns.color_palette("aired")+sns.color_palette("set2"))
    #plt.show()
    #quit()
    

    
    predRatio = np.mean(predTotal * noise1, axis=1) / np.mean(RDR * noise1, axis=1)

    #RDRscaled = RDR * np.mean(predTotal, axis=1).reshape((-1, 1))
    #RDRnaiveScaled = RDR * np.mean(naiveTotal, axis=1).reshape((-1, 1))

    RDRscaled = RDR * predRatio.reshape((-1, 1))


    argNoWGD = np.argwhere(np.median(predTotal, axis=1) == 2)[:, 0]
    argWGD = np.argwhere(np.median(predTotal, axis=1) != 2)[:, 0]

    #print (np.unique(chr))
    #sns.clustermap(  predTotal[:, chr==13-1]  , col_cluster=False, row_cluster=True,linewidths=0.0) #, cmap=sns.color_palette("aired")+sns.color_palette("set2"))
    #plt.show()
    #quit()

    #sns.clustermap(  predTotal[argWGD]  , col_cluster=False, row_cluster=True,linewidths=0.0) #, cmap=sns.color_palette("aired")+sns.color_palette("set2"))
    #plt.show()


    #predTotal = predTotal[argNoWGD]
    #predCNA = predCNA[argNoWGD]
    #naiveTotal = naiveTotal[argNoWGD]
    #RDRscaled = RDRscaled[argNoWGD]
    #measureBAF = measureBAF[argNoWGD]
    #predBAF = predBAF[argNoWGD]

    #print (np.min(predTotal[:, 11]))
    #print (np.unique(chr))
    #quit()
    #sns.clustermap(  predTotal  , col_cluster=False, row_cluster=True, linewidths=0.0, cmap=sns.color_palette("Paired"))
    #plt.show()
    #quit()
    
    #sns.clustermap(  predBAF  , col_cluster=False, row_cluster=True,linewidths=0.0, cmap='bwr')
    #sns.clustermap(  measureBAF  , col_cluster=False, row_cluster=True,linewidths=0.0, cmap='bwr')
    #plt.show()

    #quit()


    #chrIndexs = np.zeros((predTotal.shape[0], 22)).astype(int)
    #for a in range(22):
    #    pred1 = predTotal[:, chr==a]
    #    inverse1 = uniqueValMaker(pred1)
    #    chrIndexs[:, a] = np.copy(inverse1)

        #print (np.unique(inverse1).shape)

        #sns.clustermap(  pred1  , col_cluster=False, row_cluster=True,linewidths=0.0)
        #plt.show()

    #[[[49, 59], array([53, 60])]]


    #'''
    #argGood = np.argwhere(  np.logical_and(   np.isin(chrIndexs[:, 0], np.array([49, 59])),   np.isin(chrIndexs[:, 2], np.array([24, 54])  ) ) )[:, 0]

    #argGood2 = np.argwhere(  np.logical_and( chrIndexs[:, 0] == 49, chrIndexs[:, 1] == 60 )  )[:, 0]

    #plt.plot(RDRscaled[argGood2[1]][chr <= 2 ])
    #plt.plot(naiveTotal[argGood2[0]][chr <= 2 ].T)
    #plt.plot(naiveTotal[argGood2[0]][chr <= 2 ])
    #plt.plot(naiveTotal[argGood2[2]][chr <= 2 ].T+0.1)
    #plt.plot(measureBAF[argGood2[0]][chr <= 2 ])
    #plt.show()
    #quit()
    

    #13, 26# no spike, bright
    
    #sns.clustermap(  predTotal[argGood][:, np.isin(chr ,   np.array([0, 2])  )]  , col_cluster=False, row_cluster=True,linewidths=0.0)
    #plt.show()
    #quit()
    '''

    #chr1 = 0
    #chr2 = 2

    for chr1 in range(21):
        for chr2 in range(chr1, 22):

            matrix1 = np.zeros((200, 200))
            matrix2 = np.zeros((200, 200))
            for a in range(chrIndexs.shape[0]):
                matrix1[chrIndexs[a, chr1], chrIndexs[a, chr2]] = 1
                matrix2[chrIndexs[a, chr1], chrIndexs[a, chr2]] += 1

            matrix3 = np.copy(matrix2)
            matrix3[matrix3<=10] = 0
            matrix3[matrix3>=1] = 1

            pairing = []
            for a in range(matrix3.shape[0]-1):
                for b in range(a+1, matrix3.shape[0]):
                    ar1 = matrix3[a] * matrix3[b]
                    if np.sum(ar1) >= 2:
                        ar1 = np.argwhere(ar1 != 0)[:, 0]
                        pairing.append([ [a, b], np.copy(ar1) ])
            
            if len(pairing) > 0:
                print (chr1, chr2)
                print (pairing)
                quit()

    
    inverse1 = uniqueValMaker(chrIndexs[:, :2])
    _, index1, count1 = np.unique(inverse1, return_index=True, return_counts=True)
    
    ar1 = chrIndexs[index1, :3]
    ar1[:, 2] = count1

    #print (ar1[ ar1[:, 2] > 3 ])
    print (ar1)


    
    
    quit()
    '''
    


    
    #measureBAF[np.sum(HAP_all, axis=2) < 20] =  measureBAF[np.sum(HAP_all, axis=2) < 20] / 0

    #sns.clustermap(  predTotal  , col_cluster=False, row_cluster=True,linewidths=0.0)
    #sns.clustermap(  measureBAF  , col_cluster=False, row_cluster=True,linewidths=0.0)
    #sns.clustermap(  RDRscaled  , col_cluster=False, row_cluster=True,linewidths=0.0)
    #plt.show()

    #from scipy.cluster.hierarchy import dendrogram, linkage


    #86, 94, val 1.0


    #diff1 = np.sum(np.abs(predTotal[:, 84:95] - 1), axis=1)
    #argBad = np.argwhere(diff1 == 0)[:, 0]
    #plt.plot(predTotal[argBad].T)
    #plt.plot(RDRscaled[argBad[1] ])
    #plt.plot(predTotal[argBad[1] ])
    #plt.plot(predBAF[argBad[1] ])
    #plt.plot(measureBAF[argBad[1] ])
    #plt.show()
    #quit()


    


    

    #argCell = np.argsort( np.mean(predTotal, axis=1) )[115:130]
    argCell = np.argsort( np.mean(predTotal, axis=1) )#[100:200]

    #for a in range(100):
    #    plt.plot(RDRscaled[argCell[a], 130:140])
    #    plt.show()

    adjustment = model.biasAdjuster()


    
    #quit()
    #print (HAP_all[857, 13])
    #print (measureBAF[857, 13])
    #quit()

    
    if False:

        #singlePoints = [11, 13, 17, 19, 26, 30, 67, 75, 89]
        singlePoints = [13]
        #151, 153

        #123, 140

        #130, 135

        #11, 13

        #argCheck = np.argwhere( predTotal[:, 113] == 1 )[:, 0]

        argCheck = np.argwhere( np.logical_and( predTotal[:, 71] == 2,  predTotal[:, 72] == 3  ) )[:, 0]


        print (argCheck.shape)
        
        for a in argCheck:
            plt.plot( predTotal[a ] )
            plt.plot(RDRscaled[a])
            plt.plot( predBAF[ a ] )
            plt.plot( measureBAF[ a ] )
            plt.show()
        quit()
        
        #plt.plot( naiveTotal[argCell[501] ] )
        #plt.plot( RDRnaiveScaled[argCell[501]  ] )
        #plt.plot( predTotal[argCell[501] ] )
        #plt.plot( RDRscaled[argCell[501]  ] )
        #plt.show()
        #uit()

        
        for a in singlePoints:
            
            print (a)
            print (adjustment[a])

            adjust1 = 1 / (1 + adjustment[a].data.numpy())


            #plt.plot( naiveTotal[argCell , a ] )
            #plt.plot( RDRnaiveScaled[argCell , a ] )
            #plt.show()

            
            plt.plot( predTotal[argCell , a ] )
            plt.plot( naiveTotal[argCell , a ] )
            plt.plot( RDRscaled[argCell , a ] * adjust1 )
            plt.plot( measureBAF[argCell , a ] )
            plt.plot( np.zeros(argCell.shape[0]) + 0.5, c='grey'  )
            plt.plot( np.zeros(argCell.shape[0]) + 0.666, c='grey'  )
            plt.plot( np.zeros(argCell.shape[0]) + 0.333, c='grey'  )
            plt.show()
        #quit()



    #quit()


    RDR_lim = np.copy(RDR)
    RDR_lim[RDR_lim>10] = 10


    #quit()

    #RDR_scale1 = RDR * np.mean(predCNA, axis= (1, 2) ).reshape((-1, 1)) * 2


    scale1 = np.mean(predCNA, axis=1)

    #RDR_scale2 = RDR * np.mean(naiveCNA, axis=1).reshape((-1, 1))



    #plt.plot(np.mean(RDR, axis=1))
    #plt.show()

    #plt.hist(RDR_scale2[:200, 0], bins=100)
    #plt.hist(RDR_scale2[:200, 1], bins=100)
    #plt.show()

    sum1 = np.sum(predCNA, axis=1)
    predCNA_sort = predCNA[np.argsort(sum1)]




    #inverse_call = uniqueValMaker(callCNA)
    #inverse_pred = uniqueValMaker(predCNA)

    #unique_call, count_call = np.unique(inverse_call, return_counts=True)
    #unique_pred, index_pred, count_pred = np.unique(inverse_pred, return_index=True, return_counts=True)
    #count_inverse_pred = count_pred[inverse_pred]

    #index_pred_best = index_pred[count_pred > 5]

    

    #args1 = np.argwhere( np.isin(inverse_pred, inverse_pred[index_pred_best]) )[:, 0]
    #args1 = args1[np.argsort(count_inverse_pred[args1])]

    



    scale_naive = np.mean(naiveCNA, axis=1)
    scale_pred = np.mean(predCNA, axis=1)
    



    np.random.seed(1)
    #issueArg = issueArg[np.random.permutation(issueArg.shape[0])]
    #issueArg = np.argwhere(np.logical_and(int_naive == 2, int_pred == 1))[:, 0]
    #issueArg = np.argwhere(np.logical_and(int_naive == 1, int_pred == 2))[:, 0]

    #issueArg = np.argwhere( np.abs(scale_naive - scale_pred) > 1 )[:, 0]

    #issueArg = np.arange(scale_pred.shape[0])

    #np.random.seed(0)
    issueArg = np.random.permutation(scale_pred.shape[0])

    #bad: 0, 49,


    for a in range(0, issueArg.shape[0]):

        print (issueArg[a])

        #print (a)
        #plt.plot(RDR[issueArg[a]])
        #plt.plot(pred_RDR[issueArg[a]])
        #plt.plot(naive_RDR[issueArg[a]])
        #plt.plot(call_RDR[issueArg[a]])
        #plt.show()

        #diff1 = pred_RDR[issueArg[a]] - RDR[issueArg[a]]
        #diff2 = naive_RDR[issueArg[a]] - RDR[issueArg[a]]

        #plt.plot(diff1)
        #plt.plot(diff2)
        #plt.show()

        #print (callCNA[issueArg[a]].shape)
        #quit()

        #plt.plot(RDR[issueArg[a]]  )
        #plt.plot( predTotal[issueArg[a], :]  / np.mean(predTotal[issueArg[a], :])     )
        #plt.plot( naiveTotal[issueArg[a], :] / np.mean(naiveTotal[issueArg[a], :])    )
        
        
        plt.plot(RDR[issueArg[a]] * np.mean(predTotal[issueArg[a]]) )
        plt.plot( predTotal[issueArg[a], :].astype(float))

        #plt.plot(RDR[issueArg[a]] * np.mean(naiveTotal[issueArg[a]]) )
        #plt.plot( naiveTotal[issueArg[a], :].astype(float))

        #plt.show()

        plt.plot(predBAF[issueArg[a]])
        plt.plot(measureBAF[issueArg[a]])
        plt.show()


        #pred_fft =np.fft.fft(pred_RDR - RDR[issueArg[a]])
        #call_fft =np.fft.fft(call_RDR - RDR[issueArg[a]])
        #pred_fft = np.fft.fft(error_pred)

        #print (pred_fft.shape)
        #print (call_fft.shape)
        #quit()

        #plt.plot(np.abs(call_fft))
        #plt.plot(np.abs(pred_fft))
        #plt.show()




    quit()







def investigateTree():


    #folder1 = '10x'
    folder1 = 'ACT10x'

    modelName =  './data/' + folder1 + '/model2.pt'
    predict_file = './data/' + folder1 + '/pred2.npz'
    Ncall = 20


    chr_file = './data/' + folder1 + '/binScale/chr_avg.npz'
    chr = loadnpz(chr_file)
    _, start1 = np.unique(chr, return_index=True)
    end1 = np.concatenate((start1[1:], np.zeros(1) + chr.shape[0])).astype(int)

    CNAfull = loadnpz(predict_file)
    model = torch.load(modelName)
    info = []

    predTotal = np.sum(CNAfull, axis=2)

    inverse1 = uniqueValMaker(CNAfull.reshape((CNAfull.shape[0],CNAfull.shape[1]*2 ))  )
    _, index1, counts1 = np.unique(inverse1, return_index=True, return_counts=True)

    #sns.clustermap( predTotal  , col_cluster=False, row_cluster=True,linewidths=0.0) 
    #plt.show()

    indexCommon = index1[np.argsort(counts1)[-1::-1]]


    plt.imshow( predTotal[indexCommon[:5]], extent=[0, 2, 0, 1])
    plt.show()

    #plt.imshow( predTotal[indexCommon[:5] ], extent=[0, 2, 0, 1])
    #plt.show()
    #quit()

    #index1 = index1[np.argmax(counts1)]

    #N = 4
    #CNAfull2 = CNAfull[index1:index1+1] + np.zeros((N, 1, 1))

    CNAfull2 = CNAfull[indexCommon[:4]]

    modelProbSum, sampleProbSum, treeLoss, CNAused, savedCNA, stepLast = modelCNAgenerator(CNAfull2, chr, start1, end1, model, Ncall, info, returnReg=True, doDouble=True)
    modelProbSum = modelProbSum.data.numpy()

    probEnd = modelProbSum[stepLast+2, np.arange(stepLast.shape[0])]


    plt.plot(savedCNA[:stepLast[1]+1, 1, 89, 0])
    plt.plot(savedCNA[:stepLast[1]+1, 1, 89, 1])
    plt.show()
    #quit()


    savedCNA_sum = np.sum(savedCNA, axis=3)

    #plt.plot(savedCNA_sum[stepLast[0], 0])
    #plt.plot(savedCNA_sum[stepLast[1], 1])
    #plt.show()


    #plt.imshow(savedCNA_sum[:stepLast[0]+1, 0])
    #plt.show()
    #quit()

    if True:
        figure, axis = plt.subplots(2, 2)

        max1 = np.max(savedCNA_sum)

        axis[0, 0].imshow(savedCNA_sum[:stepLast[0]+1, 0], vmin=0,vmax=max1)
        axis[1, 0].imshow(savedCNA_sum[:stepLast[1]+1, 1], vmin=0,vmax=max1)
        axis[0, 1].imshow(savedCNA_sum[:stepLast[2], 2], vmin=0,vmax=max1)
        axis[1, 1].imshow(savedCNA_sum[:stepLast[3], 3], vmin=0,vmax=max1)
        plt.show()


    #plt.plot(modelProbSum[:, 0])
    #plt.plot(modelProbSum[:, 1])
    #plt.plot(modelProbSum[:, 2])
    #plt.show()

    #print (modelProbSum.shape)
    #quit()
    
    plt.hist(probEnd, bins=100)
    plt.show()





def draw(table, bins, pos, cells, index, clones, palette, center, method, metric, title, out, args):
    chr_palette = cycle(['#525252', '#969696', '#cccccc'])
    #chr_colors = {c : next(chr_palette) for c in sorted(set(b[0] for b in bins), key=orderchrs)}
    seen = set()
    seen_add = seen.add
    #ordclones = [clones[x] for x in table.index if not (clones[x][0] in seen or seen_add(clones[x][0]))]
    #cell_palette = cycle(sns.color_palette("muted", len(set(ordclones))))
    disc_palette = cycle(sns.color_palette("Greys", 8))
    #clone_colors = {i[0] : next(cell_palette) if i[1] != 'None' else '#f0f0f0' for i in ordclones}
    #cell_colors = {x : clone_colors[clones[x][0]] for x in table.index}

    para = {}
    para['data'] = table
    para['cmap'] = palette
    if center:
        para['center'] = center
    para['yticklabels'] = False
    para['row_cluster'] = False
    para['xticklabels'] = False
    para['col_cluster'] = False
    #para['figsize'] = args['gridsize']
    para['rasterized'] = True
    #para['col_colors'] = pd.DataFrame([{'index' : s, 'chromosomes' : chr_colors[pos[x][0]]} for x, s in enumerate(table.columns)]).set_index('index')
    #para['row_colors'] = pd.DataFrame([{'index' : x, 'Clone' : cell_colors[x]} for x in table.index]).set_index('index')
    #with warnings.catch_warnings():
    if True:
        #warnings.simplefilter("ignore")
        g = sns.clustermap(**para)
        addchr(g, pos)
        #g.fig.suptitle(title)
    #plt.savefig(out + args['format'], bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()



def states(): #(bins, pos, cells, index=None, clones=None, selected=None, args=None, out='allelecn.', val='CNS'):
    #avail = [(t - i, i) for t in xrange(7) for i in reversed(xrange(t+1)) if i <= t - i]
    #order = (lambda p : (max(p), min(p)))
    #convert = (lambda p : order(p) if sum(p) <= 6 else min(avail, key=(lambda x : abs(p[0] - x[0]) + abs(p[1] - x[1]))))
    #df = []
    #mapc = {}
    #found = set()
    #for x, e in enumerate(index):
    #   df.extend([{'Cell' : x, 'Genome' : bins[b][e]['Genome'], 'Value' : convert(bins[b][e][val])} for b in pos])
    #    mapc[x] = (clones[e], selected[e])
    #df = pd.DataFrame(df)
    #found = [v for v in avail if v in set(df['Value'])]
    
    #table = pd.pivot_table(df, values='CN states', columns=['Genome'], index=['Cell'], aggfunc='first')

    
    #df_original = pd.read_csv('./data/calls.tsv', sep="\t")#, header=None)
    df = pd.read_csv('./data/calls.tsv', sep="\t")#, header=None)
    #df = pd.read_csv('./data/combo.tsv', sep="\t", names=df_original.columns )#, header=None)

    #print (df)
    #quit()

    avail = [(t - i, i) for t in range(7) for i in reversed(range(t+1)) if i <= t - i]

    order = (lambda p : (max(p), min(p)))
    convert = (lambda p : order(p) if sum(p) <= 6 else min(avail, key=(lambda x : abs(p[0] - x[0]) + abs(p[1] - x[1]))))
    simpleSplitter = (lambda p: (  int( p.split('|')[0]) , int( p.split('|')[1])  )  )
    stringCombine = (lambda p, p2: p + p2  )


    #print (df['CN_STATE'])
    #quit()

    df['CN_STATE'] = df.apply(lambda r : simpleSplitter(r['CN_STATE']), axis=1)
    df['CN_STATE'] = df.apply(lambda r : convert(r['CN_STATE']), axis=1)
    #df['CN states'] = df.apply(lambda r : smap[r['Value']], axis=1)

    #df['Genome'] = df.apply(lambda r : stringCombine(r['CHR'], r['START']), axis=1)
    df['Genome'] = df["START"]


    set1 = set(df['CN_STATE'])
    found = [v for v in avail if v in set1]
    


    #df['CN_STATE'] = df.extend([{'Cell' : x, 'Genome' : bins[b][e]['Genome'], 'Value' : convert(bins[b][e][val])} for b in pos])
    #df['CN_STATE'] = [simpleSplitter(b) for b in df['CN_STATE']]
    #print (df['CN_STATE'])
    #quit()
    smap = {v : x for x, v in enumerate(found)}
    df['CN_STATE'] = df.apply(lambda r : smap[r['CN_STATE']], axis=1)

    #print (df['CN_STATE'])
    #quit()


    table = pd.pivot_table(df, values='CN_STATE', columns=['Genome'], index=['CELL'], aggfunc='first') #Start only works cuz only chr6 in this example. 
    #quit()

    #print (data)
    #quit()

    title = 'hello title'
    out='allelecn.'
    args=None

    title = 'Copy-number states'
    #found = set(df['CN states'] for i, r in df.iterrows())
    palette = {}
    palette.update({(0, 0) : 'darkblue'})
    palette.update({(1, 0) : 'lightblue'})
    palette.update({(1, 1) : 'lightgray', (2, 0) : 'dimgray'})
    palette.update({(2, 1) : 'lightgoldenrodyellow', (3, 0) : 'gold'})
    palette.update({(2, 2) : 'navajowhite', (3, 1) : 'orange', (4, 0) : 'darkorange'})
    palette.update({(3, 2) : 'salmon', (4, 1) : 'red', (5, 0) : 'darkred'})
    palette.update({(3, 3) : 'plum', (4, 2) : 'orchid', (5, 1) : 'purple', (6, 0) : 'indigo'})
    colors = [palette[c] for c in found]
    #print (colors)
    cmap = LinearSegmentedColormap.from_list('multi-level', colors, len(colors))
    #quit()
    draw(table, 'bins', 'pos', 'cells', 'index', 'mapc', palette=cmap, center=None, method='single', metric='cityblock', title=title, out=out, args=args)



def addchr(g, pos, color=None):
    corners = []
    prev = 0
    for x, b in enumerate(pos):
        if x != 0 and pos[x-1][0] != pos[x][0]:
            corners.append((prev, x))
            prev = x
    corners.append((prev, x))
    ax = g.ax_heatmap
    ticks = []
    for o in corners:
        ax.set_xticks(np.append(ax.get_xticks(), int(float(o[1] + o[0] + 1) / 2.0)))
        ticks.append(pos[o[0]][0])
    ax.set_xticklabels(ticks, rotation=45, ha='center')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)



#states()
#quit()




def specialPlotting():

    from itertools import cycle



    folder1 = 'ACT10x'
    #folder1 = '10x'
    #folder1 = 'DLP'
    #folder1 = 'TN3'


    chr_file = './data/' + folder1 + '/binScale/chr_avg.npz'
    predict_file = './data/' + folder1 + '/model/pred_good.npz'
    bins_file = './data/' + folder1 + '/binScale/bins.npz'

    chr = loadnpz(chr_file)
    predCNA = loadnpz(predict_file)
    bins = loadnpz(bins_file)

    



    #print (predCNA.shape)
    #quit()
    

    shape1 = predCNA.shape

    predCNA = predCNA.reshape((predCNA.shape[0]*predCNA.shape[1], 2))
    argBad = np.argwhere(np.sum(predCNA, axis=1) > 6)[:, 0]
    predCNA[argBad, 0] = 6
    predCNA[argBad, 1] = 0
    
    #print (predCNA.shape)
    predCNA = [(v[0], v[1]) for v in predCNA]
    predCNA = list(predCNA)
    

    avail = [(t - i, i) for t in range(7) for i in reversed(range(t+1)) if i <= t - i]
    order = (lambda p : (max(p), min(p)))
    convert = (lambda p : order(p) if sum(p) <= 6 else min(avail, key=(lambda x : abs(p[0] - x[0]) + abs(p[1] - x[1]))))
    
    predCNA = [convert(v) for v in predCNA]

    
    set1 = set(predCNA)
    found = [v for v in avail if v in set1]
    

    smap = {v : x for x, v in enumerate(found)}
    predCNA = [smap[v] for v in predCNA]

    #print (len(predCNA))



    palette = {}
    palette.update({(0, 0) : 'darkblue'})
    palette.update({(1, 0) : 'lightblue'})
    palette.update({(1, 1) : 'lightgray', (2, 0) : 'dimgray'})
    palette.update({(2, 1) : 'lightgoldenrodyellow', (3, 0) : 'gold'})
    palette.update({(2, 2) : 'navajowhite', (3, 1) : 'orange', (4, 0) : 'darkorange'})
    palette.update({(3, 2) : 'salmon', (4, 1) : 'red', (5, 0) : 'darkred'})
    palette.update({(3, 3) : 'plum', (4, 2) : 'orchid', (5, 1) : 'purple', (6, 0) : 'indigo'})
    colors = [palette[c] for c in found]
    cmap = LinearSegmentedColormap.from_list('multi-level', colors, len(colors))


    predCNA = np.array(predCNA).reshape((shape1[0], shape1[1]))


    predCNA = predCNA[:, bins]
    chr = chr[bins]


    #cbar_ax = plt.gca()
    #sns.clustermap(data, cbar_ax=cbar_ax)

    #print (predCNA.shape)

    #col_colors = pd.DataFrame([{'index' : s, 'chromosomes' : chr_colors[pos[x][0]]} for x, s in enumerate(table.columns)]).set_index('index')

    #chr_palette = cycle(['#525252', '#969696', '#cccccc'])
    chr_palette = ['#525252', '#969696', '#cccccc']
    #range1 = list(np.arange(chr.shape[0]))
    #chr_colors = {c : next(chr_palette) for c in range1, key=orderchrs)}
    chr_colors = [ chr_palette[chr[a]%3] for a in range(chr.shape[0])  ]


    #para = {}
    #para['data'] = predCNA
    #para['col_cluster'] = False
    #para['row_cluster'] = True
    #para['linewidths'] = 0.0
    #para['cmap'] = cmap
    #para['cbar_pos'] = None
    #para['xticklabels']=False
    #para['col_colors'] = chr_colors

    g = sns.clustermap( predCNA, col_cluster=False, row_cluster=True, linewidths=0.0, cmap=cmap, cbar_pos=None, yticklabels=False, xticklabels=False, col_colors=chr_colors)

    #sns.clustermap( para)

    

    #chr_colors = {c : next(chr_palette) for c in sorted(set(b[0] for b in bins), key=orderchrs)}


    
    corners = []
    prev = 0
    #for x, b in enumerate(pos):
    #    if x != 0 and pos[x-1][0] != pos[x][0]:
    #        corners.append((prev, x))
    #        prev = x

    
    for a in range(chr.shape[0]-1):
        if chr[a] != chr[a+1]:
            corners.append((a, a+1))
    corners.append((chr.shape[0]-1, chr.shape[0]))
    #corners.append((prev, x))
    
    #print (corners)
    #print (chr)
    #quit()

    #print (len(corners))
    #quit()
    
    ax = g.ax_heatmap

    #print (ax.get_xticks())
    #quit()

    ticks = []
    for o in corners:
        #print (o)
        #print (np.append(ax.get_xticks(), int(float(o[1] + o[0] + 1) / 2.0)))
        #print (len(ax.get_xticks()))
        ax.set_xticks(np.append(ax.get_xticks(), int(float(o[1] + o[0] + 1) / 2.0)))
        #ticks.append(pos[o[0]][0])
        ticks.append(chr[o[0]]+1)

    #print (ticks)
    #quit()

    #ax.set_xticklabels(ticks, rotation=45, ha='center')
    ax.set_xticklabels(ticks, rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    

    plt.tight_layout()
    #plt.show()

    #imageFileName = './images/' + folder1 + '_heatmap.pdf'
    imageFileName = './images/ACT/' + folder1 + '_heatmap.png'
    plt.savefig(imageFileName)



#specialPlotting()
#quit()


def plotBAF():



    #folder1 = 'ACT10x'
    #folder1 = '10x'
    #folder1 = 'DLP'
    folder1 = 'TN3'


    chr_file = './data/' + folder1 + '/binScale/chr_avg.npz'
    predict_file = './data/' + folder1 + '/pred_good.npz'
    bins_file = './data/' + folder1 + '/binScale/bins.npz'

    chr = loadnpz(chr_file)
    predCNA = loadnpz(predict_file)
    bins = loadnpz(bins_file)


    
    predCNA = predCNA[:, bins]
    chr = chr[bins]


    predBAF = predCNA[:, :, 1] / ( np.sum(predCNA, axis=2) + 1e-5 )


    chr_palette = ['#525252', '#969696', '#cccccc']
    chr_colors = [ chr_palette[chr[a]%3] for a in range(chr.shape[0])  ]


    g = sns.clustermap( predBAF, col_cluster=False, row_cluster=True, linewidths=0.0, cmap='bwr', yticklabels=False, xticklabels=False, col_colors=chr_colors)

    


    
    corners = []
    prev = 0
    
    
    for a in range(chr.shape[0]-1):
        if chr[a] != chr[a+1]:
            corners.append((a, a+1))
    corners.append((chr.shape[0]-1, chr.shape[0]))
    
    
    ax = g.ax_heatmap

    

    ticks = []
    for o in corners:
        print (o)
        print (np.append(ax.get_xticks(), int(float(o[1] + o[0] + 1) / 2.0)))
        #print (len(ax.get_xticks()))
        ax.set_xticks(np.append(ax.get_xticks(), int(float(o[1] + o[0] + 1) / 2.0)))
        #ticks.append(pos[o[0]][0])
        ticks.append(chr[o[0]]+1)

    
    ax.set_xticklabels(ticks, rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    

    plt.tight_layout()
    #plt.show()

    #imageFileName = './images/' + folder1 + '_heatmap.pdf'
    imageFileName = './images/' + folder1 + '_BAF_heatmap.png'
    plt.savefig(imageFileName)



#plotBAF()
#quit()



def testDLP():


    

    #df_ov2295_cn_origin = pd.read_csv("./fromServer/ov2295_cell_cn.csv.gz")
    #ar = df_ov2295_cn_origin.to_numpy()

    #print (df_ov2295_cn_origin.keys())
    #print (ar[:10, :10])

    #quit()


    folder1 = '10x'
    #folder1 = 'TN3'
    #folder1 = 'ACT10x'
    #folder1 = 'DLP'


    Ncall = 20
    RDR_file = './data/' + folder1 + '/binScale/filtered_RDR_avg.npz'
    #RDR_file = './data/' + folder1 + '/binScale/RDR_adjusted.npz'

    chr_file = './data/' + folder1 + '/binScale/chr_avg.npz'
    region_file = './data/' + folder1 + '/binScale/regions.npz'
    HAP_file = './data/' + folder1 + '/binScale/filtered_HAP_avg.npz'
    initialCNA_file = './data/' + folder1 + '/binScale/initialCNA.npz'
    initialUniqueCNA_file = './data/' + folder1 + '/binScale/initialUniqueCNA.npz'
    originalError_file = './data/' + folder1 + '/originalError.npz' #2
    #modelName =  './data/DLP/model4.pt'
    #predict_file = './data/DLP/pred4.npz'

    modelName =  './data/' + folder1 + '/model/model_2.pt' 
    predict_file = './data/' + folder1 + '/model/pred_2.npz'
    ##############call_file = './data/DLP/calls_avg2.npz'
    noise_file = './data/' + folder1 + '/binScale/filtered_RDR_noise.npz'
    BAF_noise_file = './data/' + folder1 + '/binScale/BAF_noise.npz'
    bins_file = './data/' + folder1 + '/binScale/bins.npz'
    

    #135, 138


    naiveCNA = loadnpz(initialCNA_file)
    naiveTotal = np.sum(naiveCNA, axis=2)
    naiveTotal[naiveTotal > 10] = 10


    #sns.clustermap( naiveTotal  , col_cluster=False, row_cluster=True,linewidths=0.0) #, cmap=sns.color_palette("aired")+sns.color_palette("set2"))
    #plt.show()
    #quit()

    


    #14

    #model = torch.load(modelName)
    #adjustment = model.biasAdjuster()
    #adjustment = adjustment.data.numpy()


    


    

    noise1 = loadnpz(noise_file)
    noiseBAF = loadnpz(BAF_noise_file)

    naiveCNA = loadnpz(initialCNA_file)
    predCNA = loadnpz(predict_file)#[:, :, 0]
    #callCNA = loadnpz(call_file)



    
    


    RDR  = loadnpz(RDR_file)
    HAP_all = loadnpz(HAP_file)
    bins = loadnpz(bins_file)
    chr = loadnpz(chr_file)

    #print (RDR.shape)
    #print (naiveCNA.shape)
    #print (predCNA.shape)
    #quit()

    


    predBAF = predCNA[:, :, 1] / ( np.sum(predCNA, axis=2) + 1e-5 )
    naiveBAF = naiveCNA[:, :, 1] / ( np.sum(naiveCNA, axis=2) + 1e-5 )

    measureBAF = HAP_all[:, :, 1] / (np.sum(HAP_all, axis=2) + 1e-5)


    sns.clustermap(  measureBAF[0::5, chr == 3]  , col_cluster=False, row_cluster=True,linewidths=0.0, cmap='bwr')
    plt.show()
    quit()

    

    


    measureBAF[np.sum(HAP_all, axis=2) < 10] = measureBAF[np.sum(HAP_all, axis=2) < 10] / 0
    #measureBAF[np.sum(HAP_all, axis=2) < 30] = measureBAF[np.sum(HAP_all, axis=2) < 30] / 0


    predTotal = np.sum(predCNA, axis=2).astype(float)
    naiveTotal = np.sum(naiveCNA, axis=2).astype(float)

    RDR_scaled = RDR * np.mean(predTotal, axis=1).reshape((-1, 1))

    inverse1 = uniqueValMaker(predTotal)
    #print (np.unique(inverse1).shape)


    argNormal = np.sum(np.abs(predCNA - 1), axis=(1, 2))
    argNormal = np.argwhere(argNormal < 5)[:, 0]

    #RDRmean = np.mean(RDR_scaled[argNormal], axis=0)

    #plt.plot(RDRmean)
    #plt.show()

    #154, 155

    #print (HAP_all[58, 154:156] )
    #quit()

    print ('banana')

    #print (RDR_scaled[0, :].shape)
    #print (predTotal.shape)
    #print (predTotal[0, :].shape)

    #print (predTotal[0, :])
    #print (RDR_scaled[0, :])

    args2 = np.argwhere(  np.logical_and( np.max(predCNA[:, bins[500] ], axis=1) == 3, np.min(predCNA[:, bins[500] ], axis=1) == 2  )   )[:, 0]

    #print (args2.shape)
    #quit()

    haplotypePlotter(predCNA, doCluster=True)
    
    #haplotypePlotter(naiveCNA[:][:, bins[:1000]], doCluster=True)
    quit()

    
    RDR_file2 = './data/' + folder1 + '/initial/RDR_100k.npz'
    RDR = loadnpz(RDR_file2)

    #_, index1 = np.unique(bins, return_index=True)
    #print (index1[:10])

    #a = args2[1]

    args2 = [479, 812, 148, 1, 729, 794, 577, 293, 174, 269]
    for a0 in range(len(args2)):
        a = args2[a0]
        plt.plot(RDR[a, :1000] * np.mean(naiveTotal[a]))
        plt.plot(naiveTotal[a, bins[:1000]])
        plt.plot(bins[:1000] % 2)
        plt.show()

    
    quit()

    
    plt.plot(RDR_scaled[0, :1000])
    plt.plot(predTotal[0, :1000])
    #plt.plot(measureBAF[58, :])
    plt.show()
    quit()


    #RDR_scaled2 = np.copy(RDR_scaled)
    #RDR_scaled2[RDR_scaled2 > 5] = 5
    #plt.imshow(predTotal[argNormal], extent=[0, 1, 0, 1])
    #sns.clustermap( measureBAF[argNormal]  , col_cluster=False, row_cluster=True,linewidths=0.0) #, cmap=sns.color_palette("aired")+sns.color_palette("set2"))
    #plt.show()
    #quit()

    #sns.clustermap( predTotal  , col_cluster=False, row_cluster=True,linewidths=0.0) #, cmap=sns.color_palette("aired")+sns.color_palette("set2"))
    #plt.show()
    #quit()
    

    
    predRatio = np.mean(predTotal * noise1, axis=1) / np.mean(RDR * noise1, axis=1)

    #RDRscaled = RDR * np.mean(predTotal, axis=1).reshape((-1, 1))
    #RDRnaiveScaled = RDR * np.mean(naiveTotal, axis=1).reshape((-1, 1))

    RDRscaled = RDR * predRatio.reshape((-1, 1))


    argNoWGD = np.argwhere(np.median(predTotal, axis=1) == 2)[:, 0]
    argWGD = np.argwhere(np.median(predTotal, axis=1) != 2)[:, 0]


    #sns.clustermap( predBAF[argWGD]  , col_cluster=False, row_cluster=True,linewidths=0.0, cmap='bwr') #, cmap=sns.color_palette("aired")+sns.color_palette("set2"))
    #plt.show()
    #quit()

    #sns.clustermap(  predTotal[0::5]  , col_cluster=False, row_cluster=True,linewidths=0.0) #, cmap=sns.color_palette("aired")+sns.color_palette("set2"))
    #plt.show()


    #predTotal = predTotal[argNoWGD]
    #predCNA = predCNA[argNoWGD]
    #naiveTotal = naiveTotal[argNoWGD]
    #RDRscaled = RDRscaled[argNoWGD]
    #measureBAF = measureBAF[argNoWGD]
    #predBAF = predBAF[argNoWGD]

    #print (np.min(predTotal[:, 11]))
    #print (np.unique(chr))
    #quit()
    #sns.clustermap(  predTotal  , col_cluster=False, row_cluster=True, linewidths=0.0, cmap=sns.color_palette("Paired"))
    #plt.show()
    #quit()
    
    #sns.clustermap(  predBAF  , col_cluster=False, row_cluster=True,linewidths=0.0, cmap='bwr')
    #sns.clustermap(  measureBAF  , col_cluster=False, row_cluster=True,linewidths=0.0, cmap='bwr')
    #plt.show()

    #quit()




    #argCheck = np.argwhere( predTotal[:, 14] == 2)[:, 0]
    #argCheck = np.argwhere( np.logical_and( predTotal[:, 14] == 2 , predTotal[:, 13] == 3   ) )[:, 0]
    #for a in argCheck:
    #    plt.plot(predTotal[a])
    #   plt.plot(RDR_scaled[a])
    #    plt.plot(measureBAF[a])
    #    plt.show()
    #quit()
    

    


    

    #argCell = np.argsort( np.mean(predTotal, axis=1) )[115:130]
    argCell = np.argsort( np.mean(predTotal, axis=1) )#[100:200]

    #for a in range(100):
    #    plt.plot(RDRscaled[argCell[a], 130:140])
    #    plt.show()

    adjustment = model.biasAdjuster()


    
    


    RDR_lim = np.copy(RDR)
    RDR_lim[RDR_lim>10] = 10


    #quit()

    #RDR_scale1 = RDR * np.mean(predCNA, axis= (1, 2) ).reshape((-1, 1)) * 2


    scale1 = np.mean(predCNA, axis=1)

    

    sum1 = np.sum(predCNA, axis=1)
    predCNA_sort = predCNA[np.argsort(sum1)]






    scale_naive = np.mean(naiveTotal, axis=1)
    scale_pred = np.mean(predTotal, axis=1)
    

    #plt.scatter(scale_naive, scale_pred)
    #plt.show()



    #np.random.seed(1)
    #issueArg = issueArg[np.random.permutation(issueArg.shape[0])]
    #issueArg = np.argwhere(np.logical_and(int_naive == 2, int_pred == 1))[:, 0]
    #issueArg = np.argwhere(np.logical_and(int_naive == 1, int_pred == 2))[:, 0]

    #issueArg = np.argwhere( np.abs(scale_naive - scale_pred) > 1 )[:, 0]

    #issueArg = np.arange(scale_pred.shape[0])

    #np.random.seed(0)
    np.random.seed(1)
    issueArg = np.random.permutation(scale_pred.shape[0])

    #bad: 0, 49,


    for a in range(0, issueArg.shape[0]):

        print (issueArg[a])

        #print (a)
        #plt.plot(RDR[issueArg[a]])
        #plt.plot(pred_RDR[issueArg[a]])
        #plt.plot(naive_RDR[issueArg[a]])
        #plt.plot(call_RDR[issueArg[a]])
        #plt.show()

        #diff1 = pred_RDR[issueArg[a]] - RDR[issueArg[a]]
        #diff2 = naive_RDR[issueArg[a]] - RDR[issueArg[a]]

        #plt.plot(diff1)
        #plt.plot(diff2)
        #plt.show()

        #print (callCNA[issueArg[a]].shape)
        #quit()

        plt.plot(RDR[issueArg[a]]    )
        plt.plot( predTotal[issueArg[a], :]   / np.mean(predTotal[issueArg[a], :])    )
        plt.plot( naiveTotal[issueArg[a], :] / np.mean(naiveTotal[issueArg[a], :])    )
        
        
        #plt.plot(RDR[issueArg[a]] * np.mean(predTotal[issueArg[a]]) )
        #plt.plot( predTotal[issueArg[a], :].astype(float))

        #plt.plot(RDR[issueArg[a]] * np.mean(naiveTotal[issueArg[a]]) )
        #plt.plot( naiveTotal[issueArg[a], :].astype(float))

        plt.show()

        #plt.plot(predBAF[issueArg[a]])
        #plt.plot(measureBAF[issueArg[a]])
        #plt.show()


        #pred_fft =np.fft.fft(pred_RDR - RDR[issueArg[a]])
        #call_fft =np.fft.fft(call_RDR - RDR[issueArg[a]])
        #pred_fft = np.fft.fft(error_pred)

        #print (pred_fft.shape)
        #print (call_fft.shape)
        #quit()

        #plt.plot(np.abs(call_fft))
        #plt.plot(np.abs(pred_fft))
        #plt.show()




    quit()




#testDLP()
#quit()


def plotReadSim():

    def rebinChr(X, M, chr):

        print (X.shape)
        print (chr.shape)

        X2 = np.zeros(X.shape[0], dtype=float)
        count1 = 0
        for a in range(22):
            args1 = np.argwhere(chr == a)[:, 0]
            X_mini = rebin(X[args1], M)
            size1 = X_mini.shape[0]
            X2[count1:count1+size1] = X_mini
            count1 += size1 
        
        X2 = X2[:count1]

        return X2



    folder1 = '10x'


    RDR = loadnpz('./data/' + folder1 + '/initial/RDR_100k.npz')

    chr = loadnpz('./data/' + folder1 + '/initial/chr_100k.npz')

    profiles_full = loadnpz('./data/simulation/' + folder1 + '/profiles_sim' + str(14) + '.npz')
    RDR_sim = loadnpz('./data/simulation/' + folder1 + '/RDR_sim' + str(14) + '.npz')
    RDR_sim1 = RDR_sim[0]
    

    totalRead_file = './data/' + folder1 + '/initial/totalReads.npz'

    totalRead = loadnpz(totalRead_file)
    readMean = int(np.mean(totalRead))

    #plt.plot(np.sum(profiles_full[0], axis=1))
    #plt.show()
    #quit()
    #print (readMean)
    #quit()
    

    trueProfile1 = np.sum(profiles_full[0], axis=1).astype(float)
    probCopy = trueProfile1 / np.sum(trueProfile1)

    values = np.random.choice(probCopy.shape[0], size=readMean, replace=True, p=probCopy)

    vectorSum = np.zeros(probCopy.shape[0])

    unique1, count1 = np.unique(values, return_counts=True)

    vectorSum[unique1] = count1
    vectorSum = rebinChr(vectorSum, 10, chr)
    vectorSum = vectorSum.astype(float)
    vectorSum = vectorSum / np.mean(vectorSum)


    RDR1 = RDR[1000]
    RDR1 = rebinChr(RDR1, 10, chr)#rebin(RDR1, 10)
    RDR1 = RDR1 / np.mean(RDR1)

    RDR_sim1 = rebinChr(RDR_sim1, 10, chr)
    RDR_sim1 = RDR_sim1 / np.mean(RDR_sim1)

    trueProfile1 = trueProfile1[0::10]


    RDR_sim1 = RDR_sim1 * np.mean(trueProfile1)
    vectorSum = vectorSum * np.mean(trueProfile1)

    saveName1 = './images/simExample/readDepth_ourSim.pdf'
    saveName2 = './images/simExample/readDepth_simpleSim.pdf'
    saveName3 = './images/simExample/readDepth_realData.pdf'

    saveNames = [saveName1, saveName2, saveName3]
    plotData = [RDR_sim1, vectorSum, RDR1]

    for a in range(3):

        plt.plot(plotData[a])
        if a != 2:
            plt.plot(trueProfile1)
        plt.xlabel("genomic bin")
        if a == 2: 
            plt.ylabel('read depth')
        else:
            plt.ylabel('scaled read depth')
        if a != 2:
            plt.legend(['scaled read depth', 'ground truth'])
        #plt.gcf().set_size_inches(8, 3)
        plt.gcf().set_size_inches(10, 3)
        plt.tight_layout()
        plt.savefig(saveNames[a])
        plt.show()



#plotReadSim()
#quit()




def plotBAFsim():


    folder1 = '10x'
    HAP = loadnpz('./data/' + folder1 + '/initial/HAP_100k.npz')[1000]

    #profiles_full = loadnpz('./data/simulation/' + folder1 + '/profiles_sim' + str(14) + '.npz')
    #HAP_sim = loadnpz('./data/simulation/' + folder1 + '/HAP_sim' + str(14) + '.npz')
    HAP_sim = loadnpz('./data/simulation/' + folder1 + '/HAP_sim' + str(24) + '.npz')
    HAP_sim = HAP_sim[0]

    print (HAP.shape)

    HAP0 = rebin(HAP[:, 0], 10) * 10
    HAP1 = rebin(HAP[:, 1], 10) * 10

    HAP_sim0 = rebin(HAP_sim[:, 0], 10) * 10
    HAP_sim1 = rebin(HAP_sim[:, 1], 10) * 10

    BAF = (HAP1 + 1e-5)/ (HAP0 + HAP1 + 2e-5)

    BAF_sim = (HAP_sim1 + 1e-5)/ (HAP_sim0 + HAP_sim1 + 2e-5)

    #plt.plot(BAF)
    plt.plot(BAF_sim)
    plt.show()



#plotBAFsim()
#quit()


def plotCompare():

    folder1 = '10x'

    chr_file = './data/' + folder1 + '/initial/allChr_10k.npz'
    goodSubset_file = './data/' + folder1 + '/initial/subset.npz'
    chr = loadnpz(chr_file)

    print (chr.shape)


    #data = loadnpz('./data/comparison/input/10x_signals.npz')
    #data = data[1:]

    #nameStart = 'DLP_signals'
    nameStart = '10x_signals'
    #nameStart = '10x_chisel'

    copyMatrix = loadnpz('./data/comparison/input/' + nameStart + '_copyNumbers.npz')
    #copyMatrix = loadnpz('./data/comparison/input/' + nameStart + '_copyNoMissing.npz') #_copyNumbers
    sampleList = loadnpz('./data/comparison/input/' + nameStart + '_cell.npz')
    positionList = loadnpz('./data/comparison/input/' + nameStart + '_positions.npz')

    #for a in range(len(positionList)):
    #    print (positionList[a])
    #quit()

    #print (copyMatrix.shape)
    #quit()

    #copyMatrix[copyMatrix<0] = 0
    #copyMatrix[copyMatrix>=0] = 0
    #copyMatrix[copyMatrix<=-1] = -1

    copyMatrix = copyMatrix.reshape((copyMatrix.shape[0], copyMatrix.shape[1]*2))

    inverse1 = uniqueValMaker(copyMatrix)

    unique1 = np.unique(inverse1)

    print (inverse1.shape)
    print (unique1.shape)


    #haplotypePlotter(copyMatrix, doCluster=True)

    #print (np.mean(copyMatrix))

    #print (np.unique(copyMatrix))
    #quit()
    #quit()

    #print (copyMatrix.shape)

    #plt.imshow(np.sum(copyMatrix, axis=2))
    #plt.show()

    #quit()



    sampleList, index1 = np.unique(data[:, 3], return_index=True)
    sampleList = sampleList[np.argsort(index1)]
    sampleNum = sampleList.shape[0]

    #print (np.unique(data[:, 0]))
    #quit()
    positionNum = data.shape[0] // sampleNum
    positionList = data[:positionNum, :2]

    

    print (data[positionNum-2:positionNum+2])

    

    copyNumber = copyNumber.reshape((  sampleNum, copyNumber.shape[0] // sampleNum, 2))

    print (copyNumber.shape)

    np.savez_compressed('./data/comparison/input/10x_signals_copyNumbers.npz', copyNumber)
    np.savez_compressed('./data/comparison/input/10x_signals_cell.npz', sampleList)
    np.savez_compressed('./data/comparison/input/10x_signals_positions.npz', positionList)





    #4
    
    quit()


    

    #positions = data



    #binNum = 

    print (data[:1000, 1])

    #print (data.shape)




#plotCompare()
#quit()



def plotACTcloneSizes():


    def getCounts(predCNA1):
        inverse1 = uniqueProfileMaker(predCNA1)
        _, counts1 = np.unique(inverse1, return_counts=True)
        counts1 = np.sort(counts1)[-1::-1]
        return counts1
  

    predict_file1 = './data/ACT10x/model/pred_good.npz'
    predict_file2 = './data/TN3/model/pred_good.npz'

    naive_file1 = './data/ACT10x/binScale/initialCNA.npz'
    naive_file2 = './data/TN3/binScale/initialCNA.npz'

    
    predCNA1 = loadnpz(predict_file1)
    predCNA2 = loadnpz(predict_file2)
    naiveCNA1 = loadnpz(naive_file1)
    naiveCNA2 = loadnpz(naive_file2)

    counts1 = getCounts(predCNA1)
    counts2 = getCounts(predCNA2)
    countsNaive1 = getCounts(naiveCNA1)
    countsNaive2 = getCounts(naiveCNA2)

    #print (counts1.shape)
    #print (counts2.shape)
    #quit()

    print (np.max(counts1))
    print (np.max(countsNaive1))
    print (np.max(counts2))
    print (np.max(countsNaive2))



    
    N = 20

    arange1 = np.arange(20) + 1

    plt.plot(arange1, counts1[:N], color='blue')
    plt.plot(arange1, countsNaive1[:N], color='lightblue')
    plt.yscale('log')
    plt.legend(['DeepCopy', 'NaiveCopy'])
    plt.ylabel('clone size (number of cells)')
    plt.xlabel('clone number')
    plt.xticks(arange1)
    plt.tight_layout()
    plt.savefig('./images/ACT/cloneSizes_ACT10x.pdf')
    plt.show()

    N = 20

    plt.plot(arange1, counts2[:N], color='blue')
    plt.plot(arange1, countsNaive2[:N], color='lightblue')
    plt.yscale('log')
    plt.legend(['DeepCopy', 'NaiveCopy'])
    plt.ylabel('clone size (number of cells)')
    plt.xlabel('clone number')
    plt.xticks(arange1)
    plt.tight_layout()
    plt.savefig('./images/ACT/cloneSizes_TN3.pdf')
    plt.show()

    

#plotACTcloneSizes()
#quit()


def runtimePlotSim():

    folder1 = '10x_2'

    runtimeName = 'runtime (hours)'
    plotData = {}
    plotData[runtimeName] = []
    plotData['method'] = []

    naiveRuntime = []
    deepRuntime = []
    for simNum in range(10, 30):
        timeVector = loadnpz('./data/simulation/' + folder1 + '/timeVector_' + str(simNum) + '.npz')
        timeVector = timeVector - timeVector[0]

        timeVector = (timeVector / 60) / 60

        #print (timeVector[-2:])

        naiveRuntime.append(timeVector[-2])
        deepRuntime.append(timeVector[-1])
        
        plotData[runtimeName].append(timeVector[-1])
        plotData[runtimeName].append(timeVector[-2])
        plotData['method'].append('DeepCopy')
        plotData['method'].append('NaiveCopy')

    naiveRuntime = np.array(naiveRuntime)
    deepRuntime = np.array(deepRuntime)
    print (np.min(naiveRuntime) * 60)
    print (np.max(naiveRuntime) * 60)
    quit()

    
    import pandas as pd

    maxRuntime = np.max(np.array(plotData[runtimeName]))

    df = pd.DataFrame(data=plotData)
    palette = ['blue', 'lightblue']

    doLog = True


    ax = sns.boxplot(df, x='method', y=runtimeName, palette=palette)
    plt.xticks(rotation = 90)
    plt.gcf().set_size_inches(4, 4)
    if not doLog:
        plt.ylim( 0,  maxRuntime * 1.05)
        saveFile = './images/simulation/runtime.pdf'
    else:
        plt.yscale('log')
        saveFile = './images/simulation/runtime_log.pdf'

    plt.xlabel('')

    
    plt.tight_layout()
    plt.savefig(saveFile)
    plt.show()


runtimePlotSim()
quit()


def simulationPlot():


    #simNum = 8

    simNum = 5
    folder1 = '10x'
    predict_file = './data/simulation/' + folder1 + '/pred_sim' + str(simNum) + '.npz'
    profiles_full = loadnpz('./data/simulation/' + folder1 + '/profiles_sim' + str(simNum) + '.npz')
    bins = loadnpz('./data/simulation/' + folder1 + '/bins_sim' + str(simNum) + '.npz')
    pred1 = loadnpz(predict_file)
    naive1 = loadnpz('./data/simulation/' + folder1 + '/initialCNA_sim' + str(simNum) + '.npz')


    inverse1 = uniqueProfileMaker(profiles_full  )


    #sns.clustermap( np.sum(profiles_full[:, 0::10], axis=2), col_cluster=False, row_cluster=True, linewidths=0.0)
    #sns.clustermap( np.sum(naive1, axis=2), col_cluster=False, row_cluster=True, linewidths=0.0)
    #sns.clustermap( np.sum(pred1, axis=2), col_cluster=False, row_cluster=True, linewidths=0.0)
    #plt.show()
    #quit()

    _, counts1_true = np.unique(inverse1, return_counts=True)
    counts1_true = np.sort(counts1_true)[-1::-1]
    #print (counts1.shape)
    #plt.plot(np.sort(counts1)[-1::-1])
    #plt.show()
    #quit()


    pred_inverse1 = uniqueProfileMaker(pred1)
    naive_inverse1 = uniqueProfileMaker(naive1)

    pred1 = pred1[:, bins]
    naive1 = naive1[:, bins]



    #plt.scatter( np.mean(profiles_full, axis=(1, 2)), np.mean(pred1, axis=(1, 2)) )
    #plt.show()
    #a = 0
    #plt.plot(np.sum(naive1[a], axis=1)  )
    #plt.plot(np.sum(profiles_full[a], axis=1)  )
    #plt.show()
    #quit()

    #plt.hist( np.mean(pred1, axis=(1, 2)) , bins=100 )
    #plt.show()

    #print (np.mean(pred1))
    #print (np.mean(naive1))
    #quit()

    profiles_full = profiles_full[:, :, -1::-1]


    error1 = np.sum(np.abs(profiles_full - pred1), axis=2)
    error1[error1!=0] = 1


    error2 = np.sum(np.abs(profiles_full - naive1), axis=2)
    error2[error2!=0] = 1
    error2 = np.mean(error2.astype(float), axis=1)

    error2_B = np.sum(np.abs(profiles_full - (naive1*2)), axis=2)
    error2_B[error2_B!=0] = 1
    error2_B = np.mean(error2_B.astype(float), axis=1)

    error2_min = np.min(np.array([error2, error2_B]), axis=0)




    print (np.mean(error1.astype(float)))
    print (np.mean(error2.astype(float)))
    print (np.mean(error2_min))



    _, pred_counts1 = np.unique(pred_inverse1, return_counts=True)
    _, naive_counts1 = np.unique(naive_inverse1, return_counts=True)

    pred_counts1 = np.sort(pred_counts1)[-1::-1]
    naive_counts1 = np.sort(naive_counts1)[-1::-1]

    print (pred_counts1.shape)
    print (naive_counts1.shape)
    print (counts1_true.shape)

    N = 200# 20
    #N = 400# 20

    plt.plot(pred_counts1[:N])
    plt.plot(naive_counts1[:N])
    plt.plot(counts1_true[:N])
    plt.yscale('log')
    plt.show()


#simulationPlot()
#quit()






def genPlotFig():

    ar_CNA1 = np.ones(570)
    ar_CNA2 = np.ones(570)
    ar_CNA3 = np.ones(570)
    ar1 = np.ones(570)
    ar2 = np.ones(570)



    ar_CNA1[100:200] = 2
    ar_CNA2[400:450] = 3
    ar_CNA3[300:450] = 0
    ar1[100:200] = 2
    ar1[400:450] = 3
    ar2[300:450] = 0

    saveFile1 = './images/demoHaplotype1.pdf'
    saveFile2 = './images/demoHaplotype2.pdf'

    #'''
    ar1_blank = np.copy(ar1)
    ar1_blank[ar1_blank!=1] = float('nan') # 1/0
    ar_red = np.copy(ar_CNA1)
    ar_green = np.copy(ar_CNA2)
    ar_red[ar_CNA2 != 1] = float('nan')
    ar_green[ar_CNA1 != 1] = float('nan')
    plt.plot( ar_red , color='red')#, linewidth=5)
    plt.plot( ar_green , color='green')
    plt.plot( ar1_blank , color='grey')#, linewidth=5 )
    plt.ylim(-0.5, 3.5)
    plt.yticks([0, 1, 2, 3])
    plt.grid(axis='y')
    #plt.gcf().set_size_inches(8, 3)
    plt.gcf().set_size_inches(8, 2.5)
    plt.tight_layout()
    #plt.legend(['Incorrect Minority Haplotype', 'Correct Minority Haplotype'], loc='lower center')
    plt.savefig(saveFile1)
    plt.show()
    plt.clf()
    #'''
    #quit()

    ar2_blank = np.copy(ar2)
    ar2_blank[ar2_blank!=1] = float('nan') # 1/0
    plt.plot( ar_CNA3 , color='blue')
    plt.plot( ar2_blank , color='grey')#, linewidth=5 )
    plt.ylim(-0.5, 3.5)
    plt.yticks([0, 1, 2, 3])
    plt.grid(axis='y')
    #plt.gcf().set_size_inches(8, 3)
    plt.gcf().set_size_inches(8, 2.5)
    plt.tight_layout()
    #plt.legend(['Incorrect Minority Haplotype', 'Correct Minority Haplotype'], loc='lower center')
    plt.savefig(saveFile2)
    plt.show()
    plt.clf()





#genPlotFig()
#quit()



def saveSimErrors():

    def getPredictedRDR(copyNumbers):

        return np.mean(copyNumbers, axis=2) / np.mean(copyNumbers, axis=(1, 2)).reshape((-1, 1))
    

    def getPredictedBAF(copyNumbers):

        return (np.min(copyNumbers, axis=2) + 1e-5) / (np.sum(copyNumbers, axis=2) + 2e-5)


    #simNum = 3

    matrixErrors = np.zeros((20, 2, 3))

    trueErrors = np.zeros((20, 2, 2))

    clonesAll = np.zeros((20, 3, 1000), dtype=int)

    for simNum in range(0, 20):

        print (simNum)

        simNum1 = simNum + 10
    
        folder1 = '10x'
        profiles_full = loadnpz('./data/simulation/' + folder1 + '/profiles_sim' + str(simNum1) + '.npz')
        bins = loadnpz('./data/simulation/' + folder1 + '/bins_sim' + str(simNum1) + '.npz')
        pred1 = loadnpz('./data/simulation/' + folder1 + '/pred_sim' + str(simNum1) + '_new.npz')
        naive1 = loadnpz('./data/simulation/' + folder1 + '/initialCNA_sim' + str(simNum1) + '.npz')
        RDR = loadnpz('./data/simulation/' + folder1 + '/RDR_sim' + str(simNum1) + '.npz')
        HAP = loadnpz('./data/simulation/' + folder1 + '/HAP_sim' + str(simNum1) + '.npz')


        if False:
            inverse_pred = uniqueProfileMaker(pred1)
            inverse_naive = uniqueProfileMaker(naive1)
            inverse_true = uniqueProfileMaker(profiles_full)


            clonesAll[simNum, 0] = inverse_pred
            clonesAll[simNum, 1] = inverse_naive
            clonesAll[simNum, 2] = inverse_true



        pred1 = pred1[:, bins]
        naive1 = naive1[:, bins]
        profiles_full = profiles_full[:, :, -1::-1]
        
        
        
        error1 = np.sum(np.abs(profiles_full - pred1), axis=2)
        L1_error1 = np.mean(error1)
        error1[error1!=0] = 1
        #error1 = np.mean(error1.astype(float), axis=1)
        accuracy1 = 1 - np.mean(error1)
        trueErrors[simNum, 0, 0] = accuracy1
        trueErrors[simNum, 0, 1] = L1_error1
        


        error2 = np.sum(np.abs(profiles_full - naive1), axis=2)
        L1_error2 = np.mean(error2)
        error2[error2!=0] = 1
        accuracy2 = 1 - np.mean(error2)
        trueErrors[simNum, 1, 0] = accuracy2
        trueErrors[simNum, 1, 1] = L1_error2

        
        #unique_profile = [np.unique(inverse_pred).shape[0], np.unique(inverse_naive).shape[0], np.unique(inverse_true).shape[0]]
        
        if False:
            RDR_deep = getPredictedRDR(pred1)
            RDR_naive = getPredictedRDR(naive1)
            RDR_true = getPredictedRDR(profiles_full)

            error_RDR_deep = np.mean(np.abs(RDR - RDR_deep ))
            error_RDR_naive = np.mean(np.abs(RDR - RDR_naive ))
            error_RDR_true = np.mean(np.abs(RDR - RDR_true))

            #print (error_RDR_deep, error_RDR_naive, error_RDR_true)
            #quit()

            error_BAF_deep = np.mean(np.abs(   getPredictedBAF(pred1) -  getPredictedBAF(HAP)   ))
            error_BAF_naive = np.mean(np.abs(   getPredictedBAF(naive1) -  getPredictedBAF(HAP)   ))
            error_BAF_true = np.mean(np.abs(   getPredictedBAF(profiles_full) -  getPredictedBAF(HAP)   ))


            #error_RDR = [error_RDR_deep, error_RDR_naive, error_RDR_true]
            #error_BAF = [error_BAF_deep, error_BAF_naive, error_BAF_true]

            matrixErrors[simNum, 0, 0] = error_RDR_deep
            matrixErrors[simNum, 0, 1] = error_RDR_naive
            matrixErrors[simNum, 0, 2] = error_RDR_true
            matrixErrors[simNum, 1, 0] = error_BAF_deep
            matrixErrors[simNum, 1, 1] = error_BAF_naive
            matrixErrors[simNum, 1, 2] = error_BAF_true

    #np.savez_compressed('./data/simulation/' + folder1 + '/errorMatrix_new.npz', matrixErrors)
    #np.savez_compressed('./data/simulation/' + folder1 + '/clonesResult_new.npz', clonesAll)
    np.savez_compressed('./data/simulation/' + folder1 + '/trueErrors_new.npz', trueErrors)

    



#saveSimErrors()
#quit()


def showSpecialErrors():

    def getErrors(pred1, profiles_full):

        error_L1 = np.mean(np.abs(pred1 - profiles_full))

        return error_L1
    

    def smallCNAerrors(pred1, naive1, profiles_full, chr, b, L1=False):

        print (profiles_full.shape)

        _, chrBreak = np.unique(chr, return_index=True)
        chrBreak = np.concatenate((chrBreak,  np.zeros(1) + chr.shape[0] ))



        diff1 = np.sum(np.abs(profiles_full[:, 1:] - profiles_full[:, :-1]), axis=2)

        boolDiff = np.zeros((pred1.shape[0], pred1.shape[1]))

        for a in range(diff1.shape[0]):

            argDiff = np.argwhere(diff1[a] != 0)[:, 0] + 1
            argDiff = np.concatenate((argDiff, chrBreak))
            argDiff = np.unique(argDiff).astype(int)

            sizes1 = argDiff[1:] - argDiff[:-1]

            #argSize = np.argwhere(sizes1 <= 20)[:, 0]
            #argSize = np.argwhere(sizes1 <= 30)[:, 0]
            #argSize = np.argwhere(sizes1 <= 40)[:, 0]
            #argSize = np.argwhere(sizes1 <= 50)[:, 0]
            #argSize = np.argwhere(sizes1 <= 100)[:, 0]
            #argSize = np.argwhere(sizes1 <= 150)[:, 0]
            #argSize = np.argwhere(sizes1 <= 200)[:, 0]
            #argSize = np.argwhere(sizes1 <= 300)[:, 0]
            #argSize = np.argwhere(sizes1 <= 500)[:, 0]
            #print (sizes1.shape, argSize.shape)

            if b == 0:
                argSize = np.argwhere(sizes1 < 50)[:, 0]
            if b == 1:
                argSize = np.argwhere(np.logical_and(sizes1 >= 50, sizes1 < 200))[:, 0]
            if b == 2:
                argSize = np.argwhere(sizes1 >= 200)[:, 0]

            bool1 = np.zeros(boolDiff.shape[1]+1, dtype=int)
            bool1[argDiff[:-1][argSize] ] = bool1[argDiff[:-1][argSize] ] + 1
            bool1[argDiff[1:][argSize] ] = bool1[argDiff[1:][argSize] ] - 1
            bool1 = np.cumsum(bool1, axis=0)
            assert bool1[-1] == 0

            bool1 = bool1[:-1]
            boolDiff[a] = bool1

            #plt.plot(boolDiff[a])
            #plt.show()

        

        error1 = np.sum(np.abs(pred1 - profiles_full), axis=2).astype(float)
        error2 = np.sum(np.abs(naive1 - profiles_full), axis=2).astype(float)
        if not L1:
            error1[error1!=0] = 1
            error2[error2!=0] = 1

        percentBool = np.mean(boolDiff.astype(float))
        #print (percentBool)
        error1 = np.mean(error1 * boolDiff) / percentBool
        error2 = np.mean(error2 * boolDiff) / percentBool

        return error1, error2, percentBool




    if False:
        trueErrors = np.zeros((20, 3, 3))

        for simNum in range(0, 20):

            print (simNum)

            simNum1 = simNum + 10
        
            folder1 = '10x'
            chr = loadnpz('./data/' + folder1 + '/initial/chr_100k.npz')
            profiles_full = loadnpz('./data/simulation/' + folder1 + '/profiles_sim' + str(simNum1) + '.npz').astype(float)
            bins = loadnpz('./data/simulation/' + folder1 + '/bins_sim' + str(simNum1) + '.npz')
            pred1 = loadnpz('./data/simulation/' + folder1 + '/pred_sim' + str(simNum1) + '_new.npz').astype(float)
            naive1 = loadnpz('./data/simulation/' + folder1 + '/initialCNA_sim' + str(simNum1) + '.npz').astype(float)
            

            pred1 = pred1[:, bins]
            naive1 = naive1[:, bins]
            profiles_full = profiles_full[:, :, -1::-1]

            for b in range(3):
                error1, error2, percentSmall = smallCNAerrors(pred1, naive1, profiles_full, chr, b, L1=True)
                
                trueErrors[simNum, b, 0] = error1
                trueErrors[simNum, b, 1] = error2
                trueErrors[simNum, b, 2] = percentSmall

            #print (trueErrors)

            #error1_mod = np.mean(trueErrors[:simNum+1, 0] * trueErrors[:simNum+1, 2]) / np.mean(trueErrors[:simNum+1, 2])
            #error2_mod = np.mean(trueErrors[:simNum+1, 1] * trueErrors[:simNum+1, 2]) / np.mean(trueErrors[:simNum+1, 2])
            #print (error1_mod, error2_mod)


        #np.savez_compressed('./data/simulation/' + folder1 + '/altTrueErrors_small1.npz', trueErrors)
        np.savez_compressed('./data/simulation/' + folder1 + '/altTrueErrors_small_L1.npz', trueErrors)



    if False:

        trueErrors = loadnpz('./data/simulation/' + '10x' + '/altTrueErrors_small_L1.npz')



        errorDeep = np.mean(trueErrors[:, :, 0] * trueErrors[:, :, 2], axis=0) / np.mean(trueErrors[:, :, 2], axis=0)
        errorNaive = np.mean(trueErrors[:, :, 1] * trueErrors[:, :, 2], axis=0) / np.mean(trueErrors[:, :, 2], axis=0)

        #print (errorDeep)
        #print (errorNaive)
        #quit()


        #array = []
        #array.append([0.27177643413074914, 0.5883619734740563])
        #array.append([0.11954089176758326, 0.4932497485009198])
        #array.append([0.03296603497473248, 0.31973556026259986])

        #array = np.array(array)

        #array = 1.0 - array

        #sizes1 = [50, 30, 20, 15, 10, 5, 4, 3, 2]
        #sizes1 = np.array(sizes1)
        sizes1 = ['small (<5Mb)', 'medium($\geq$5Mb, <20Mb)', 'large($\geq$20Mb)']

        saveFile = './images/simulation/smallCNA_L1.pdf'

        plt.plot(sizes1, errorDeep)
        plt.plot(sizes1, errorNaive, color='lightblue')
        plt.scatter(sizes1, errorDeep)
        plt.scatter(sizes1, errorNaive, color='lightblue')
        plt.xlabel("CNA size")
        #plt.ylabel("accuracy")
        plt.ylabel("L1 error")
        plt.legend(['DeepCopy', 'NaiveCopy'])
        plt.gcf().set_size_inches(6, 4)
        plt.tight_layout()
        
        plt.savefig(saveFile)
        plt.show()
    
    

    if False:

        array = []
        array.append([0.27177643413074914, 0.5883619734740563])
        array.append([0.11954089176758326, 0.4932497485009198])
        array.append([0.03296603497473248, 0.31973556026259986])

        array = np.array(array)

        array = 1.0 - array

        #sizes1 = [50, 30, 20, 15, 10, 5, 4, 3, 2]
        #sizes1 = np.array(sizes1)
        sizes1 = ['small (<5Mb)', 'medium($\geq$5Mb, <20Mb)', 'large($\geq$20Mb)']

        saveFile = './images/simulation/smallCNA.pdf'

        plt.plot(sizes1, array[:, 0])
        plt.plot(sizes1, array[:, 1], color='lightblue')
        plt.scatter(sizes1, array[:, 0])
        plt.scatter(sizes1, array[:, 1], color='lightblue')
        plt.xlabel("CNA size")
        plt.ylabel("accuracy")
        plt.legend(['DeepCopy', 'NaiveCopy'])
        plt.gcf().set_size_inches(6, 4)
        plt.tight_layout()
        
        plt.savefig(saveFile)
        plt.show()


    

    if True:
        folder1 = folder1 = '10x'
        trueErrors = loadnpz('./data/simulation/' + folder1 + '/altTrueErrors_new.npz')
        trueErrors[:, :, 0] = trueErrors[:, :, 0] * 2 #correcting for mean rather than sum across the 2 haplotypes

        print (np.mean(trueErrors[:, :, 0], axis=0))
        quit()

        methodList = []
        errorList1 = []
        errorList2 = []
        errorList3 = []

        for a in range(20):
            methodList.append('DeepCopy')
            errorList1.append(trueErrors[a, 0, 0])
            errorList2.append(trueErrors[a, 0, 1])
            errorList3.append(trueErrors[a, 0, 2])
        for a in range(20):
            methodList.append('NaiveCopy')
            errorList1.append(trueErrors[a, 1, 0])
            errorList2.append(trueErrors[a, 1, 1])
            errorList3.append(trueErrors[a, 1, 2])




        plotData = {}
        plotData['index'] = methodList
        plotData['L1 error'] = errorList1
        plotData['total copy number error'] = errorList2
        plotData['BAF error'] = errorList3
        plotData['label1'] = methodList

        import pandas as pd

        

        df = pd.DataFrame(data=plotData)
        palette = ['blue', 'lightblue']


        ax = sns.boxplot(df, x='index', y='L1 error', palette=palette)
        plt.xticks(rotation = 90)
        plt.gcf().set_size_inches(4, 4)
        saveFile = './images/other/sim_L1.pdf'
        plt.tight_layout()
        plt.savefig(saveFile)
        plt.show()

        ax = sns.boxplot(df, x='index', y='total copy number error', palette=palette)
        plt.xticks(rotation = 90)
        plt.gcf().set_size_inches(4, 4)
        saveFile = './images/other/sim_total.pdf'
        plt.tight_layout()
        plt.savefig(saveFile)
        plt.show()


        ax = sns.boxplot(df, x='index', y='BAF error', palette=palette)
        plt.xticks(rotation = 90)
        plt.gcf().set_size_inches(4, 4)
        saveFile = './images/other/sim_BAFerror.pdf'
        plt.tight_layout()
        plt.savefig(saveFile)
        plt.show()




#showSpecialErrors()
#quit()



def simulationMeasurementError():

    

    folder1 = '10x'



    trueErrors = loadnpz('./data/simulation/' + folder1 + '/trueErrors_new.npz')

    plotData = {}
    plotData['accuracy'] = []
    plotData['L1 error'] = []
    plotData['method'] = []
    for a in range(20):
        plotData['accuracy'].append(trueErrors[a, 0, 0])
        plotData['L1 error'].append(trueErrors[a, 0, 1])
        plotData['method'].append('DeepCopy')
    for a in range(20):
        plotData['accuracy'].append(trueErrors[a, 1, 0])
        plotData['L1 error'].append(trueErrors[a, 1, 1])
        plotData['method'].append('NaiveCopy')

    print (np.mean(trueErrors, axis=0))
    quit()


    import pandas as pd
    df = pd.DataFrame(data=plotData)

    palette = ['blue', 'lightblue']
    saveFile1 = './images/simulation/accuracy.pdf'
    ax = sns.boxplot(data=df, x="method", y="accuracy", hue="method", showfliers=False, palette=palette)#, width=2)
    #plt.gcf().set_size_inches(3, 3.6)
    plt.gcf().set_size_inches(3.6, 3.6)
    plt.tight_layout()
    plt.savefig(saveFile1)
    plt.show()

    palette = ['blue', 'lightblue']
    #saveFile1 = './images/simulation/accuracy.pdf'
    saveFile1 = './images/simulation/L1.pdf'
    ax = sns.boxplot(data=df, x="method", y="L1 error", hue="method", showfliers=False, palette=palette)#, width=2)
    #plt.gcf().set_size_inches(3, 3.6)
    plt.gcf().set_size_inches(3.6, 3.6)
    plt.tight_layout()
    plt.savefig(saveFile1)
    plt.show()
    

    quit()





    matrixErrors = loadnpz('./data/simulation/' + folder1 + '/errorMatrix_new.npz')
    clonesAll = loadnpz('./data/simulation/' + folder1 + '/clonesResult_new.npz')
    #trueErrors = loadnpz('./data/simulation/' + folder1 + '/trueErrors_new.npz')


    plotData2 = {}
    plotData2['method'] = []
    plotData2['unique_profile'] = []
    
    
    plotData3 = {}
    plotData3['method'] = []
    plotData3['accuracy'] = []

    N = 200
    cloneSizeMatrix = np.zeros((20, 3, N))

    accuracyMean = []
    accuracyMean_Naive = []


    


    uniqueProfileNumbers = np.zeros((20, 3))
    for simNum in range(20):
        uniqueProfileNumbers[simNum, 0] = np.unique(clonesAll[simNum, 0]).shape[0]
        uniqueProfileNumbers[simNum, 1] = np.unique(clonesAll[simNum, 1]).shape[0]
        uniqueProfileNumbers[simNum, 2] = np.unique(clonesAll[simNum, 2]).shape[0]

        _, counts1 = np.unique(clonesAll[simNum, 0], return_counts=True)
        _, counts2 = np.unique(clonesAll[simNum, 1], return_counts=True)
        _, counts3 = np.unique(clonesAll[simNum, 2], return_counts=True)
        counts1 = np.sort(counts1)[-1::-1]
        counts2 = np.sort(counts2)[-1::-1]
        counts3 = np.sort(counts3)[-1::-1]

        #print (counts1)
        #quit()

        N1, N2, N3 = min(N, counts1.shape[0]), min(N, counts2.shape[0]), min(N, counts3.shape[0])

        cloneSizeMatrix[simNum, 0, :N1] = counts1[:N1]
        cloneSizeMatrix[simNum, 1, :N2] = counts2[:N2]
        cloneSizeMatrix[simNum, 2, :N3] = counts3[:N3]





        if True:#simNum < 7:
            plotData2['method'].append('DeepCopy')
            plotData2['method'].append('NaiveCopy')
            plotData2['method'].append('Ground Truth')

            plotData2['unique_profile'].append(np.unique(clonesAll[simNum, 0]).shape[0])
            plotData2['unique_profile'].append(np.unique(clonesAll[simNum, 1]).shape[0])
            plotData2['unique_profile'].append(np.unique(clonesAll[simNum, 2]).shape[0])


            plotData3['method'].append('DeepCopy')
            plotData3['method'].append('NaiveCopy')
            plotData3['accuracy'].append(1.0 - np.mean(trueErrors[simNum, 0]))
            plotData3['accuracy'].append(1.0 - np.mean(trueErrors[simNum, 1]))

            accuracyMean.append( 1.0 - np.mean(trueErrors[simNum, 0]) )
            accuracyMean_Naive.append( 1.0 - np.mean(trueErrors[simNum, 1]) )



    #accuracyMean = np.array(accuracyMean)
    #accuracyMean_Naive = np.array(accuracyMean_Naive)
    #print ('accuracy ours', np.mean(accuracyMean[:]))
    #print ('accuracy Naive', np.mean(accuracyMean_Naive[:]))
    #quit()



    if False:
        saveFile = './images/simulation/cloneSizes.pdf'
        for simNum in range(7):
            plt.plot( np.arange(N)+1, cloneSizeMatrix[simNum, 0], color='blue', alpha=0.5)
            plt.plot( np.arange(N)+1, cloneSizeMatrix[simNum, 1], color='lightblue', alpha=0.5)
            plt.plot( np.arange(N)+1, cloneSizeMatrix[simNum, 2], color='red', alpha=0.5)
        plt.legend(['DeepCopy', 'NaiveCopy', 'Ground Truth'])
        plt.yscale('log')
        plt.xlabel('clone number')
        plt.ylabel('clone size (number of cells)')
        plt.gcf().set_size_inches(3, 3)
        plt.tight_layout()
        
        plt.savefig(saveFile)
        plt.show()
        #quit()
    

    unique_profile = [np.mean( uniqueProfileNumbers[:, 0] )  ,   np.mean( uniqueProfileNumbers[:, 1] ),   np.mean( uniqueProfileNumbers[:, 2] )]


    if True:
        #min1, max1 = np.min(uniqueProfileNumbers[:, 2]), np.max(uniqueProfileNumbers[:, 2])
        #

        min1 = np.min(uniqueProfileNumbers) * 0.9
        max1 = np.max(uniqueProfileNumbers) * 1.1
        line1 = [min1, max1]

        print (np.mean(np.abs( (uniqueProfileNumbers[:, 1] / uniqueProfileNumbers[:, 2]) - 1 )))
        #quit()

        #print (scipy.stats.spearmanr(uniqueProfileNumbers[:, 2] , uniqueProfileNumbers[:, 1]))
        print (scipy.stats.pearsonr(uniqueProfileNumbers[:, 2] , uniqueProfileNumbers[:, 1]))
        print (scipy.stats.pearsonr(uniqueProfileNumbers[:, 2] , uniqueProfileNumbers[:, 0]))
        quit()
        
        
        plt.scatter( uniqueProfileNumbers[:, 2] , uniqueProfileNumbers[:, 0], color='blue')
        plt.scatter( uniqueProfileNumbers[:, 2] , uniqueProfileNumbers[:, 1], color='lightblue')
        #plt.scatter( uniqueProfileNumbers[10:, 2] , uniqueProfileNumbers[10:, 0], color='blue', marker='x')
        #plt.scatter( uniqueProfileNumbers[10:, 2] , uniqueProfileNumbers[10:, 1], color='lightblue', marker='x')
        plt.plot( line1, line1, color='red' , label='ground truth' )


        
        

        
        plt.ylabel('predicted # of unique profiles')
        plt.xlabel('true # of unique profiles')
        #plt.legend(['DeepCopy no WGD', "NaiveCopy no WGD", 'DeepCopy WGD', "NaiveCopy WGD", 'ground truth line'])
        #plt.legend(['DeepCopy', "NaiveCopy", 'ground truth line'])
        plt.legend(['DeepCopy', "NaiveCopy"])

        plt.ylim(min1, max1)
        plt.xlim(min1, max1)

        saveFile = './images/simulation/uniqueCopyNumberScatter.pdf'

        plt.gcf().set_size_inches(4.5, 4)
        plt.tight_layout()
        
        plt.yscale('log')
        plt.xscale('log')
        
        plt.savefig(saveFile)
        plt.show()

        quit()
    #quit()

    error_RDR = np.mean(matrixErrors[:, 0], axis=0)
    error_BAF = np.mean(matrixErrors[:, 1], axis=0)

    #print (matrixErrors[:7, 0])
    #quit()



    x = ['DeepCopy', 'NaiveCopy', 'Ground Truth']

    


    plotData = {}
    plotData['index'] = x
    plotData['error_RDR'] = error_RDR
    plotData['error_BAF'] = error_BAF
    plotData['unique_profile'] = unique_profile
    plotData['label1'] = x

    import pandas as pd

    

    df = pd.DataFrame(data=plotData)

    df2 = pd.DataFrame(data=plotData2)
    df3 = pd.DataFrame(data=plotData3)


    

    if True:
        palette = ['blue', 'lightblue']
        saveFile1 = './images/simulation/accuracy.pdf'
        ax = sns.boxplot(data=df3, x="method", y="accuracy", hue="method", showfliers=False, palette=palette)#, width=2)
        #plt.gcf().set_size_inches(2.5, 3)
        plt.gcf().set_size_inches(3, 3.6)
        plt.tight_layout()
        plt.savefig(saveFile1)
        plt.show()
    
    

    

    if False:

        palette = ['blue', 'lightblue', 'red']

        ax = sns.barplot(df, x='index', y='unique_profile', label='label1', palette=palette)
        #ax = sns.boxplot(data=df2, x="method", y="unique_profile", hue="method", showfliers=False, palette=palette)
        #ax.legend([],[], frameon=False)
        

        
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
        #ax2.legend(labels=labels, handles=handles, loc='middle center')
        ax2.legend(labels=labels, handles=handles, loc='center')

        plt.gcf().set_size_inches(4, 3)

        
        #saveFile = './images/plots/error_' + folder1 + '.pdf'

        saveFile = './images/simulation/measureError.pdf'

        plt.tight_layout()
        plt.savefig(saveFile)
        plt.show()


    quit()

    
    




simulationMeasurementError()
quit()


def plotSimHeatmap():

    from scipy.cluster.hierarchy import linkage

    #predict_file = './data/simulation/' + folder1 + '/pred_sim' + str(simNum) + '_new.npz'

    #initialCNA_file = './data/simulation/' + folder1 + '/initialCNA_sim' + str(simNum) + '.npz'
    #bins_file = './data/simulation/' + folder1 + '/bins_sim' + str(simNum) + '.npz'

    #maybe 11 is good for heatmap. Though we seem to have 1 wrong WGD. 
    #12 is also reasonable. Less noisy.



    #simNum = 14
    #simNum = 24

    for simNum in [11]:#range(10, 20):
        folder1 = '10x'
        chr2 = loadnpz('./data/' + folder1 + '/initial/chr_100k.npz')
        chr = loadnpz('./data/simulation/' + folder1 + '/chr_avg_sim' + str(simNum) + '.npz')
        bins = loadnpz('./data/simulation/' + folder1 + '/bins_sim' + str(simNum) + '.npz')

        naive1 = loadnpz( './data/simulation/' + folder1 + '/initialCNA_sim' + str(simNum) + '.npz')
        pred1 = loadnpz( './data/simulation/' + folder1 + '/pred_sim' + str(simNum) + '_new.npz')
        profiles_full = loadnpz('./data/simulation/' + folder1 + '/profiles_sim' + str(simNum) + '.npz')

        print (np.unique(uniqueProfileMaker(naive1)).shape)
        print (np.unique(uniqueProfileMaker(pred1)).shape)
        print (np.unique(uniqueProfileMaker(profiles_full)).shape)

        quit()
        
        

        linkage_matrix = linkage(profiles_full.reshape((profiles_full.shape[0],  profiles_full.shape[1] * 2 ))  , method='ward', metric='euclidean')

        haplotypePlotter(naive1[:, bins].astype(int), doCluster=True, chr=[], withLinkage=[linkage_matrix], saveFile='./images/simExample/sim_' + str(simNum) + '_naive.png', plotSize=[6, 4])
        haplotypePlotter(pred1[:, bins].astype(int), doCluster=True, chr=[], withLinkage=[linkage_matrix], saveFile='./images/simExample/sim_' + str(simNum) + '_deep.png', plotSize=[6, 4])
        haplotypePlotter(profiles_full.astype(int), doCluster=True, chr=[], withLinkage=[linkage_matrix], saveFile='./images/simExample/sim_' + str(simNum) + '_true.png', plotSize=[6, 4])



#plotSimHeatmap()
#quit()


#from skbio import TreeNode
#tree1 = TreeNode.read(["((a,b),(c,d));"])
#tree2 = TreeNode.read(["(((a,b),c),d);"])
#print (tree1.compare_rfd(tree2))
#quit()


def simulationParsimony():


    #simNum1 = 10

    #errorList = []
    folder1 = '10x'

    if False:

        errorMatrix = np.zeros((20, 3))

        for simNum1 in range(10, 30):

            treeBoth = []

            for methodNum in range(1):#range(3):
                
                method1 = ['deep', 'truth', 'naive'][methodNum]
                if method1 == 'deep':
                    chr2 = loadnpz('./data/simulation/' + folder1 + '/chr_avg_sim' + str(simNum1) + '.npz')
                    pred1 = loadnpz('./data/simulation/' + folder1 + '/pred_sim' + str(simNum1) + '_new.npz')
                if method1 == 'naive':
                    chr2 = loadnpz('./data/simulation/' + folder1 + '/chr_avg_sim' + str(simNum1) + '.npz')
                    pred1 = loadnpz('./data/simulation/' + folder1 + '/initialCNA_sim' + str(simNum1) + '.npz')
                if method1 == 'truth':
                    pred1 = loadnpz('./data/simulation/' + folder1 + '/profiles_sim' + str(simNum1) + '.npz')        
                    chr2 = loadnpz('./data/' + folder1 + '/initial/chr_100k.npz')

                pred1[pred1 >= 19] = 19
                

            
                #profiles_full = loadnpz('./data/simulation/' + folder1 + '/profiles_sim' + str(simNum1) + '.npz')
                #bins = loadnpz('./data/simulation/' + folder1 + '/bins_sim' + str(simNum1) + '.npz')
                #pred1 = loadnpz('./data/simulation/' + folder1 + '/pred_sim' + str(simNum1) + '.npz')
                #naive1 = loadnpz('./data/simulation/' + folder1 + '/initialCNA_sim' + str(simNum1) + '.npz')

                #naive1[naive1 >= 19] = 19
                #profiles_full[profiles_full >= 19] = 19

                #pred1 = profiles_full
                #chr2 = chr
                

                
                distMatrix = calcDiffMatrix(pred1, chr2)
                #np.savez_compressed('./temp/distMatrix.npz', distMatrix)
                #distMatrix = loadnpz('./temp/distMatrix.npz')

                #plt.imshow(distMatrix)
                #plt.show()

                #print (distMatrix)
                #quit()

                clades, tree1 = getTree(distMatrix)

                tree1 = modifyTree(tree1)

                treeBoth.append(tree1)

                #print (tree1)
                treeInternal_choice, cladeSizes, pairList, pairListLength, errors, treeWithLength = runParsimony(tree1, pred1, chr2)


                errorMatrix[simNum1 - 10, methodNum] = errors[0]

                print (errorMatrix)


            #np.savez_compressed('./temp/treeBoth.npz', np.array(treeBoth)  )

            #from skbio import TreeNode
            #tree1 = TreeNode.read([treeBoth[0]])
            #tree2 = TreeNode.read([treeBoth[1]])
            #print (tree1.compare_rfd(tree2))

            

            np.savez_compressed('./data/simulation/10x/parsimonyError_new.npz', errorMatrix)

    #quit()

    if True:
        
        errorMatrix = loadnpz('./data/simulation/10x/parsimonyError.npz')
        errorMatrix_new = loadnpz('./data/simulation/10x/parsimonyError_new.npz')
        errorMatrix[:, 0] = errorMatrix_new[:, 0]

        print (errorMatrix[1])
        print (errorMatrix_new[1])
        quit()

        #print (scipy.stats.spearmanr(errorMatrix[:, 0], errorMatrix[:, 1]))
        print (scipy.stats.pearsonr(errorMatrix[:, 0], errorMatrix[:, 1]))
        print (scipy.stats.pearsonr(errorMatrix[:, 2], errorMatrix[:, 1]))

        #print (np.mean(errorMatrix, axis=0))
        quit()

        #print (errorMatrix)

        #quit()

        #print (errorMatrix)

        #errorMatrix[errorMatrix > 10000] = 10000

        plt.scatter(errorMatrix[:, 1], errorMatrix[:, 0], color='blue')
        plt.scatter(errorMatrix[:, 1], errorMatrix[:, 2], color='lightblue')
        line1 = np.sort(errorMatrix[:, 1])
        plt.plot(line1, line1, color='red')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel("ground truth parsimony")
        plt.ylabel('predicted parsimony')
        plt.legend(['DeepCopy', 'NaiveCopy'])

        plt.gcf().set_size_inches(4.5, 4)
        plt.tight_layout()

        plt.savefig('./images/simulation/parsimony.pdf')
        plt.show()


    #Deep
    #[array([ 488., 1070.]), array([ 552., 1384.]), array([465., 977.]), array([ 904., 2559.]), array([ 787., 2003.]), array([1011., 2877.]), array([1390., 4118.]), array([1645., 5218.]), array([1682., 5325.]), array([1677., 5192.])]
    #488, 552, 465, 904, 787,      1011, 1390, 1645, 1682, 1677
    #95, 312, 211, 735, 561,       798, 1213, 1644, 1684, 1676

    #Naive
    #[array([28934., 34994.]), array([5918., 8391.]), array([2738., 4045.]), array([1580., 3427.]), array([1627., 3671.]), array([1625., 4156.]), array([2207., 5433.]), array([1898., 5994.]), array([1829., 5647.]), array([2000., 6122.])]


    #Ground Truth
    #[array([   95., 48989.]), array([   312., 169072.]), array([   211., 125155.]), array([   735., 507395.]), array([   561., 396498.]), array([   798., 575877.]), array([  1213., 936236.]), array([   1644., 1275213.]), array([   1684., 1299684.]), array([   1676., 1315184.])]


#simulationParsimony()
#quit()


def saveACTtreeParsimony():


    
    folder1_list = ['ACT10x', 'TN3']

    if True:

        errorMatrix = np.zeros((20, 3))

        for folder1 in folder1_list:

            chr = loadnpz('./data/' + folder1 + '/binScale/chr_avg.npz')

            for method1 in ['deep', 'naive']:
                
                if True:
                    if method1 == 'deep':
                        pred1 = loadnpz('./data/' + folder1 + '/model/pred_good.npz')
                    if method1 == 'naive':
                        pred1 = loadnpz('./data/' + folder1 + '/binScale/initialCNA.npz')

                    
                    pred1[pred1>=19] = 19

                    
                if False:
                    distMatrix = calcDiffMatrix(pred1, chr)

                    np.savez_compressed('./data/comparison/ACT/distMatrix_' + folder1 + '_' + method1 + '.npz', distMatrix)


                    clades, tree1 = getTree(distMatrix)
                    tree1 = modifyTree(tree1)

                    np.savez_compressed('./data/comparison/ACT/tree_' + folder1 + '_' + method1 + '.npz', np.array([tree1]))

                tree1 = loadnpz('./data/comparison/ACT/tree_' + folder1 + '_' + method1 + '.npz')[0]

                #treeBoth.append(tree1)

                #print (tree1)
                treeInternal_choice, cladeSizes, pairList, pairListLength, errors, treeWithLength = runParsimony(tree1, pred1, chr)


                #errorMatrix[simNum1 - 10, methodNum] = errors[0]

                np.savez_compressed('./data/comparison/ACT/parsimony_' + folder1 + '_' + method1 + '.npz', errors)
                np.savez_compressed('./data/comparison/ACT/treeWithLength_' + folder1 + '_' + method1 + '.npz', np.array([treeWithLength]))






#saveACTtreeParsimony()
#quit()


def plotACTtree():

    

    #errors = loadnpz('./data/comparison/ACT/parsimony_' + 'TN3' + '_' + 'naive' + '.npz')
    #print (errors[0])
    #quit()

    #folder1 = 'ACT10x'
    folder1 = 'TN3'
    #method1 = 'deep'
    method1 = 'naive'

    tree1 = loadnpz('./data/comparison/ACT/treeWithLength_' + folder1 + '_' + method1 + '.npz')[0]


    tree1 = tree1 + ';'

    #print (len(tree1.split(')')))
    #print (len(tree1.split('(')))
    #quit()


    tree1 = tree1.replace('root', '')
    


    miniTree = False
    
    
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
        if method1 == 'naive':
            if folder1 == 'ACT10x':
                ts.scale = 1.5 #Proportional to height!
            if folder1 == 'TN3':
                ts.scale = 1.3

    #ts.show_scale = True
    #ts.show_branch_length = True
    ts.min_leaf_separation = 0
    ts.branch_vertical_margin = 0
    t.show(tree_style=ts)
    quit()


#plotACTtree()
#quit()


def weirdTreePlotting():



    from copy import deepcopy
    from scipy.cluster.hierarchy import ClusterNode

    def to_linkage(tree: ClusterNode, n_clusters):
        tree = deepcopy(tree)
        linkage = []
        # Remove all nodes that merge two leaves, and place them on the linkage list
        def prune(tr):
            if tr.left.is_leaf() and tr.right.is_leaf():
                linkage.append((tr.left.id, tr.right.id, tr.dist, tr.count))
                tr.id = n_clusters + len(linkage) - 1
                tr.left = None
                tr.right = None
            if tr.left is not None and not tr.left.is_leaf():
                prune(tr.left)
            if tr.right is not None and not tr.right.is_leaf():
                prune(tr.right)

        while not tree.is_leaf():
            prune(tree)

        return np.array(linkage, dtype="float64")


    import numpy as np
    from matplotlib import pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    from sklearn.cluster import AgglomerativeClustering
    from sklearn.datasets import load_iris


    def plot_dendrogram(model, **kwargs):
        # Create linkage matrix and then plot the dendrogram

        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)


    iris = load_iris()
    X = iris.data

    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

    model = model.fit(X)
    print (model)
    plt.title("Hierarchical Clustering Dendrogram")
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, truncate_mode="level", p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()


    quit()
