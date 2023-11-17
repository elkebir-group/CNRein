
#CNA.py

import numpy as np


#import matplotlib.pyplot as plt
import time
import scipy
from scipy import stats
from scipy.special import logsumexp



import pandas as pd
#%matplotlib inline
#%config InlineBackend.figure_format = 'svg'

from tqdm import tqdm

#from scaler import *
#from RLCNA import *










def loadnpz(name, allow_pickle=False):

    #This simple function more easily loads in compressed numpy files.

    if allow_pickle:
        data = np.load(name, allow_pickle=True)
    else:
        data = np.load(name)
    data = data.f.arr_0
    return data



def tweakBAF(x):

    delta = 0.01

    x = (x * (1.0 - (delta * 2) ) ) + delta

    return x


def easyUniqueValMaker(X):
    inverse1 = uniqueValMaker(X)
    _, index1 = np.unique(inverse1, return_index=True)
    return X[index1]


def fastAllArgwhere(ar):
    ar_argsort = np.argsort(ar)
    ar1 = ar[ar_argsort]
    _, indicesStart = np.unique(ar1, return_index=True)
    _, indicesEnd = np.unique(ar1[-1::-1], return_index=True) #This is probably needless and can be found from indicesStart
    indicesEnd = ar1.shape[0] - indicesEnd - 1
    return ar_argsort, indicesStart, indicesEnd


def paddedCumSum(x):

    x = np.cumsum(x)
    x = np.concatenate((  np.zeros(1), x  ))

    return x



def uniqueValMaker(X):

    _, vals1 = np.unique(X[:, 0], return_inverse=True)

    for a in range(1, X.shape[1]):

        #vals2 = np.copy(X[:, a])
        #vals2_unique, vals2 = np.unique(vals2, return_inverse=True)
        vals2_unique, vals2 = np.unique(X[:, a], return_inverse=True)

        vals1 = (vals1 * vals2_unique.shape[0]) + vals2
        _, vals1 = np.unique(vals1, return_inverse=True)

    return vals1



def rebin(data, M, doPytorch=False):

    if len(data.shape) == 1:
        #M = 10
        N = data.shape[0] // M
        data = data[:(N*M)]
        data = data.reshape( (N, M) )
        if doPytorch:
            data = torch.mean(data, axis=1)
        else:
            data = np.mean(data, axis=1)
        return data

    if len(data.shape) == 2:
        #M = 10
        N = data.shape[1] // M
        data = data[:, :(N*M)]
        data = data.reshape( (data.shape[0], N, M) )
        if doPytorch:
            data = torch.mean(data, axis=2)
        else:
            data = np.mean(data, axis=2)
        return data
    
    if len(data.shape) == 3:
        #M = 10
        N = data.shape[1] // M
        data = data[:, :(N*M)]
        data = data.reshape( (data.shape[0], N, M, data.shape[2]) )
        if doPytorch:
            data = torch.mean(data, axis=2)
        else:
            data = np.mean(data, axis=2)
        return data


def uniqueProfileMaker(X):

    inverse1 = uniqueValMaker(X.reshape((X.shape[0], X.shape[1] * X.shape[2])))

    return inverse1




def haplotypePlotter(predCNA, doCluster=False, withLinkage=[], saveFile='', chr=[], plotSize=[], vertLine=[]):
    
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import seaborn as sns

    from matplotlib.colors import LinearSegmentedColormap
    
    shape1 = predCNA.shape

    predCNA = predCNA.reshape((predCNA.shape[0]*predCNA.shape[1], 2))
    argBad = np.argwhere(np.sum(predCNA, axis=1) > 6)[:, 0]
    predCNA[argBad, 0] = 6
    predCNA[argBad, 1] = 0
    
    
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

    #print ('hi')
    #plt.imshow(predCNA)
    #plt.show()



    if len(chr) != 0:
        chr0 = chr[0]
        chr_palette = ['#525252', '#969696', '#cccccc']
        chr_colors = [ chr_palette[chr0[a]%3] for a in range(chr0.shape[0])  ]

        #g = sns.clustermap( predCNA, col_cluster=False, row_cluster=True, linewidths=0.0, cmap=cmap, cbar_pos=None, yticklabels=False, xticklabels=False, col_colors=chr_colors)
    else:
        chr_colors = []
    

    
    if doCluster:
        if len(withLinkage) == 0:
            if len(chr) != 0:
                g = sns.clustermap( predCNA, col_cluster=False, row_cluster=True, linewidths=0.0, cmap=cmap, cbar_pos=None, yticklabels=False, xticklabels=False, col_colors=chr_colors)
            else:
                g = sns.clustermap( predCNA, col_cluster=False, row_cluster=True, linewidths=0.0, cmap=cmap, cbar_pos=None, yticklabels=False, xticklabels=False)
        else:
            if len(chr) != 0:
                g = sns.clustermap( predCNA, col_cluster=False, row_cluster=True, row_linkage=withLinkage[0], linewidths=0.0, cmap=cmap, cbar_pos=None, yticklabels=False, xticklabels=False, col_colors=chr_colors)
            else:
                g = sns.clustermap( predCNA, col_cluster=False, row_cluster=True, row_linkage=withLinkage[0], linewidths=0.0, cmap=cmap, cbar_pos=None, yticklabels=False, xticklabels=False)

        g.ax_row_dendrogram.set_visible(False)

    else:
        #g = sns.heatmap( predCNA, cmap=cmap)
        g = plt.imshow( predCNA, cmap=cmap)

    #plt.show()
    #quit()


    if len(chr) != 0:

        chr = chr[0]

        corners = []
        prev = 0
        for a in range(chr.shape[0]-1):
            if chr[a] != chr[a+1]:
                corners.append((a, a+1))
        corners.append((chr.shape[0]-1, chr.shape[0]))
        
        ax = g.ax_heatmap

        ticks = []
        for o in corners:
            #print (o)
            #print (np.append(ax.get_xticks(), int(float(o[1] + o[0] + 1) / 2.0)))
            ax.set_xticks(np.append(ax.get_xticks(), int(float(o[1] + o[0] + 1) / 2.0)))
            ticks.append(chr[o[0]]+1)
        
        ax.set_xticklabels(ticks, rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    if len(vertLine) != 0:

        for a in range(predCNA.shape[1] // vertLine[0]):
            plt.axvline(x=(a+1) * vertLine[0], color='black', linestyle=':', linewidth=2)

    plt.tight_layout()

    #plt.show()
    

    if len(plotSize) > 0:
        plt.gcf().set_size_inches(plotSize[0], plotSize[1])

    if saveFile == '':
        plt.show()
    else:
        plt.savefig(saveFile)






def calcDiffMatrix(predCNA, chr):

    def insertBoundaries(predCNA, chr):

        predCNA2 = np.zeros( (predCNA.shape[0], predCNA.shape[1] + 100, 2) , dtype=int)
        count1 = 0
        for a in range(22):
            args1 = np.argwhere(chr == a)[:, 0]
            size1 = args1.shape[0] + 2
            predCNA2[:, count1+1:count1+size1-1] = predCNA[:, args1]
            count1 += size1
        predCNA2 = predCNA2[:, :count1]
        return predCNA2


    def convertCNAdiff(predCNA, chr):

        predCNA = insertBoundaries(predCNA, chr)

        predCNA2 = np.zeros( (predCNA.shape[0], predCNA.shape[1], 2, 2) , dtype=int)
        sizeList = np.zeros(( predCNA.shape[0], 2 ), dtype=int)
        
        for a in range(predCNA.shape[0]):
            pred1 = predCNA[a, :, 0]
            pred2 = predCNA[a, :, 1]
            diff1 = pred1[1:] - pred1[:-1]
            diff2 = pred2[1:] - pred2[:-1]


            #print (np.max(diff1))
            #quit()

            args1 = np.argwhere(diff1 != 0)[:, 0]
            args2 = np.argwhere(diff2 != 0)[:, 0]
            diff1 = diff1[args1]
            diff2 = diff2[args2]

            predCNA2[a, :args1.shape[0], 0, 0] = args1
            predCNA2[a, :args1.shape[0], 0, 1] = diff1
            predCNA2[a, :args2.shape[0], 1, 0] = args2
            predCNA2[a, :args2.shape[0], 1, 1] = diff2

            sizeList[a, 0] = args1.shape[0]
            sizeList[a, 1] = args2.shape[0]

        
        maxCount = int(np.max(sizeList))
        predCNA2 = predCNA2[:, :maxCount]

        #plt.plot(sizeList)
        #plt.show()
        #quit()



        return predCNA2, sizeList
    

    def getDiff1(pred1, pred2):


        #print (pred1)
        #quit()

        dict1 = {}
        for a in range(pred1.shape[0]):
            arg1 = int(pred1[a, 0])
            val1 = int(pred1[a, 1])
            if arg1 in dict1:
                dict1[arg1] += val1
            else:
                dict1[arg1] = val1
        
        for a in range(pred2.shape[0]):
            arg1 = int(pred2[a, 0])
            val1 = int(pred2[a, 1])
            if arg1 in dict1:
                dict1[arg1] -= val1
            else:
                dict1[arg1] = -1 * val1
        
        keys = dict1.keys()
        sum1 = 0
        for key1 in keys:
            sum1 += abs(dict1[key1])
            #if abs(dict1[key1]) != 0:
            #    sum1 += 1

        return sum1


    def getDist(pred1_A, pred1_B, pred2_A, pred2_B, double1, double2):

        if double1:
            pred1_A[:, 1] = pred1_A[:, 1] * 2
            pred1_B[:, 1] = pred1_B[:, 1] * 2
        if double2:
            pred2_A[:, 1] = pred2_A[:, 1] * 2
            pred2_B[:, 1] = pred2_B[:, 1] * 2

        dist1 = getDiff1(pred1_A, pred2_A)
        dist2 = getDiff1(pred1_B, pred2_B)
        distFull = (dist1 + dist2) / 2 #/2 is part of the ZCNT distance. 
        return distFull




    normal1 = (predCNA[:1] * 0) + 1

    predCNA = np.concatenate((predCNA , normal1))

    #meanList = np.mean(predCNA.astype(float), axis=(1, 2))
    
    predCNA, sizeList = convertCNAdiff(predCNA, chr)

    distMatrix = np.zeros((predCNA.shape[0], predCNA.shape[0]), dtype=int)

    for a in tqdm(range(predCNA.shape[0]-1)):
        #print (a)
        count1_A = sizeList[a, 0]
        count1_B = sizeList[a, 1]
        pred1_A = predCNA[a, :count1_A, 0]
        pred1_B = predCNA[a, :count1_B, 1]
        for b in range(a+1, predCNA.shape[0]):
            count2_A = sizeList[b, 0]
            count2_B = sizeList[b, 1]
            pred2_A = predCNA[b, :count2_A, 0]
            pred2_B = predCNA[b, :count2_B, 1]

            distFull1 = getDist(np.copy(pred1_A), np.copy(pred1_B), np.copy(pred2_A), np.copy(pred2_B), False, False)
            distFull2 = getDist(np.copy(pred1_A), np.copy(pred1_B), np.copy(pred2_A), np.copy(pred2_B), True, False) + 1
            distFull3 = getDist(np.copy(pred1_A), np.copy(pred1_B), np.copy(pred2_A), np.copy(pred2_B), False, True) + 1
            distFull = min(distFull1, min(distFull2, distFull3))
            #distFull = distFull1


            #print (distFull)

            

            distMatrix[a, b] = distFull
            distMatrix[b, a] = distFull


    return distMatrix







def getTree(data):

    import dendropy
    from skbio import DistanceMatrix
    from skbio.tree import nj
    import sys

    sys.setrecursionlimit(10000) 





    ids = list(np.arange(len(data)).astype(str))
    #ids[0] = 'banana'
    dm = DistanceMatrix(data, ids)
    
    tree = nj(dm)

    #tree = tree.root_at( str( len(data)-1 ) )
    #print (tree)
    #tree = tree.root_at(  '254') 

    #tree1 = str(tree)

    #print (tree1)
    #tree1 = reRootTree(tree1, ['447'])
    rootString = str(len(data)-1)
    tree = tree.find(rootString).unrooted_copy()
    tree = tree.root_at(rootString)
    
    tree1 = str(tree)

    #np.savez_compressed('./temp/tree.npz', np.array([tree1] ) )

    #else:

    #only keep between ( and :    or  , and :

    #data = loadnpz('./data/comparison/tree/dist_' + folder2 + '_ours.npz')
    #tree1 = loadnpz('./temp/tree.npz')[0]

    tree2 = ''
    lastOne = ''
    for a in range(len(tree1)):
        if tree1[a] in ['(', ':', ')', ',']:
            lastOne = tree1[a]
            if tree1[a] != ':':
                tree2 = tree2 + tree1[a]
            
        elif lastOne in ['(', ',']:
            tree2 = tree2 + tree1[a]
    
    
    args1 = np.argwhere(np.array(list(tree2)) == '(' )[:, 0]

    matrix1 = np.zeros(( args1.shape[0], len(data) ))

    for a in range(args1.shape[0]):
        value1 = 1
        index1 = args1[a]
        while value1 != 0:
            index1 += 1
            if tree2[index1] == '(':
                value1 += 1
            if tree2[index1] == ')':
                value1 -= 1
            
        string2 = tree2[args1[a]:index1]
        string2 = string2.replace('(', ',')
        string2 = string2.replace(')', ',')
        while ',,' in string2:
            string2 = string2.replace(',,', ',')
        if string2[0] == ',':
            string2 = string2[1:]
        if string2[-1] == ',':
            string2 = string2[:-1]
        subset1 = string2.split(',')
        subset1 = np.array(subset1).astype(int)

        matrix1[a, subset1] = 1

    matrix1 = matrix1[:, :-1]

    


    return matrix1, tree1








def calculateZNT(vector1, vector2):

        #print (vector1.shape, vector2.shape)
        diff1 = vector1[1:] - vector1[:-1]
        diff2 = vector2[1:] - vector2[:-1]

        error1 = np.sum(np.abs( diff2 -  diff1 )) 

        return error1


def runParsimony(tree1, predCNA, chr):



    def insertBoundaries(predCNA, chr):

        originalBool = np.zeros(predCNA.shape[1] + 100, dtype=int)
        predCNA2 = np.zeros( (predCNA.shape[0], predCNA.shape[1] + 100, 2) , dtype=int)
        count1 = 0
        for a in range(22):
            args1 = np.argwhere(chr == a)[:, 0]
            size1 = args1.shape[0] + 2
            predCNA2[:, count1+1:count1+size1-1] = predCNA[:, args1]
            originalBool[count1+1:count1+size1-1] = 1
            count1 += size1
        predCNA2 = predCNA2[:, :count1]
        originalBool = originalBool[:count1]
        return predCNA2, originalBool



    def getSmallClade(tree1):
        finalPair = [0, 0]
        lastPos = 0
        for a in range(len(tree1)):
            if tree1[a] in ['(', ')']:
                
                if (tree1[lastPos] == '(') and (tree1[a] == ')'):
                    finalPair = [lastPos, a]
                lastPos = a
        return finalPair


    def vectorSumMaxer(vectorSum):
        for b in range(predCNA.shape[1]):
            vectorSum_0 = vectorSum[b, 0]
            vectorSum_1 = vectorSum[b, 1]
            vectorSum_0[vectorSum_0 != np.max(vectorSum_0)] = 0
            vectorSum_0[vectorSum_0 == np.max(vectorSum_0)] = 1
            vectorSum_1[vectorSum_1 != np.max(vectorSum_1)] = 0
            vectorSum_1[vectorSum_1 == np.max(vectorSum_1)] = 1
            vectorSum[b, 0] = vectorSum_0
            vectorSum[b, 1] = vectorSum_1
        return vectorSum


    def new_vectorSumMaxer(vectorSum):

        max1 = np.max(vectorSum, axis=2)
        vectorSum = vectorSum - max1.reshape((max1.shape[0], max1.shape[1], 1)) + 1
        vectorSum[vectorSum<=0] = 0
        
        return vectorSum


    def vectorChooser(vectorSum):

        choiceVector = np.zeros((vectorSum.shape[0], 2), dtype=int)

        errors = 0
        
        for b in range(vectorSum.shape[0]):
            vectorSum_0 = vectorSum[b, 0]
            vectorSum_1 = vectorSum[b, 1]

            max0 = np.max(vectorSum_0)
            max1 = np.max(vectorSum_1)

            arg1 = np.argwhere(vectorSum_0 == max0 )[0, 0]
            arg2 = np.argwhere(vectorSum_1 == max1 )[0, 0]

            choiceVector[b, 0] = arg1
            choiceVector[b, 1] = arg2

            vectorSum[b, 0, :] = 0
            vectorSum[b, 1, :] = 0
            vectorSum[b, 0, arg1] = 1
            vectorSum[b, 1, arg2] = 1
        
        return vectorSum, choiceVector

    def new_vectorChooser(vectorSum):

            
        choiceVector = np.argsort(vectorSum*-1, axis=2)[:, :, 0]


        choiceVector_0 = choiceVector[:, 0]
        choiceVector_1 = choiceVector[:, 1]

        arange1 = np.arange(choiceVector_0.shape[0])

        vectorSum[:] = 0
        vectorSum[arange1, 0, choiceVector_0] = 1
        vectorSum[arange1, 1, choiceVector_1] = 1

        return vectorSum, choiceVector


    predCNA, originalBool = insertBoundaries(predCNA, chr)

    #print (originalBool.shape)
    #print (predCNA.shape)
    #quit()


    tree_original = tree1

    #print (predCNA.shape)


    treeLeafs = np.zeros( (predCNA.shape[0], predCNA.shape[1], 2, 20)  , dtype=int)
    for a in range(predCNA.shape[0]):
        treeLeafs[a, np.arange(predCNA.shape[1]), 0, predCNA[a, :, 0]  ] = 1
        treeLeafs[a, np.arange(predCNA.shape[1]), 1, predCNA[a, :, 1]  ] = 1

    treeInternal = np.zeros(treeLeafs.shape, dtype=int)
    treeInternal_choice = np.zeros(predCNA.shape, dtype=int)

    cladeSizes = np.zeros(predCNA.shape[0], dtype=int)

    pairList = []





    #print (tree1)
    #print ('')
    count1 = 0
    while ',' in tree1:
        #print (tree1)
        finalPair = getSmallClade(tree1)
        pairNow = tree1[finalPair[0]+1:  finalPair[1] ]

        pairList.append(  pairNow )

        #print (tree1[finalPair[0]:  finalPair[1]+1 ])
        pairNow = pairNow.split(',')

        if len(pairNow) == 3:
            pairNow.remove( str(int(predCNA.shape[0]))  )

        cladeSize1 = 0
        
        value1, value2 = pairNow[0], pairNow[1]
        if value1[0] == 'C':
            vector1 = treeInternal[int(value1[1:])]
            cladeSize1 += cladeSizes[int(value1[1:])]
        else:
            vector1 = treeLeafs[int(value1)]
            cladeSize1 += 1
        if value2[0] == 'C':
            vector2 = treeInternal[int(value2[1:])]
            cladeSize1 += cladeSizes[int(value2[1:])]
        else:
            vector2 = treeLeafs[int(value2)]
            cladeSize1 += 1

        vector1[0, 0, 0] = 1
        vector1[0, 0, 2] = 0

        vectorSum = vector1 + vector2


        #vectorSum_old = vectorSumMaxer(np.copy(vectorSum))

        #print (np.sum(np.abs(   vectorSum_new -vectorSum_old  )))
        #quit()

        vectorSum = new_vectorSumMaxer(vectorSum)
        

        #print (vectorSum.shape)
        #print (treeInternal[count1].shape)

        cladeSizes[count1] = cladeSize1

        treeInternal[count1] = vectorSum
        
        newName = 'C' + str(count1)
        tree1 = tree1[:finalPair[0]] + newName + tree1[finalPair[1]+1:]
        #print (tree1)

        #print (len(tree1) - len(tree1.replace(')', '')))
        #quit()

        count1 += 1

    #plt.hist(cladeSizes, bins=100)
    #plt.show()

    treeInternal_choice = treeInternal_choice[:len(pairList)]

    #Deciding the root:    
    vectorSum, choiceVector = vectorChooser(treeInternal[len(pairList)-1])
    treeInternal[len(pairList)-1] = vectorSum

    errorTotal = 0
    errorTotal_dumb = 0

    treeWithLength = tree1

    pairListLength = []

    for a0 in range(len(pairList)):
        a = len(pairList) - 1 - a0

        name1 = 'C' + str(a)

        pairNow = pairList[a]
        pairNow_str = '(' + pairNow + ')'

        tree1 = tree1.replace(name1, pairNow_str)

        pairNow = pairNow.split(',')
        if len(pairNow) == 3:
            pairNow.remove( str(int(predCNA.shape[0]))  )

        valueParent = a
        valueChild1 = pairNow[0]
        valueChild2 = pairNow[1]

        vectorParent = treeInternal[valueParent]

        assert np.sum(vectorParent) == vectorParent.shape[0] * 2

        if valueChild1[0] == 'C':
            vector1 = treeInternal[int(valueChild1[1:])]
        else:
            vector1 = treeLeafs[int(valueChild1)]
        if valueChild2[0] == 'C':
            vector2 = treeInternal[int(valueChild2[1:])]
        else:
            vector2 = treeLeafs[int(valueChild2)]

        vectorSum1 = (vector1 * 2) + vectorParent
        vectorSum2 = (vector2 * 2) + vectorParent


        #A, B = new_vectorChooser(np.copy(vectorParent))
        #C, D = vectorChooser(np.copy(vectorParent))

        #print (np.mean(np.abs(A - C)))
        #print (np.mean(np.abs(B - D)))
        #quit()

        _, parentChoice = new_vectorChooser(vectorParent)
        vectorSum1, choiceVector1 =  new_vectorChooser(vectorSum1)
        vectorSum2, choiceVector2 = new_vectorChooser(vectorSum2)


        if valueChild1[0] == 'C':
            treeInternal[int(valueChild1[1:])] = vectorSum1
            treeInternal_choice[int(valueChild1[1:])] = choiceVector1
        else:
            treeLeafs[int(valueChild1)] = vectorSum1
        if valueChild2[0] == 'C':
            treeInternal[int(valueChild2[1:])] = vectorSum2
            treeInternal_choice[int(valueChild2[1:])] = choiceVector2
        else:
            treeLeafs[int(valueChild2)] = vectorSum2
        

        diffparent = parentChoice[1:] - parentChoice[:-1]
        diff1 = choiceVector1[1:] - choiceVector1[:-1]
        diff2 = choiceVector2[1:] - choiceVector2[:-1]

        error1 = np.sum(np.abs( diffparent -  diff1 )) / 2
        error2 = np.sum(np.abs( diffparent -  diff2 )) / 2
        error1_dumb = np.sum(np.abs(  parentChoice -  choiceVector1  ))
        error2_dumb = np.sum(np.abs(  parentChoice -  choiceVector2  ))

        errorTotal += error1
        errorTotal += error2

        errorTotal_dumb += error1_dumb
        errorTotal_dumb += error2_dumb

        pairDistString = '(' + valueChild1 + ':' + str(error1) + ',' + valueChild2 + ':' + str(error2) + ')'
        #print (pairDistString)

        pairListLength.append(pairDistString)

        treeWithLength = treeWithLength.replace(name1, pairDistString)

        #print (error1, error2)
        #print (errorTotal)
        #print (errorTotal_dumb)

        #print (treeWithLength)

    #quit()

    #print (errorTotal, errorTotal_dumb)

    pairListLength = np.array(pairListLength)[-1::-1]

    errors = np.array([errorTotal, errorTotal_dumb])

    assert tree1 == tree_original

    pairList = np.array(pairList)

    #print (treeInternal_choice.shape)
    treeInternal_choice = treeInternal_choice[:, originalBool == 1]
    #print (treeInternal_choice.shape)


    return treeInternal_choice, cladeSizes, pairList, pairListLength, errors, treeWithLength





def modifyTree(tree1):

    tree2 = ''
    lastOne = ''
    for a in range(len(tree1)):
        if tree1[a] in ['(', ':', ')', ',']:
            lastOne = tree1[a]
            if tree1[a] != ':':
                tree2 = tree2 + tree1[a]
            
        elif lastOne in ['(', ',']:
            tree2 = tree2 + tree1[a]
    
    return tree2







def simplifyClonesTree(tree1):


        
        def getSmallClade(tree1, startPos):
            finalPair = []
            lastPos = -1
            #for a in range(startPos, len(tree1)):
            a = startPos
            while (len(finalPair) == 0) and (a < len(tree1)):
                if tree1[a] in ['(', ')']:
                    
                    if lastPos != -1:
                        if (tree1[lastPos] == '(') and (tree1[a] == ')'):
                            finalPair = [lastPos, a]

                    lastPos = a
                a += 1
            return finalPair
        
        
        #pairOld = [-1, -1]

        startPos = 0
        pairNew = [-2, -2]
        while len(pairNew) > 0:#pairOld[0]!=pairNew[0]:

            #pairOld = pairNew
            pairNew = getSmallClade(tree1, startPos)

            #print (pairNew)

            if len(pairNew) > 0:
            

                

                pairNow_str = tree1[pairNew[0]+1:  pairNew[1] ]
                pairNow = pairNow_str.split(',')

                #print (pairNow)
                #quit()

                value1 = float(pairNow[0].split(':')[1])
                value2 = float(pairNow[1].split(':')[1])
                if value1 + value2 == 0:
                    
                    leaf1 = pairNow[0].split(':')[0]
                    #leaf2 = int(pairNow2[1].split(':')[0])

                    pairNow_str = '(' + pairNow_str + ')'
                    
                    tree1 = tree1.replace(pairNow_str, leaf1)

                else:

                    startPos = pairNew[0] + 1
                    #print ("keep")
        
        #print (tree1)
        #quit()
        return tree1



def findTreeFromFile(outLoc, runEasy=True, fileMatrix=[], fileChr=''):

    import os

    if runEasy:
        pred1 = loadnpz(outLoc + '/model/pred_now.npz')
        chr = loadnpz(outLoc + '/binScale/chr_avg.npz')

        cellNames = loadnpz(outLoc + '/initial/cellNames.npz')
        np.savetxt(outLoc + '/tree/cellBarcodes.txt', cellNames, fmt='%s')

        #np.savetxt(outLoc + '/tree/pred_hap1.csv', pred1[:, :, 0], delimiter=',')
        #np.savetxt(outLoc + '/tree/pred_hap2.csv', pred1[:, :, 1], delimiter=',')
        #np.savetxt(outLoc + '/tree/chr_avg.csv', chr, delimiter=',')

        #print ("Done1")

        #quit()

    else:
        if '.npz' in fileChr:
            chr = loadnpz(fileChr)
        if '.npy' in fileChr:
            chr = np.load(fileChr)
        if '.csv' in fileChr:
            chr = np.loadtxt(fileChr, dtype=str, delimiter=',').astype(float).astype(int)
        chr = chr - np.min(chr) #starting count at zero not 1. 
        
        if len(fileMatrix) == 1:
            fileMatrix = fileMatrix[0]
            if '.npz' in fileMatrix:
                pred1 = loadnpz(fileMatrix)
            if '.npy' in fileMatrix:
                pred1 = np.load(fileMatrix)
        
        else:
            #String to float to int is a way of avoiding issues indpendent of the way it is stored in the csv. 
            pred_0 = np.loadtxt(fileMatrix[0], dtype=str, delimiter=',').astype(float).astype(int)
            pred_1 = np.loadtxt(fileMatrix[1], dtype=str, delimiter=',').astype(float).astype(int)
            pred_0 = pred_0.reshape((pred_0.shape[0], pred_0.shape[1], 1))
            pred_1 = pred_1.reshape((pred_1.shape[0], pred_1.shape[1], 1))

            pred1 = np.concatenate((pred_0, pred_1), axis=2)
            del pred_0
            del pred_1

    


    
    
    
    pred1[pred1>=19] = 19

    print ('Tree prediction — Step 1/3: Calculating distance matrix... ')

    if True:
        distMatrix = calcDiffMatrix(pred1, chr)
        command1 = 'mkdir ' + outLoc + '/tree'
        os.system(command1) 
        np.savez_compressed(outLoc + '/tree/distMatrix.npz', distMatrix)
    else:
        distMatrix = loadnpz(outLoc + '/tree/distMatrix.npz')

    print ('Tree prediction — Step 2/3: Calculating tree... ', end='')
    clades, tree1 = getTree(distMatrix)
    tree1 = modifyTree(tree1)

    #np.savez_compressed(outLoc + '/tree/tree.npz', np.array([tree1]))
    np.savetxt(outLoc + '/tree/tree.txt', np.array([tree1]), fmt='%s')
    print ("Done")


    print ('Tree prediction — Step 3/3: Calculating edge lengths and parsimony... ', end='')
    treeInternal_choice, cladeSizes, pairList, pairListLength, errors, treeWithLength = runParsimony(tree1, pred1, chr)
    

    #print (errors[0])

    #np.savez_compressed(outLoc + '/tree/parsimony.npz', errors)
    #np.savez_compressed(outLoc + '/tree/treeWithLength.npz', np.array([treeWithLength]))

    np.savetxt(outLoc + '/tree/parsimony.txt', errors[:1].astype(int).astype(str), fmt='%s'  )
    np.savetxt(outLoc + '/tree/treeWithLength.txt', np.array([treeWithLength]), fmt='%s')
    print ("Done")





#fileMatrix = [outLoc + '/tree/pred_hap1.csv', outLoc + '/tree/pred_hap2.csv']
#fileChr = outLoc + '/tree/chr_avg.csv'
#findTreeFromFile(outLoc, runEasy=False, fileMatrix=fileMatrix, fileChr=fileChr)
#quit()