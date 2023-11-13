import os
import subprocess

#os.system('ls')
#quit()

import numpy as np
from pathlib import Path
import sys
import time

#from raw import countReads

load1 = True
if len(sys.argv) == 2:
    if sys.argv[1] == 'noLoad':
        load1 = False


if load1:
    import pandas as pd
    import pysam
    from pysam import VariantFile
    #import matplotlib.pyplot as plt
    #from scipy.special import logsumexp


#from process import runProcessFull




#DLP bam uses chromosomes like "1" whereas ACT uses like "chr1"


def loadnpz(name, allow_pickle=False):

    #This simple function more easily loads in compressed numpy files.

    if allow_pickle:
        data = np.load(name, allow_pickle=True)
    else:
        data = np.load(name)
    data = data.f.arr_0
    return data




def systemPrint(command1):

    #print (command1)
    #os.system(command1)
    subprocess.run(command1, shell=True)


def makeAllDirectories(name):


    command1 = 'mkdir ' + name
    systemPrint(command1)

    systemPrint(command1 + '/temp')
    systemPrint(command1 + '/counts')
    systemPrint(command1 + '/info')
    systemPrint(command1 + '/phased')
    systemPrint(command1 + '/phasedCounts')
    systemPrint(command1 + '/readCounts')
    systemPrint(command1 + '/readCounts/pos')
    systemPrint(command1 + '/initial')
    systemPrint(command1 + '/binScale')
    systemPrint(command1 + '/model')
    

    
    for chrNum in range(1, 22+1):
        systemPrint(command1 + '/readCounts/pos/' + str(chrNum) )



#makeAllDirectories('TN3')
#quit()


def runParallel(commandList, outLoc):

    #Different operating systems have different issues in terms of running commands and waiting for them to complete
    #This simple "hack" ensures the command has completed before moving on. 

    randomInt = np.random.randint(1000000000)

    tempLoc = outLoc + '/temp/'
    tempFileList = []
    commandList2 = []
    for a in range(len(commandList)):
        tempFile = tempLoc + str(randomInt) + '_' + str(a) + '.txt'
        command2 = ' ( ' + commandList[a] + ' ; touch ' + tempFile + ' ) '
        commandList2.append(command2)
        tempFileList.append(tempFile)
    
    commandFull = '&'.join(commandList2)

    systemPrint(commandFull)
    #os.popen(commandFull).read()
    #print (commandFull)

    wait1 = True
    while wait1:
        #print ('')
        wait1 = False
        for file1 in tempFileList:
            if not os.path.exists(file1):
                wait1 = True
                #print ('nope')
                #print (file1)

        if wait1:
            time.sleep(10)
    
    for file1 in tempFileList:
        commandRemove = 'rm ' + file1
        systemPrint(commandRemove)
    






def addReadGroup(dataName):

    samtoolsLocation = '/scratch/data/stefan/stefanSoftware/samtools_1.17/bin/samtools'
    #samtoolsLocation = 'samtools'

    bamFolder = './data/' + dataName + '/bam/split/'

    
    bamList = os.listdir(bamFolder)
    barcodes = []
    for file1 in bamList:
        if ('.bam' in file1) and not ('.bai' in file1):
            barcode = file1.replace('.bam', '')
            barcodes.append(barcode)


    print (len(barcodes))
    quit()

    
    for a in range(len(barcodes) ):#range(len(barcodes)):

        print (a, len(barcodes))

        barcode = barcodes[a]
        bamFile = bamFolder + barcode + '.bam'
        modifiedBamFile = bamFolder + barcode + '_mod.bam'
        #print (bamFile)
        #quit()

        command1 = 'nice -10 ' + samtoolsLocation + ' addreplacerg -@ 20 -r "@RG\tID:' + barcode + '\tSM:' + barcode + '"' +  ' -o ' + modifiedBamFile + ' ' +  bamFile
        print (command1)
        quit()
        systemPrint(command1)

        command2 = 'mv ' + modifiedBamFile + ' ' + bamFile
        systemPrint(command2)

        #quit()
    

#addReadGroup('ACT10x')
#quit()

def mergeBams(dataName):

    #if dataName == '10x': #Just for now
    #    folder1 = './data/' + dataName + '/bam/resampleSplit/'
    #else:
    folder1 = './data/' + dataName + '/bam/split/'

    locationData0 = os.listdir(folder1)
    locationData = []
    for loc1 in locationData0:
        if ('.bam' in loc1) and not ('.bai' in loc1):
            if '_mod' in loc1:
                locationData.append(folder1 + loc1)
    

    #locationData = locationData[:30] #Todo Remove 

    finalFile = './data/' + dataName + '/bam/merged/FullMerge.bam'
    #finalFile = './data/' + dataName + '/bam/merged/FullMerge_fake.bam'

    #print (locationData)
    #quit()
    #N = 10
    N = 1000

    bamListFile = './data/' + dataName + '/info/bamAll.txt'
    np.savetxt(bamListFile, locationData, fmt='%s')
    #quit()


    if len(locationData) <= N: 
        
        samtoolsLocation = '/scratch/data/stefan/stefanSoftware/samtools_1.17/bin/samtools'
        
        #command1 = 'nice -10  samtools merge --threads 20 -b ' + bamListFile + ' ' + finalFile
        command1 = 'nice -10  ' + samtoolsLocation + ' merge --threads 20 -b ' + bamListFile + ' ' + finalFile
        systemPrint(command1)

        quit()

        command2 = 'samtools index ' + finalFile
        systemPrint(command2)

    else:

        
        M = ((len(locationData) - 1) // N) + 1

        chunkLocations = []
        for a in range(0, M):
            print (a, M)
            locationDataNow = locationData[(a*N):]
            locationDataNow = locationDataNow[:N]

            outputFile = './data/' + dataName + '/bam/merged/remergedChunk_' + str(a) + '.bam'

            #print (len(locationDataNow))

            bamListFile = './data/' + dataName + '/info/rebamChunk_' + str(a) + '.txt'
            np.savetxt(bamListFile, locationDataNow, fmt='%s')
        
            #command1 = 'nice -10 samtools merge --threads 20 -b ' + bamListFile + ' ' + outputFile
            command1 = 'samtools merge --threads 20 -b ' + bamListFile + ' ' + outputFile

            chunkLocations.append(outputFile)

            systemPrint(command1)


        bamListFile = './data/' + dataName + '/info/bamChunks.txt'
        np.savetxt(bamListFile, chunkLocations, fmt='%s')

        command1 = 'samtools merge --threads 20 -b ' + bamListFile + ' ' + finalFile

        systemPrint(command1)

        for a in range(len(chunkLocations)):

            outputFile = chunkLocations[a]

            command1 = 'rm ' + outputFile
            systemPrint(command1)


        command2 = 'samtools index -@ 20 ' + finalFile
        systemPrint(command2)
        

#mergeBams('ACT10x')
#quit()




def renameSample10x():


    folder1 = './data/10x/bam/split/'
    folder2 = './data/10x/bam/resampleSplit/'
    #folder2 = './data/10x/bam/attempt2Resplit/'

    

    
    fnames_exist = os.listdir(folder2)
    fnames_exist = np.array(fnames_exist)

    fnames0 = os.listdir(folder1)
    fnames0 = np.array(fnames0)

    
    
    fnames0 = fnames0[np.isin(fnames0,  fnames_exist) == False]

    fnames = []
    count1 = -1
    newNames = []
    for a in range(len(fnames0)):
        #print (a, len(fnames))
        #print (fnames0[a])
        if ('.bam' in fnames0[a]) and not ('bai' in fnames0[a]):
            fnames.append(fnames0[a])

            

    
    #chunkNum = sys.argv[1]
    #chunkNum = int(chunkNum)

    
    #max1 = min(50*chunkNum, len(fnames))

    #for a in range(2000, len(fnames)):#len(fnames)):
    #for a in range(50*(chunkNum - 1), max1 ):
    for a in range( len(fnames) ):
        
        print (a)
        name1 = folder1 + fnames[a]

        barcode = name1.split('/')[-1]
        barcode = barcode.split('_')[-1].split('.')[0]
        #print (barcode)
        #CB_CGCACCAGCACCCCG.bam

        bamOutput = folder2 + fnames[a]

        #print (bamOutput, name1)
        #quit()
        #command1 = 'nice -10 samtools addreplacerg -@ 20 -r "@RG\tID:' + barcode + '"' +  ' -o ' + bamOutput + ' ' +  name1

        samtoolsLocation = '/scratch/data/stefan/stefanSoftware/samtools_1.17/bin/samtools'

        #command1 = 'nice -10 ' + samtoolsLocation + ' addreplacerg -@ 20 -r "@RG\tID:' + barcode + '"' +  ' -o ' + bamOutput + ' ' +  name1
        command1 = 'nice -10 ' + samtoolsLocation + ' addreplacerg -@ 20 -r "@RG\tID:' + barcode + '\tSM:' + barcode + '"' +  ' -o ' + bamOutput + ' ' +  name1
        
        


        systemPrint(command1)

        #quit()

        command2 = 'samtools index ' + bamOutput
        systemPrint(command2)



#renameSample10x()
#quit()



def picardCommands():

    #chrNum = '1'
    #folder1 = './data/reference/vcf_copy/'

    #command3 = 'java -Djava.io.tmpdir=/scratch/data/stefan/doShapeit/tmp  -jar ' + folder1 + 'picard.jar LiftoverVcf I=' + folder1 + 'chr' + str(chrNum) + '_rename.vcf O=' + folder1 + 'chr' + str(chrNum) + '_hg38_rename.vcf CHAIN=' + folder1 + 'hg19ToHg38.over.chain.gz REJECT=' + folder1 + 'rejected_variants_chr' + str(chrNum) + '.vcf R=' + folder1 + 'hg38.fa'
    #systemPrint(command3)
    #quit()

    #conda activate bio #Important to keep this! 
    #Also, comment out all "import" that creates errors in conda bio envirement

    #chrNum = 22

    #1, 2, 3, 4 is problematic 

    for chrNum in ['X']:##range(1, 22):
        #if not chrNum in [14, 15]:

        folder1 = './data/reference/vcf_copy/'

        command1 = 'cp ./data/reference/vcf/chr' + str(chrNum) + '_rename.vcf.gz ./data/reference/vcf_copy/chr' + str(chrNum) + '_rename.vcf.gz'
        systemPrint(command1)
        #quit()
        command2 = 'gunzip ./data/reference/vcf_copy/chr' + str(chrNum) + '_rename.vcf.gz'
        systemPrint(command2)
        #quit()

        command3 = 'java -Djava.io.tmpdir=/scratch/data/stefan/doShapeit/tmp -jar ' + folder1 + 'picard.jar LiftoverVcf I=' + folder1 + 'chr' + str(chrNum) + '_rename.vcf O=' + folder1 + 'chr' + str(chrNum) + '_hg38_rename.vcf CHAIN=' + folder1 + 'hg19ToHg38.over.chain.gz REJECT=' + folder1 + 'rejected_variants_chr' + str(chrNum) + '.vcf R=' + folder1 + 'hg38.fa'
        systemPrint(command3)

        command4 = 'bcftools view -I ' + folder1 + 'chr' + str(chrNum) + '_hg38_rename.vcf -O z -o ' + folder1 + 'chr' + str(chrNum) + '_hg38_rename.vcf.gz'
        systemPrint(command4)

        command5 = 'bcftools index -t ' + folder1 + 'chr' + str(chrNum) + '_hg38_rename.vcf.gz'
        systemPrint(command5)

        #quit()
    
    #quit()
    #java -jar picard.jar CreateSequenceDictionary R=hg38.fa O=hg38.dict
    #java -jar picard.jar LiftoverVcf I=chr22_rename.vcf O=chr22_hg38_rename.vcf CHAIN=hg19ToHg38.over.chain.gz REJECT=rejected_variants.vcf R=hg38.fa
    #bcftools view -I chr22_hg38_rename.vcf -O z -o chr22_hg38_rename.vcf.gz
    True

#picardCommands()
#quit()


def loadReference(chrNum):

    vname = 'v5b'
    if chrNum == 'X':
        vname = 'v1c'

    tbiPart = ''
    originalName = 'ALL.chr' + chrNum + '.phase3_shapeit2_mvncall_integrated_' + vname + '.20130502.genotypes.vcf.gz' + tbiPart

    link1 = 'http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/' + originalName
    command1 = 'wget ' + link1
    systemPrint(command1)
    command2 = 'mv ./' + originalName +  ' ./data/reference/vcf/chr' + chrNum + '.vcf.gz' + tbiPart
    systemPrint(command2)
    #quit()

    tbiPart = '.tbi'
    originalName = 'ALL.chr' + chrNum + '.phase3_shapeit2_mvncall_integrated_' + vname + '.20130502.genotypes.vcf.gz' + tbiPart
    link1 = 'http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/' + originalName
    command1 = 'wget ' + link1
    systemPrint(command1)
    command2 = 'mv ./' + originalName +  ' ./data/reference/vcf/chr' + chrNum + '.vcf.gz' + tbiPart
    systemPrint(command2)


#loadReference('X')
#quit()
#for a in range(4, 5):
#    chr1 = str(a)
#    loadReference(chr1)
#quit()


def renameReference(chrNum):

    newRef = './data/reference/vcf/chr' + chrNum + '.vcf.gz'
    renameRef = './data/reference/vcf/chr' + chrNum + '_rename.vcf.gz'

    #newRef = './data/reference/vcf_copy/chr' + chrNum + '_hg38.vcf.gz' 
    #renameRef = './data/reference/vcf_copy/chr' + chrNum + '_rename_hg38.vcf.gz'

    command1 = 'bcftools annotate --rename-chrs ./data/reference/chr_name_conv.txt ' + newRef + ' -Oz -o ' +  renameRef
    systemPrint(command1)

    command2 = 'bcftools index -t ' + renameRef
    systemPrint(command2)



#chr1 = 3
#renameReference('X')
#quit()



def loadGeneticMaps():

    hgNum = '37'

    for chr1 in range(1, 23):
        chrNum = str(chr1)
        command1 = 'wget https://github.com/odelaneau/shapeit5/blob/main/resources/maps/b' + hgNum  + '/chr' + chrNum + '.b' + hgNum + '.gmap.gz'
        systemPrint(command1)
        command2 = 'mv ./chr' + chrNum + '.b' + hgNum + '.gmap.gz ' + ' ./data/reference/genetic_maps_hg' + hgNum + '/chr' + chrNum + '.b' + hgNum + '.gmap.gz'
        systemPrint(command2)
        #quit()

#loadGeneticMaps()
#quit()



def renameFasta():

    reference1 = './data/reference/hg19.fa'
    # Using readlines()
    file1 = open(reference1, 'r')
    Lines = file1.readlines()

    reference2 = './data/reference/hg19_unname.fa'



    for a in range(len(Lines)):
        if a % 1000 == 0:
            print (a // 1000, len(Lines) // 1000)
        #line1 = Lines[a]
        if 'chr' in Lines[a]:
            Lines[a] = Lines[a].replace('chr', '')


    file2 = open(reference2, 'w')
    file2.writelines((Lines))
    file2.close()


    file1.close()

    quit()

#renameFasta()
#quit()



def findCombinedCounts(bamLoc, refLoc, outLoc, refGenome):

    #if useHG38:
    #    refName = 'hg38'
    #else:
    #    refName = 'hg19'

    commandList1 = []

    commandList_1 = []
    commandList_2 = []

    checkFiles = []

    command1 = ''
    for a in range(1, 22+1):
        chrNum = str(a)

        #if a != 1:
        #    command1 = command1 + ' & '
        #commandMini = 'python3 piler.py combined ' + bamLoc + ' ' + outLoc + ' ' + refLoc + ' ' + chrNum + ' ' + refGenome
        #systemPrint(commandMini)
        #command1 = command1 + commandMini

        countsFile = outLoc + '/counts/ignore_chr' + chrNum + '.vcf.gz'

        if refGenome == 'hg38':
            refFasta = refLoc + '/hg38.fa'
            renameRef = refLoc + '/vcf_hg38/chr' + chrNum + '.vcf.gz'
        else:
            refFasta = refLoc + '/hg19.fa'
            renameRef = refLoc + '/vcf_hg19/chr' + chrNum + '.vcf.gz'

        checkFiles.append(countsFile + '.tbi')


        command1 = 'bcftools mpileup --ignore-RG -Ou -R ' + renameRef + ' -f ' + refFasta + ' ' + bamLoc + ' | bcftools call -vmO z -o ' + countsFile
        commandList_1.append(command1)

        command2 = 'bcftools index -t ' + countsFile
        commandList_2.append(command2)

        command1 = ' ( ' + command1 + ' ; ' + command2 + ' ) '

        commandList1.append(command1)

    

    if False:
        commandFull1 = ' & '.join(commandList1)
        systemPrint(commandFull1)

        #runParallel(commandList1, outLoc)
        
        
        #commandFull2 = ' & '.join(commandList2)
        #systemPrint(commandFull2)

        #for a in range(len(commandList2)):
        #    command2 = commandList2[a]
        #    systemPrint(command2)

        wait1 = True
        while wait1:
            print ('')
            wait1 = False
            for file1 in checkFiles:
                if not os.path.exists(file1):
                    print (file1)
                    print ('nope')
                    wait1 = True

            if wait1:
                time.sleep(10)




        #systemPrint(command1)
        True

    runParallel(commandList_1, outLoc)
    runParallel(commandList_2, outLoc)






def runPhasing(outLoc, refGenome, refLoc):


    #if useHG38:
    #    refName = 'hg38'
    #else:
    #    refName = 'hg19'

    #command1 = ''
    commandList1 = []
    commandList2 = []
    for a in range(1, 22+1):
        chrNum = str(a)

        #commandMini = 'python3 piler.py phase ' + dataName + ' ' + chrNum + ' ' + refName
        #systemPrint(commandMini)

        #chrNum = str(chrNum0)


        shapeItLocation = 'shapeit4'
        phasedFile = outLoc + '/phased/phased_chr' + chrNum + '.bcf'
        countsFile = outLoc + '/counts/ignore_chr' + chrNum + '.vcf.gz'
        if refGenome == 'hg38':
            renameRef = refLoc + '/vcf_hg38/chr' + chrNum + '.vcf.gz'
            chrName = 'chr' + chrNum
        else:
            renameRef = refLoc + '/vcf_hg19/chr' + chrNum + '.vcf.gz'
            chrName = chrNum

        command6_input = shapeItLocation + ' --input ' + countsFile + ' --reference ' + renameRef +  ' --region ' + chrName 
        command6_output = ' --output ' + phasedFile + ' --thread 8'
        
        command6 = command6_input + command6_output
        #systemPrint(command6)
        commandList1.append(command6)

        command7 = 'bcftools index -t ' + str(phasedFile) 

        #systemPrint(command7)
        commandList2.append(command7)

    
    if False:
        commandFull1 = ' & '.join(commandList1)
        systemPrint(commandFull1)
        #commandFull2 = ' & '.join(commandList2)
        #systemPrint(commandFull2)

        for a in range(len(commandList2)):
            command2 = commandList2[a]
            systemPrint(command2)

    runParallel(commandList1, outLoc)
    runParallel(commandList2, outLoc)





def findSubsetCounting(outLoc):


    #chrNum = sys.argv[1]


    for chrNum0 in range(1, 22+1):#range(1, 22+1):
        chrNum = str(chrNum0)


        phasedFile = outLoc + '/phased/phased_chr' + chrNum + '.bcf'
        restrictedFile = outLoc + '/phased/restricted_chr' + chrNum + '.vcf.gz'

        

        bcf_in = VariantFile(phasedFile)  # auto-detect input format



        bcf_out = VariantFile(restrictedFile, 'w', header=bcf_in.header)

        count1 = 0
        count2 = 0
        for rec in bcf_in.fetch():

            recStr = str(rec)

            #print (recStr)
            #quit()
            recStr = recStr.replace('\n', '')
            #print ([recStr])
            recStr = recStr.split('\t')[-1]
            recStr = recStr.split('|')

            #print (recStr)

            count1 += 1

            if recStr[0] != recStr[1]:
                count2 += 1
                bcf_out.write(rec)

            #print (count1, count2)

        bcf_out.close()

        #print (restrictedFile)
        #print (count1, count2)

        

        command2 = 'bcftools index -t ' + restrictedFile

        systemPrint(command2)

        #quit()

    #phased_genotypes = read_bcf_phased_genotypes(phasedFile)
    #phased_genotypes.set_index(['chromosome', 'position', 'ref', 'alt'], inplace=True)

#findSubsetCounting('ACT10x')
#quit()



def findIndividualCounts(bamLoc, refLoc, outLoc, refGenome):

    #if useHG38:
    #    refName = 'hg38'
    #else:
    #    refName = 'hg19'

    #command1 = ''

    commandList1 = []
    for a in range(1, 22+1):
        chrNum = str(a)

        #if a != 1:
        #    command1 = command1 + ' & '

        #commandMini = 'python3 piler.py seperate ' + dataName + ' ' + chrNum + ' ' + refName
        #command1 = command1 + commandMini

        if refGenome == 'hg38':
            refFasta = refLoc + '/hg38.fa'
        else:
            refFasta = refLoc + '/hg19.fa'

        
        restrictedFile = outLoc + '/phased/restricted_chr' + chrNum + '.vcf.gz'
        countsFile = outLoc + '/counts/seperates_chr' + chrNum + '.vcf.gz'
        
        command1 = 'bcftools mpileup --annotate FORMAT/AD -Ou -R ' + restrictedFile + ' -f ' + refFasta + ' ' + bamLoc + ' | bcftools call -vmO z -o ' + countsFile
        #systemPrint(command1)
        commandList1.append(command1)
    
    #commandFull1 = ' & '.join(commandList1)
    #systemPrint(commandFull1)

    runParallel(commandList1, outLoc)
    


#findIndividualCounts()
#quit()




def read_bcf_phased_genotypes(bcf_filename):
    """ Read in a shapeit4 generated BCF file and return dataframe of phased alleles.

    Parameters
    ----------
    bcf_filename : str
        BCF file produced by shapeit4

    Returns
    -------
    pandas.DataFrame
        table of phased alleles
    """
    phased_genotypes = []


    

    for r in pysam.VariantFile(bcf_filename, 'r'):
        for alt in r.alts:
            chromosome = r.chrom
            position = r.pos
            ref = r.ref

            #print (str(r))

            #print (len(r.samples))

            #print (r.samples[0].items()[0][1])
            #print (r.samples[1].items()[0][1])
            #print (r.samples[2].items()[0][1])
            #print (r.samples[3].items()[0][1])
            #print (r.samples[4].items()[0][1])

            assert len(r.samples) == 1
            gt_infos = r.samples[0].items()

            assert len(gt_infos) == 1
            assert gt_infos[0][0] == 'GT'
            allele1, allele2 = gt_infos[0][1]

            phased_genotypes.append([chromosome, position, ref, alt, allele1, allele2])

    phased_genotypes = pd.DataFrame(
        phased_genotypes,
        columns=['chromosome', 'position', 'ref', 'alt', 'allele1', 'allele2'])

    return phased_genotypes


def calculate_haplotypes(phasing, changepoint_threshold=0.95):
    """ Calculate haplotype from a set phasing samples.

    Parameters
    ----------
    phasing_samples : list of pandas.Series
        set of phasing samples for a set of SNPs
    changepoint_threshold : float, optional
        threshold on high confidence changepoint calls, by default 0.95

    Returns
    ------
    pandas.DataFrame
        haplotype info with columns:
            chromosome, position, ref, alt, fraction_changepoint, changepoint_confidence,
            is_changepoint, not_confident, chrom_different, hap_label, allele1, allele2
    """

    haplotypes = None
    #n_samples = 0


    phasing = phasing[phasing['allele1'] != phasing['allele2']]


    #print (phasing['allele1'].to_numpy())
    #quit()

    # Identify changepoints.  A changepoint occurs when the alternate allele
    # of a heterozygous SNP is on a different haplotype allele from the alternate
    # allele of the previous het SNP.
    changepoints = phasing['allele1'].diff().abs().astype(float).fillna(0.0)

    #print (changepoints.to_numpy())
    #quit()

    if haplotypes is None:
        haplotypes = changepoints
    else:
        haplotypes += changepoints
        #n_samples += 1

    #haplotypes /= float(n_samples)

    haplotypes = haplotypes.rename('fraction_changepoint').reset_index()

    # Calculate confidence in either changepoint or no changepoint
    haplotypes['changepoint_confidence'] = np.maximum(haplotypes['fraction_changepoint'], 1.0 - haplotypes['fraction_changepoint'])

    #print (haplotypes['changepoint_confidence'].to_numpy())
    #quit()

    # Calculate most likely call of changepoint or no changepoint
    haplotypes['is_changepoint'] = haplotypes['fraction_changepoint'].round().astype(int)

    #print (haplotypes['changepoint_confidence'].to_numpy())
    # Threshold confident changepoint calls
    haplotypes['not_confident'] = (haplotypes['changepoint_confidence'] < float(changepoint_threshold))

    #print (haplotypes['not_confident'].to_numpy())
    #quit()

    # Calculate hap label
    haplotypes['chrom_different'] = haplotypes['chromosome'].ne(haplotypes['chromosome'].shift())

    #print (haplotypes['not_confident'].to_numpy())
    #print (haplotypes['chrom_different'].to_numpy())

    #print (haplotypes['not_confident'].to_numpy().cumsum())
    #quit()

    haplotypes['hap_label'] = (haplotypes['not_confident'] | haplotypes['chrom_different']).cumsum() - 1

    # Calculate most likely alelle1
    haplotypes['allele1'] = haplotypes['is_changepoint'].cumsum().mod(2)
    haplotypes['allele2'] = 1 - haplotypes['allele1']


    return haplotypes



def bamSplitter(input_bam, output_prefix, N, tagName='CB'):

    class BamWriter:
        def __init__(self, alignment, prefix):
            self.alignment = alignment
            self.prefix = prefix
            #self.barcodes = #set(barcodes)
            self.barcodes = set([])
            self._out_files = {}

            self.barcodes_all = set([])


        def write_record_to_barcode(self, rec, barcode, iter1, N):





            if barcode not in self.barcodes_all:

                if len(self.barcodes_all) >= (iter1*N):
                    if len(self.barcodes) < N:
                        self.barcodes.add(barcode)


                self.barcodes_all.add(barcode)





            if barcode not in self.barcodes:
                #print ('banana')
                #quit()
                return

            else:
                if barcode not in self._out_files:
                    #print ('apple')
                    #quit()
                    self._open_file_for_barcode(barcode)
                self._out_files[barcode].write(rec)

        def _open_file_for_barcode(self, barcode):

            self._out_files[barcode] = pysam.AlignmentFile(
                f"{self.prefix}_{barcode}.bam", "wb", template=self.alignment
            )


    def majorPart(input_bam, output_prefix, contigs, iter1, N, tagName):
        """Split a 10x barcoded sequencing file into barcode-specific BAMs

        input:
        barcodes_file: a file containing barcodes or a single barcode
        contigs: '.' for all contigs, 'chr1' for the contig 'chr1',
        or '1-5' for chromosomes 1, 2, 3, 4, and 5
        """
        alignment = pysam.AlignmentFile(input_bam)



        writer = BamWriter(alignment=alignment, prefix=output_prefix)
        if contigs == ".":
            print("Extracting reads from all contigs")
            recs = alignment.fetch()
        else:
            if "-" in contigs:
                start, end = contigs.split("-")
                print(f"Extracting reads from contigs {start} to {end}")
                recs = (alignment.fetch(str(contig)) for contig in range(start, end + 1))
            elif "," in contigs:
                contigs = contigs.split(",")
                print(f"Extracting reads from contigs {contigs}")
                recs = (alignment.fetch(str(contig)) for contig in contigs)
            else:
                print("Extracting reads for one contig: {contigs}")
                recs = (alignment.fetch(c) for c in [contigs])





        b = 0
        for rec in recs:

            try:
                barcode = rec.get_tag(tagName)
                writer.write_record_to_barcode(rec, barcode, iter1, N)

                #print ("A")
                #print (len(writer.barcodes_all))
                #print (len(writer.barcodes))
            except KeyError:
                pass

            b += 1

        Nloop = (len(writer.barcodes_all) - 1) // N
        Nloop = Nloop + 1
        return Nloop


    Nloop = 1
    iter1 = 0
    while iter1 < Nloop:
        Nloop = majorPart(input_bam, output_prefix, '.', iter1, N, tagName)
        iter1 += 1


def ACTsplitter():


    input_bam = './data/bam/ACT.patient1_merged.rg.bam'
    output_prefix = './data/bam/ACT/splitBam/CB'
    N = 1000

    bamSplitter(input_bam, output_prefix, N)

    folder1 = './data/bam/ACT/splitBam/'
    fnames = os.listdir(folder1)
    for a in range(len(fnames)):
        print (a, len(fnames))
        if '.bam' in fnames[a]:

            command1 = 'samtools index ' + folder1 + fnames[a]

            systemPrint(command1)

#ACTsplitter()
#quit()

def ACTremerge():

    folder1 = './data/bam/ACT/splitBam/'
    folder2 = './data/bam/ACT/resplitBam/'

    #fnames2 = os.listdir(folder2)
    #print (len(fnames2))
    #quit()

    
    fnames = os.listdir(folder1)
    #fullNames = []
    count1 = -1
    newNames = []
    for a in range(len(fnames)):
        #print (a, len(fnames))
        if ('.bam' in fnames[a]) and not ('bai' in fnames[a]):

            count1 += 1
            #print (count1)

            if count1 in range(0, 1000):

                print (count1)
            
            
                #print (fnames[a])
                name1 = folder1 + fnames[a]

                barcode = name1.split('/')[-1]
                barcode = barcode.split('_')[-1].split('.')[0]
                print (barcode)
                #CB_CGCACCAGCACCCCG.bam

                bamOutput = folder2 + fnames[a]
                newNames.append(bamOutput)

                if False:
                
                    samFile = bamOutput[:-4] + '.sam'

                    command1 = 'samtools view -h ' + name1 + ' > ' + samFile
                    systemPrint(command1)

                    if True:
                    
                        file1 = open(samFile, 'r')
                        Lines = file1.readlines()
                        for a in range(100):
                            if ('@RG' in Lines[a]) and ('SM:' in Lines[a]):
                                line1 = Lines[a]
                                parts = line1.split('SM:')
                                otherParts = parts[1].split(' ')
                                otherParts[0] = barcode
                                part2 = ' '.join(otherParts)
                                line2 = parts[0] + 'SM:' + part2
                                Lines[a] = line2

                        with open(samFile, 'w') as f:
                            for line in Lines:
                                f.write(line)
                        #resplitBam


                    command2 = 'samtools view -bS ' + samFile + ' > ' + bamOutput
                    systemPrint(command2)

                    command3 = 'rm -r ' + samFile
                    systemPrint(command3)

                    

                    command4 = 'samtools index ' + bamOutput
                    systemPrint(command4)

    
    print (len(newNames))
    bamListFile = './data/info/ACTbams_' + str(len(newNames)) + '.txt'
    np.savetxt(bamListFile, newNames, fmt='%s')
    quit()
    


#ACTremerge()
#quit()



def findHaplotypeCounts(chrNum, outLoc):

    

    phasedFile = outLoc + '/phased/phased_chr' + chrNum + '.bcf'
    countsFile = outLoc + '/counts/seperates_chr' + chrNum + '.vcf.gz'
    
    
    command1 = 'bcftools index -t ' + countsFile
    systemPrint(command1)



    phased_genotypes = read_bcf_phased_genotypes(phasedFile)

    #quit()
       
    positionsAll = phased_genotypes['position'].to_numpy()
    #quit()
    phased_genotypes.set_index(['chromosome', 'position', 'ref', 'alt'], inplace=True)
    
    haplotypes = calculate_haplotypes(phased_genotypes)
    allele1 = haplotypes['allele1'].to_numpy()
    positions = haplotypes['position'].to_numpy()

    RefAllele = haplotypes['ref'].to_numpy()
    AltAllele = haplotypes['alt'].to_numpy()
    


    countArray = []

    subsetKeep = []

    a = 0
    vcf_in = VariantFile(countsFile)  # auto-detect input format

    headerBar = vcf_in.header
    headerBar = str(headerBar)

    headerBar = headerBar.split('INFO\tFORMAT\t')
    headerBar = headerBar[1]
    headerBar = headerBar.replace('\n', '')
    headerBar = headerBar.split('\t')
    #print(headerBar.split('\t'))
    #print(headerBar)
    #quit()

    
    for rec in vcf_in.fetch():
        recStr = str(rec)
        recStr = recStr.replace('\n', '')
        recStr = recStr.split('\t')

        #print (recStr)
        #quit()
        
        #if a > 3700:
        #    print (a)
        #    #quit()

        cellInfo = recStr[9:]

        position1 = int(recStr[1])

        if False:
            if int(position1) in positionsAll:
                print ("Inside")
                #print (haplotypes[positions == int(position1)])
            else:
                print ("Outside")

        if position1 in positions:

            nowRef = recStr[3]
            nowAlt = recStr[4]
            nowAlleles = [nowRef] + nowAlt.split(',')
            #print (nowAlleles)


            subsetPos = np.argwhere(positions ==  position1)[0, 0]
            subsetKeep.append(subsetPos)

            #print (nowAlleles)
            #quit()
            arrayNow = np.zeros(( len(cellInfo), len(nowAlleles)  ), dtype=int)

            for b in range(len(cellInfo)):
                cellInfo1 = cellInfo[b]
                #print (cellInfo1)
                if cellInfo1[0] != '.':
                    #print (cellInfo1)
                    countInfo = cellInfo1.split(':')[2]
                    countInfo = countInfo.split(',')

                    genotypeInfo = cellInfo1.split(':')[0]
                    genotypeInfo = genotypeInfo.split('/')

                    #print ("B")
                    #print (countInfo)
                    #print (cellInfo1.split(':')[0])

                    for c in range(len(genotypeInfo)):
                        arrayNow[b, int(genotypeInfo[c])] += int(countInfo[c])


            #print (np.sum(arrayNow, axis=0))
            #quit()
            #copyArray.append(np.copy(arrayNow))

            

            hapRef = RefAllele[subsetPos]
            hapAlt = AltAllele[subsetPos]
            hapAlleles = [hapRef] + hapAlt.split(',')

            arrayNow2 = np.zeros(( len(cellInfo), len(hapAlleles)  ), dtype=int)

            for b in range(len(hapAlleles)):
                indexCorrespond = np.argwhere( np.array(nowAlleles) == hapAlleles[b]  )#
                if indexCorrespond.shape[0] != 0:
                    indexCorrespond = indexCorrespond[0, 0]
                    arrayNow2[:, b] = arrayNow[:, indexCorrespond]

            arrayNow2 = arrayNow2[:, :2] #For now I can only deal with two haplotypes. 

            countArray.append(np.copy(arrayNow2))

            #print ("B")
            #print (np.sum(arrayNow, axis=0))
            #print (np.sum(arrayNow2, axis=0))
            #quit()

            #if (hapRef == nowRef) and (hapAlt == nowAlt):
            #    True
            #else:

            #    print (recStr)
            #    print (nowRef, nowAlt)
            #    print (hapRef, hapAlt)
            #    quit()
        

        a += 1

        #if a == 10:
        #    quit()

    subsetKeep = np.array(subsetKeep).astype(int)
    positions = positions[subsetKeep]
    allele1 = allele1[subsetKeep]

    countArray = np.array(countArray)
    countArray = np.swapaxes(countArray, 0, 1)

    #print (allele1.shape)
    #print (countArray.shape)

    #quit()

    diff1 = np.argwhere(allele1[1:] - allele1[:-1] != 0)[:, 0]
    start1 = np.concatenate(( np.zeros(1), diff1+1  )).astype(int)
    end1 = np.concatenate(( diff1 + 1, np.zeros(1) + allele1.shape[0]  )).astype(int)
    count_cumsum = np.cumsum(countArray, axis=1)
    count_cumsum = np.concatenate((  np.zeros(( count_cumsum.shape[0]  , 1, 2)) ,   count_cumsum), axis=1)


    sum1 = count_cumsum[:, end1] - count_cumsum[:, start1]


    positionsHap = np.array([positions[start1], positions[end1-1]]).T
    barcodes = np.array(headerBar)

    #print (sum1.shape)

    np.savez_compressed(outLoc + '/counts/allcounts_chr' + chrNum + '.npz', sum1)
    np.savez_compressed(outLoc + '/phasedCounts/barcodes_chr' + chrNum + '.npz', barcodes)
    np.savez_compressed(outLoc + '/phasedCounts/positions_chr' + chrNum + '.npz', positionsHap)

    #quit()




#findHaplotypeCounts()
#quit()

#chrNum = '9'
#countsFile = './data/DLP/counts/chr' + chrNum + '_890.vcf.gz'
#vcf_in = VariantFile(countsFile)

#bamListFile = './data/info/DLPbams.txt'
#bamList = np.loadtxt(bamListFile, dtype=str)

#print (vcf_in.header)
#print (bamList[:10])
#quit()




def giveSimCount(simType):


    

    if simType == 1:
        Ncount = 40
        Ncell = 1000
        array1 = np.random.random((Ncell, Ncount))

        for a in range(10):
            array1[a, array1[a, :] < 0.333 ] = 0
            array1[a, array1[a, :] >= 0.333 ] = 1
        for a in range(10, 1000):
            array1[a, array1[a, :] < 0.5 ] = 0
            array1[a, array1[a, :] >= 0.5 ] = 1

    if simType == 0:
        Ncount = 40
        Ncell = 1000
        array1 = np.random.random((Ncell, Ncount))
        for a in range(1000):
            array1[a, array1[a, :] < 0.5 ] = 0
            array1[a, array1[a, :] >= 0.5 ] = 1

    if simType == 2:
        Ncount = 10
        Ncell = 1000
        array1 = np.random.random((Ncell, Ncount))
        M = 20
        for a in range(M):
            array1[a, array1[a, :] < 0.333 ] = 0
            array1[a, array1[a, :] >= 0.333 ] = 1
        for a in range(M, 1000):
            array1[a, array1[a, :] < 0.5 ] = 0
            array1[a, array1[a, :] >= 0.5 ] = 1

    
    if simType == 3:
        Ncount = 5
        Ncell = 1000
        array1 = np.random.random((Ncell, Ncount))
        M = 1
        for a in range(M):
            array1[a, array1[a, :] < -1 ] = 0
            array1[a, array1[a, :] >= -1 ] = 1
        for a in range(M, 1000):
            array1[a, array1[a, :] < 0.5 ] = 0
            array1[a, array1[a, :] >= 0.5 ] = 1

    

    if simType == 4:
        Ncount = 4
        Ncell = 1000
        array1 = np.random.random((Ncell, Ncount))
        M = 40
        for a in range(M):
            array1[a, array1[a, :] < 0.333 ] = 0
            array1[a, array1[a, :] >= 0.333 ] = 1
        for a in range(M, 1000):
            array1[a, array1[a, :] < 0.5 ] = 0
            array1[a, array1[a, :] >= 0.5 ] = 1

    sum1 = np.sum(array1, axis=1)
    sum1 = np.array([sum1, Ncount - sum1]).T

    #print (sum1[40])
    #quit()
    #print (sum1.shape)
    #quit()

    return sum1





def applyFlipper(countArray, flipper1):

    #print (flipper1)
    array1 = np.array([ np.zeros(flipper1.shape[0]), flipper1  ]).T
    array1 = np.argsort(array1, axis=1)

    argAll = np.argwhere( np.abs(array1) > -1 )
    
    #print (flipper1)

    #print (countArray.shape)
    #print (argAll.shape)

    countArray = countArray[argAll[:, 0], :, array1[argAll[:, 0], argAll[:, 1]] ]
    countArray = countArray.reshape((array1.shape[0], array1.shape[1], countArray.shape[1]))
    countArray = np.swapaxes(countArray, 1, 2)

    #print (countArray.shape)
    #print (np.sum(countArray, axis=1))
    #quit()
    return countArray




def plotGroupedEvidenceSVD(chrNum):


    #dataName = 'ACT10x'
    #dataName = '10x'
    dataName = 'DLP'

    

    if dataName == '10x':
        countArray = loadnpz('./data/' + dataName + '/counts/allcounts_chr' + chrNum + '.npz')
        phasedCount = loadnpz('./data/' + dataName + '/phasedCounts/chr_' + chrNum + '.npz')

    #if dataName == 'ACT10x':
    countArray = loadnpz('./data/' + dataName + '/counts/allcounts_chr' + chrNum + '.npz')
    phasedCount = loadnpz('./data/' + dataName + '/phasedCounts/chr_' + chrNum + '.npz')


    

    countArray = np.swapaxes(countArray, 0, 1)

    M = 100
    N = countArray.shape[0] // M


    for a in range(countArray.shape[0]):
        sum1 = np.sum(countArray[a], axis=0)
        if sum1[0] < sum1[1]:
            countArray[a] = countArray[a, :, -1::-1]

    sumList3 = np.zeros((N, countArray.shape[1], 2))
    sumList2 = np.zeros((N, countArray.shape[1], 2))


    sum1 = np.sum(countArray, axis=1)
    plt.scatter(np.log(sum1[:, 0] - sum1[:, 1] +1), np.log(sum1[:, 0] + sum1[:, 1] +1))
    plt.savefig('./images/temp.png')
    quit()



    #'''
    for a in range(N):
        if a != N:
            args1 = np.arange(M) + (a*M)
        else:
            args1 = np.arange(countArray.shape[0] - (a*M) ) + (a*M)
        

        sumList2[a] = np.copy(np.sum(countArray[args1], axis=0))

        sumList3[a] = np.copy(np.sum(phasedCount[args1], axis=0))

        

    #'''

    ratioImg = sumList3[:, :, 0] / (np.sum(sumList3, axis=2) + 0.01)

    sys.setrecursionlimit(100000)
    import seaborn as sns
    sns.clustermap(ratioImg.T,row_cluster=True, col_cluster=False, cmap='bwr')
    plt.savefig('./images/' + dataName + '_heatmap2_chr' + str(chrNum) + '.png')
    #plt.savefig('./images/heatmap2_chr' + str(chrNum) + '.png')
    plt.clf()
    #quit()


    ratioImg = sumList2[:, :, 0] / (np.sum(sumList2, axis=2) + 0.01)
    import seaborn as sns
    sns.clustermap(ratioImg.T,row_cluster=True, col_cluster=False, cmap='bwr')
    plt.savefig('./images/' + dataName + '_heatmap_chr' + str(chrNum) + '.png')
    #plt.savefig('./images/heatmap_chr' + str(chrNum) + '.png')
    plt.clf()
    #quit()




def groupedEvidenceSVD(outLoc, chrNum):

    #dataName = '10x'
    #dataName = 'ACT10x'


    #quit()

    countArray = loadnpz(outLoc + '/counts/allcounts_chr' + chrNum + '.npz')
    countArray = np.swapaxes(countArray, 0, 1)
    phasedCountFile = outLoc + '/phasedCounts/chr_' + chrNum + '.npz'

    




    shift1 = countArray[:, :, 0] - countArray[:, :, 1]
    sum1 = np.sum(countArray, axis=2)
    evidence = ( shift1 * np.abs(shift1) ) / (sum1.astype(float) + 0.01)
    #evidence = shift1

    U, S, Vh = np.linalg.svd(evidence, full_matrices=True)

    evidenceChunk = U[:, :10]
    #evidenceChunk = U[:, :1]

    

    #'''
    M = 100
    N = countArray.shape[0] // M

    sumList = np.zeros((N, countArray.shape[1], 2))
    sumList3 = np.zeros((N, countArray.shape[1], 2))

    for a in range(N):
        if a != N:
            args1 = np.arange(M) + (a*M)
        else:
            args1 = np.arange(countArray.shape[0] - (a*M) ) + (a*M)
        evidenceChunk_mini = evidenceChunk[args1]
        U, S, Vh = np.linalg.svd(evidenceChunk_mini, full_matrices=True)


        flipper1 = np.sign(U[:, 0])
        

        countArray_mini = countArray[args1]
        countArray_mini = applyFlipper(countArray_mini, flipper1)
        sumList[a] = np.copy(np.sum(countArray_mini, axis=0))
        
        countArray[args1] = np.copy(countArray_mini)

    shift2 = sumList[:, :, 0] - sumList[:, :, 1]
    sum2 = np.sum(sumList, axis=2)
    evidence2 = ( shift2 * np.abs(shift2) ) / (sum2.astype(float) + 0.01)
    

    #momentum1 = 0.2
    momentum1 = 0.5
    evidence3 = np.copy(evidence2)
    
    for a in range(1, evidence3.shape[0]):

        if a != N:
            args1 = np.arange(M) + (a*M)
        else:
            args1 = np.arange(countArray.shape[0] - (a*M) ) + (a*M)

        flip1 = np.sign( np.sum(  evidence3[a-1] * evidence3[a] )  )

        if flip1 == -1:
            #countArray[args1] = countArray[args1, :, -1::-1]
            sumList[a] = sumList[a, :, -1::-1]
            evidence3[a] = evidence3[a] * -1

            countArray[args1] = np.copy(countArray[args1, :, -1::-1])
        
        evidence3[a] = (momentum1 * evidence3[a-1]) + ((1 - momentum1) * evidence3[a])


    #sumList2 = np.zeros((N, countArray.shape[1], 2))

    
    np.savez_compressed(phasedCountFile, countArray)
    
    

def runAllHaplotypeCounts(outLoc):

    for chrNum0 in range(1, 22+1):
        chrNum = str(chrNum0)
        findHaplotypeCounts(chrNum, outLoc)


def runAllGroupedEvidenceSVD(outLoc):

    for chrNum0 in range(1, 22+1):
        chrNum = str(chrNum0)
        #print (chrNum)
        groupedEvidenceSVD(outLoc, chrNum)








def findReadCounts(bamLoc, outLoc):

    tagName = 'RG'

    samfile = pysam.AlignmentFile(bamLoc, "rb")

    chrType = 'num'
    chrName = ''
    try:
        samfile.fetch(chrName)
    except:
        chrType = 'name'

    
    

    for b in range(0, 22):
        chrName = str(int(b + 1))

        #print ('')
        #print (chrName)
        #print ('')

        chrNameLoad = chrName
        if chrType == 'name':
            chrNameLoad = 'chr' + chrNameLoad
        

        dictArray = {}
        dictCounts = {}
        a = 0
        for read in samfile.fetch(chrNameLoad):
            a += 1
            #if (a % 10000000) == 0:
            #    print (a)

            cellName = read.get_tag(tagName)

            N = int(1e6)
            if not cellName in dictCounts:
                dictCounts[cellName] = 0
                dictArray[cellName] = np.zeros(N, dtype=int)

            if dictCounts[cellName] >= dictArray[cellName].shape[0]:
                dictArray[cellName] = np.concatenate((dictArray[cellName], np.zeros(N, dtype=int) ), axis=0)

            
            dictArray[cellName][dictCounts[cellName]] = read.pos
            dictCounts[cellName] = dictCounts[cellName] + 1

        #print (len(dictArray.keys()))

        for cellName in dictCounts.keys():
            count1 = dictCounts[cellName]
            fileName = outLoc + '/readCounts/' + 'pos/' + str(chrName) + '/' + cellName + '.npz'
            np.savez_compressed(fileName, dictArray[cellName][:count1])

    samfile.close()

    #np.savez_compressed('./fromBam.npz', data[:a])

    #1636440000





def runAllSteps(bamLoc, refLoc, outLoc, refGenome):

    
    numSteps = '9'
    stepName = 0

    stepName += 1
    stepString = str(stepName) + '/' + numSteps
    print ('Data processing — Step ' + stepString + ': Creating directories... ', end='')
    makeAllDirectories(outLoc)
    print ("Done")

    stepName += 1
    stepString = str(stepName) + '/' + numSteps
    print ('Data processing — Step ' + stepString + ': Running bcftools on pseudobulk (may take hours to days)... ', end='')
    findCombinedCounts(bamLoc, refLoc, outLoc, refGenome)
    print ("Done")

    stepName += 1
    stepString = str(stepName) + '/' + numSteps
    print ('Data processing — Step ' + stepString + ': Running SHAPE-IT... ', end='')
    runPhasing(outLoc, refGenome, refLoc)
    print ("Done")

    stepName += 1
    stepString = str(stepName) + '/' + numSteps
    print ('Data processing — Step ' + stepString + ': Running bcftools on individual cells... ', end='')
    findSubsetCounting(outLoc) #This small processing step doesn't require its own step printed in terminal

    findIndividualCounts(bamLoc, refLoc, outLoc, refGenome)
    print ("Done")

    stepName += 1
    stepString = str(stepName) + '/' + numSteps
    print ('Data processing — Step ' + stepString + ': Calculating total read depths... ', end='')
    findReadCounts(bamLoc, outLoc)
    print ("Done")

    stepName += 1
    stepString = str(stepName) + '/' + numSteps
    print ('Data processing — Step ' + stepString + ': Phasing haplotype blocks... ', end='')
    runAllHaplotypeCounts(outLoc)

    runAllGroupedEvidenceSVD(outLoc)
    print ('Done')









