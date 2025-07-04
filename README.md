# CNRein

[![PyPI version](https://badge.fury.io/py/CNRein.svg?cacheSeconds=60)](https://badge.fury.io/py/CNRein)

CNRein (formerly known as DeepCopy) is a deep reinforcement learning based evolution-aware algorithm for haplotype-specific copy number calling on single cell DNA sequencing data. 

<p align="center">
  <img width="1000" height="220" src="./overview.png">
</p>

## Contact

Please contact Stefan Ivanovic at stefan4@illinois.edu with any issues running the software or any requests for additional features in CNRein. 

## Installation

### Manual

Manual installation is currently available and can be achieved by cloning this GitHub repository and installing the below requirements all available on conda:
- Python 3
- numpy
- pandas
- samtools
- bcftools
- pysam
- statsmodels
- pytorch
- shapeit4

Note, pip can be used as an alternative to conda for any of these packages available via pip. 

### Pip installation

Run the below command to install CNRein
```bash
pip install CNRein
```
This automatically installs numpy, pandas, pysam, statsmodels, and pytorch. However, samtools, bcftools and shapeit4 still need to be installed (all of these are available via bioconda). 




## Usage

### With manual installation

If installed manually, the default usage is 
```bash
python script.py -input <BAM file location> \
    -ref <reference folder location> \
    -output <location to store results> \
    -refGenome <either "hg19" or "hg38">
```
An example usage could be as follows
```bash
python script.py -input ./data/TN3_FullMerge.bam -ref ./data/refNew -output ./data/newTN3 -refGenome hg38
```
Additionally, "-CB" can be included if cells are indicated by cell barcode rather than read group, and "-seperateBAMs" can be included if each cell has its own BAM file. 

### With pip installation

CNRein can be run with the following command:
```bash
CNRein -input <BAM file location> \
    -ref <reference folder location> \
    -output <location to store results> \
    -refGenome <either "hg19" or "hg38">
```
An example command could be:
```bash
CNRein -input ./data/TN3_FullMerge.bam \
    -ref ./data/refNew \
    -output ./data/newTN3 \
    -refGenome hg38
```
Optional parameters include "-CB" if cells are indicated by cell barcode rather than read group, "-seperateBAMs" if each cell has its own BAM file, and "-maxPloidy {some value}" to set a maximum ploidy value for predictions. 

Additionally, one can run only parts of the CNRein pipeline with the following command:
```bash
CNRein -step <name of step to be ran> \
    -input <BAM file location> \
    -ref <reference folder location> \
    -output <location to store results> \
    -refGenome <either "hg19" or "hg38">
```
Here "name of step to be ran" can be any of the three sequential steps: "processing" for data processing steps, "CNNaive" for additional NaiveCopy steps or "CNRein" for the final deep reinforcement learning step. 
The "processing" step utilizes BAM files as inputs, and produces segments with haplotype specific read counts and GC bais corrected read depths (stored in "binScale") as well as intermediary files (stored in "initial", "counts", "info", "phased", "phasedCounts" and "readCounts"). 
The "CNNaive" step utilized the segments and read counts produced by "process" and generates NaiveCopy's predictions stored in "finalPrediction" as well as intermediate files stored in "binScale". 
The "CNRein" step utilizes the outputs of both "processing" and "CNNaive", and produces predictions in "finalPrediction", as well as the neural network model stored in "model". 
In terms of the precise files, we have the following. 

#### Processing step
Inputs: "-input" BAM file

Outputs: In ./binScale the files "BAF_noise.npz", "bins.npz", "chr_avg.npz", "filtered_HAP_avg.npz", "filtered_RDR_avg.npz", and "filtered_RDR_noise.npz". Additionally all files in ./counts, ./info, ./phased, ./phasedCounts ./readCounts, and ./initial. 

#### CNNaive step
Inputs: In ./binScale the files "BAF_noise.npz", "bins.npz", "chr_avg.npz", "filtered_HAP_avg.npz", "filtered_RDR_avg.npz", and "filtered_RDR_noise.npz". In ./initial the files chr_1M.npz, RDR_1M.npz, and HAP_1M.npz.

Outputs: In ./binScale the files "dividerAll.npz", "dividerError.npz", "dividers.npz", "initialCNA.npz", "initialIndex.npz", "initialUniqueCNA.npz" and "regions.npz". Additionally, "./finalPrediciton/CNNaivePrediction.csv". 

#### CNRein step
Inputs: In ./binScale the files "BAF_noise.npz", "bins.npz", "chr_avg.npz", "filtered_HAP_avg.npz", "filtered_RDR_avg.npz", "filtered_RDR_noise.npz", and "initialUniqueCNA.npz". 

Outputs: In ./model the files "model_now.pt", and "pred_now.npz". Additionally, "./finalPrediciton/CNReinPrediction.csv". 


The "CNNaive" and "CNRein" steps do not require bcftools, samtools or SHAPE-IT. 
Instead, they only require python package dependencies that are automatically installed when installing CNRein through pip. 
The steps "CNNaive" and "CNRein" only require the "-output" argument, and not "-ref", "refGenome", or "-input" (it is assumed that the correct data for these steps is already in the "-output" folder). 
In the "examples" folder, we provide the input to the "CNNaive" and "CNRein" steps for three datasets from our paper. 
For S0, we provide input files to "CNRein" but not "CNNaive" due to GitHub's file size constraints (since S0 contains more cells, the files are larger). 
This allows for the below command to be ran without having to download any additional data.
```bash
CNRein -step CNNaive -output ./examples/TN3
CNRein -step CNRein -output ./examples/TN3
```

## Input requirements

The default input format is a single BAM file with different read groups (or cell barcodes) for different cells. 
Future updates will also allow individual BAM files for each cell. 
The default reference files are publically available at https://zenodo.org/records/10076403. 
The final output in the form of an easily interpretable CSV file is produced in the folder "finalPrediction" within the user provided "-output" folder. 





