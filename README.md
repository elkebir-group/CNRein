# DeepCopy
DeepCopy is a deep reinforcement learning based evolution-aware algorithm for haplotype-specific copy number calling on single cell DNA sequencing data. 

<p align="center">
  <img width="1000" height="220" src="./overview.png">
</p>

## Instalation

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

Run the below command to install DeepCopy
```bash
pip install DeepSomaticCopy
```
This automatically installs numpy, pandas, pysam, statsmodels, and pytorch. However, samtools, bcftools and shapeit4 still need to be installed (all of these are available via bioconda). 




## Usage

### With manual installation

If installed manually, the default usage is 
```bash
python script.py -input <BAM file location> -ref <reference folder location> -output <location to store results> -refGenome <either "hg19" or "hg38">
```
An example usage could be as below
```bash
python script.py -input ./data/TN3_FullMerge.bam -ref ./data/refNew -output ./data/newTN3 -refGenome hg38
```

### With pip installation

DeepCopy can be ran with the following command:
```bash
DeepCopyRun -input <BAM file location> -ref <reference folder location> -output <location to store results> -refGenome <either "hg19" or "hg38">
```
An example command could be:
```bash
DeepCopyRun -input ./data/TN3_FullMerge.bam -ref ./data/refNew -output ./data/newTN3 -refGenome hg38
```

Additionally, one can run only parts of the DeepCopy pipeline with the following command:
```bash
DeepCopyRun -step <name of step to be ran> -input <BAM file location> -ref <reference folder location> -output <location to store results> -refGenome <either "hg19" or "hg38">
```
Here "name of step to be ran" can be "processing" for data processing steps, "NaiveCopy" for additional NaiveCopy steps or "DeepCopy" for the final deep reinforcement learning step. 
The steps "NaiveCopy" and "DeepCopy" only require the "-output" argument, and not "-ref", "refGenome", or "-input" (it is assumed that the correct data for these steps is already in the "-output" folder). 
In the "examples" folder, we provide the input to the "NaiveCopy" and "DeepCopy" steps for three datasets from our paper. 
For S0, we provide input files to "DeepCopy" but not "NaiveCopy" due to GitHub's file size constraints (since S0 contains more cells, the files are larger). 
This allows for the below command to be ran without having to download any additional data.
```bash
DeepCopyRun -step NaiveCopy -output ./examples/TN3
DeepCopyRun -step DeepCopy -output ./examples/TN3
```
The equivalent command can also be ran with "./examples/TN1", "./examples/Ovarian", and for the DeepCopy step "./examples/S0". 


## Input requirements

The default input format is a single BAM file with different read groups for different cells. 
Future updates will also allow individual BAM files for each cell. 
The default reference files are publically available at https://zenodo.org/records/10076403. 
The final output in the form of an easily interpretable CSV file is produced in the folder "finalPrediction" within the user provided "-output" folder. 
