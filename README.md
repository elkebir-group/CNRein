# DeepCopy
DeepCopy is a deep reinforcement learning based evolution-aware algorithm for haplotype-specific copy number calling on single cell DNA sequencing data. 

<p align="center">
  <img width="1000" height="220" src="./overview.png">
</p>

## Instalation

DeepCopy is currently being converted to a Bioconda package to allow for automatic installation including all requirement packages. 

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
- 
Note, pip can be used as an alternative to conda for any of these packages available via pip. 

## Usage

If installed manually, the default usage is 
```bash
python pipeline.py -input <BAM file location> -ref <reference folder location> -output <location to store results> -refGenome <either "hg19" or "hg38">
```
An example usage could be as below
```bash
python pipeline.py -input ./data/TN3_FullMerge.bam -ref ./data/refNew -output ./data/newTN3 -refGenome hg38
```

If installed via Bioconda, the default usage is: 
```bash
DeepCopy -input <BAM file location> -ref <reference folder location> -output <location to store results> -refGenome <either "hg19" or "hg38">
```

The default input format is a single BAM file with different read groups for different cells. 
Future updates will also allow individual BAM files for each cell. 
The default reference files are publically available at https://zenodo.org/records/10076403. 
The final output in the form of an easily interpretable CSV file is produced in the folder "finalPrediction" within the user provided "-output" folder. 
