#!/bin/bash

#SBATCH --account=arupbio-kp
#SBATCH --partition=arup-kp
#SBATCH --nodes=1
#SBATCH --time=1-0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brendan.ofallon@aruplab.com


NAME=$(echo $3 | sed -e 's/\.bam//g' | sed -e 's/\.cram//g')
FQ1=$(readlink -f $1)
FQ2=$(readlink -f $2)

# Cant activate a conda env non-interactively, so just set the  python binary
# to the right spot - seems to work

cd  /scratch/general/vast/u0379426/bam_wgs

# Yes you need to do this
mkdir -p $NAME
cd $NAME


GEM=$HOME/miniconda3/envs/py3/bin/gem-mapper

#gem-indexer --input /uufs/chpc.utah.edu/common/home/u0379426/arup-storage3/Reference/Data/B37/GATKBundle/2.8_subset_arup_v0.1/human_g1k_v37_decoy_phiXAdaptr.fasta

$GEM --index /scratch/general/vast/u0379426/human_g1k_v37_decoy_phiXAdaptr.gem -1 $FQ1 -2 $FQ2 -t 36 -r "@RG\tID:sample\tPL:ILLUMINA\tLB:sample\tSM:sample" | \
    samtools sort --reference /scratch/general/vast/u0379426/ref/human_g1k_v37_decoy_phiXAdaptr.fasta.gz -@ 8 | \
    samtools view -C --reference /scratch/general/vast/u0379426/ref/human_g1k_v37_decoy_phiXAdaptr.fasta.gz -@ 8 -h -o $3 

samtools index $3
