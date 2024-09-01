#!/bin/bash -l

# sbatch documentation: https://slurm.schedmd.com/sbatch.html

##SBATCH --account=notchpeak-gpu
##SBATCH --partition=notchpeak-gpu


#SBATCH --account=arup-gpu-np
#SBATCH --partition=arup-gpu-np
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=0-12
#SBATCH --gres=gpu:1 --constraint="2080ti|v100|3090|a6000|a100"


JOV_ROOT=$1
BED=$2
BAM=$3

REF_GENOME=/uufs/chpc.utah.edu/common/home/u0379426/vast/ref/human_g1k_v37_decoy_phiXAdaptr.fasta.gz
HAPPY_LOOKUP=$JOV_ROOT/eval/happy_lookup.py


SUFFIX=$(basename $JOV_ROOT | sed -e 's/jenever-//g')
PREFIX=$(basename $BAM | sed -e 's/.cram//g' | sed -e 's/.bam//g')

VCF="${PREFIX}_tag_${SUFFIX}.vcf"
#MODEL=/uufs/chpc.utah.edu/common/home/u0379426/storage/variant_transformer_runs/100M_s28_cont_mapsus_lolr2/100M_s28_cont_mapsus_lolr2_epoch2.model
#MODEL=/uufs/chpc.utah.edu/common/home/arup-storage4/brendan/variant_transformer_runs/100M_BWA_ftmore/100M_BWA_ftmore_epoch98.model
MODEL=/uufs/chpc.utah.edu/common/home/u0379426/storage/variant_transformer_runs/good44fix/good44fix_epoch280.model

TMPDIR=/tmp

CLASSIFIER=/uufs/chpc.utah.edu/common/home/arup-storage4/brendan/jovian/good44_classifiertraining/g44e280_clf.model

PYTHON=/uufs/chpc.utah.edu/common/home/u0379426/miniconda3/envs/py3/bin/python

export JV_LOGLEVEL=INFO

set -x 

start_time=$(date +%s)

$PYTHON $JOV_ROOT/src/dnaseq2seq/main.py call \
    --threads 24 \
    --bam $BAM \
    --bed $BED \
    -c $CLASSIFIER \
    --temp-dir $TMPDIR \
    --max-batch-size 512 \
    -r $REF_GENOME \
    -m $MODEL \
    -v $VCF

# Record the end time
end_time=$(date +%s)

# Calculate the elapsed time
elapsed_time=$((end_time - start_time))

echo "Done variant calling, now running happy"

export PATH="$HOME"/miniconda3/envs/py3/bin/:$PATH

$PYTHON $HAPPY_LOOKUP $BED $VCF


