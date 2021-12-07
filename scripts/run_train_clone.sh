#!/bin/bash

#SBATCH --account=notchpeak-gpu
#SBATCH --partition=notchpeak-gpu
#SBATCH --time=2-0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brendan.ofallon@aruplab.com
#SBATCH --gres=gpu:1



module load cuda/11.3

ROOT_DIR=/uufs/chpc.utah.edu/common/home/arup-storage3/u0379426/variant_transformer_runs/

REPO_BASE=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/

GIT_BRANCH="master"

PYTHON=$HOME/miniconda3/envs/ds2s/bin/python



#CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/chpc_conf.yaml
CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/chpc_conf3.yaml

#VAL_DIR=/uufs/chpc.utah.edu/common/home/arup-storage3/u0379426/pregen_9feats_chr20_21only/
VAL_DIR=/scratch/general/lustre/u0064568/seq2seq/exome_av/exome_av_valspc_pregen_chr20and21
#PREGEN_DIR=/uufs/chpc.utah.edu/common/home/arup-storage4/u6004674/dnaseq2seq/pregen_all_chr_except_20_21/
PREGEN_DIR=/scratch/general/lustre/u0064568/seq2seq/exome_av/exome_av_valspc_pregen_nochr20or21



LEARNING_RATE=0.0005

CHECKPOINT_FREQ=1

RUN_NAME="test_shuffle"
RUN_NOTES="test shortening seq length to 150"

set -x


cd $ROOT_DIR


mkdir -p $RUN_NAME
cd $RUN_NAME

git clone $REPO_BASE

cd dnaseq2seq
git checkout $GIT_BRANCH
ds2s=$(readlink -f main.py)
COMMIT=$(git rev-parse HEAD)


cd ..

echo "Branch: $GIT_BRANCH \n commit: $COMMIT \n" >> git_info.txt

export ENABLE_WANDB=1

$PYTHON $ds2s train \
    -c $CONF \
    -d $PREGEN_DIR \
    --val-dir $VAL_DIR \
    -n 100 \
    --learning-rate $LEARNING_RATE \
    --checkpoint-freq $CHECKPOINT_FREQ \
    -o ${RUN_NAME}.model \
    --threads 8 \
    --max-decomp-batches 8 \
    --wandb-run-name $RUN_NAME \
    --wandb-notes "$RUN_NOTES"

echo "Script is exiting"
