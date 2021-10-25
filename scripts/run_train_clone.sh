#!/bin/bash

#SBATCH --account=kingspeak-gpu
#SBATCH --partition=kingspeak-gpu
#SBATCH --time=2-0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brendan.ofallon@aruplab.com
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB



module load cuda/11.3

ROOT_DIR=/uufs/chpc.utah.edu/common/home/arup-storage3/u0379426/variant_transformer_runs/

REPO_BASE=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/

GIT_BRANCH="transforming_loaders"

PYTHON=$HOME/miniconda3/envs/ds2s/bin/python



#CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/chpc_conf.yaml
CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/chpc_conf3.yaml

VAL_DIR=/uufs/chpc.utah.edu/common/home/arup-storage3/u0379426/pregen_9feats_chr20_21only/
PREGEN_DIR=/uufs/chpc.utah.edu/common/home/arup-storage4/u6004674/dnaseq2seq/pregen_all_chr_except_20_21/


ALTPREDICTOR=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/altpredictor_pleasant-dew-54-299.sd

LEARNING_RATE=0.0005

CHECKPOINT_FREQ=1

RUN_NAME="test-shorten_seqlen_150"

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

$PYTHON $ds2s train \
    -c $CONF \
    -d $PREGEN_DIR \
    --val-dir $VAL_DIR \
    -n 100 \
    --learning-rate $LEARNING_RATE \
    --checkpoint-freq $CHECKPOINT_FREQ \
    -o my_new.model \
    --threads 16 \
    --max-decomp-batches 32

echo "Script is exiting"
