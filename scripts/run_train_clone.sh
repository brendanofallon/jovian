#!/bin/bash

#SBATCH --account=notchpeak-gpu
#SBATCH --partition=notchpeak-gpu
#SBATCH --time=2-0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brendan.ofallon@aruplab.com
#SBATCH --gres=gpu:1 --constraint="v100|3090"



module load cuda/11.3

ROOT_DIR=/uufs/chpc.utah.edu/common/home/arup-storage3/u0379426/variant_transformer_runs/

REPO_BASE=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/

GIT_BRANCH="master"

PYTHON=$HOME/miniconda3/envs/ds2s/bin/python



#CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/chpc_conf.yaml
CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/chpc_conf3.yaml

#VAL_DIR=/uufs/chpc.utah.edu/common/home/arup-storage3/u0379426/pregen_9feats_chr20_21only/
VAL_DIR=/uufs/chpc.utah.edu/common/home/arup-storage3/u0379426/pregen_wgs_multindel_nova_chr21and22
#PREGEN_DIR=/uufs/chpc.utah.edu/common/home/arup-storage4/u6004674/dnaseq2seq/pregen_all_chr_except_20_21/
PREGEN_DIR=/uufs/chpc.utah.edu/common/home/arup-storage3/u0379426/pregen_wgs_multindel_nova_nochr21or22



LEARNING_RATE=0.0001

CHECKPOINT_FREQ=1

RUN_NAME="wgs_abitbigger_cont4.1"
RUN_NOTES="WGS with all samples and a sorta big and 8 layers / 8 heads model, continuation after epoch 1 of round 4!"

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
    --threads 4 \
    --max-decomp-batches 4 \
    -i /uufs/chpc.utah.edu/common/home/u0379426/storage/variant_transformer_runs/wgs_abitbigger_cont3/wgs_abitbigger_cont3_epoch0.model \
    --wandb-run-name $RUN_NAME \
    --wandb-notes "$RUN_NOTES"

echo "Script is exiting"
