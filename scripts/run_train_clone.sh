#!/bin/bash

#SBATCH --account=notchpeak-gpu
#SBATCH --partition=notchpeak-gpu
#SBATCH --time=1-0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brendan.ofallon@aruplab.com
#SBATCH --gres=gpu:1 --constraint="a100|3090"



module load cuda/11.3

ROOT_DIR=/uufs/chpc.utah.edu/common/home/arup-storage3/u0379426/variant_transformer_runs/

REPO_BASE=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/

GIT_BRANCH="50m"

PYTHON=$HOME/miniconda3/envs/ds2s/bin/python



#CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/chpc_conf.yaml
CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/chpc_conf3.yaml

#VAL_DIR=/uufs/chpc.utah.edu/common/home/arup-storage3/u0379426/pregen_9feats_chr20_21only/
VAL_DIR=/uufs/chpc.utah.edu/common/home/arup-storage3/u0379426/pregen_wgs_readwindowfix_w150_chr21and22
#PREGEN_DIR=/uufs/chpc.utah.edu/common/home/arup-storage4/u6004674/dnaseq2seq/pregen_all_chr_except_20_21/
PREGEN_DIR=/uufs/chpc.utah.edu/common/home/arup-storage3/u0379426/pregen_wgs_w150_nochr21or22_bigger



LEARNING_RATE=0.00005

CHECKPOINT_FREQ=1

RUN_NAME="wgs_50m_bigger_cont7"
RUN_NOTES="50M model, width 150, contination7"

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
    --threads 1 \
    --max-decomp-batches 4 \
    -i /uufs/chpc.utah.edu/common/home/u0379426/storage/variant_transformer_runs/wgs_50m_bigger_cont6/wgs_50m_bigger_cont6_epoch1.model \
    --wandb-run-name $RUN_NAME \
    --wandb-notes "$RUN_NOTES"

echo "Script is exiting"
