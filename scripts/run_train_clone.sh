#!/bin/bash

##SBATCH --account=notchpeak-gpu
##SBATCH --partition=notchpeak-gpu

#SBATCH --account=arup-gpu-np
#SBATCH --partition=arup-gpu-np
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=8-0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brendan.ofallon@aruplab.com
#SBATCH --gres=gpu:1 --constraint="a6000|a100"


module load gcc/11.2.0 # Required for recent version of glibc / libstdc++ (GLIBCXXX errors)
module load cuda/11.6.2

ROOT_DIR=/uufs/chpc.utah.edu/common/home/arup-storage3/u0379426/variant_transformer_runs/

REPO_BASE=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/

GIT_BRANCH="decoder_big"

PYTHON=$HOME/storage/miniconda3/envs/jv/bin/python


#CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/chpc_conf.yaml
CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/chpc_conf3.yaml

#VAL_DIR=/uufs/chpc.utah.edu/common/home/arup-storage3/u0379426/pregen_wgs_readwindowfix_w150_chr21and22
#VAL_DIR=/uufs/chpc.utah.edu/common/home/u0379426/storage/wgs_pregen_mqfeat_chrs21and22_val
VAL_DIR=/uufs/chpc.utah.edu/common/home/arup-storage3/u0379426/pregen_depth200_chrD

#PREGEN_DIR=/uufs/chpc.utah.edu/common/home/arup-storage3/u0379426/wgs_pregen_halfhuge
#PREGEN_DIR=/uufs/chpc.utah.edu/common/home/arup-storage3/u0379426/test_pregen_mqfeat_kmers

#PREGEN_DIR=/uufs/chpc.utah.edu/common/home/arup-storage3/u0379426/pregen_wgs_w150_nochr21or22_big

#PREGEN_DIR=/uufs/chpc.utah.edu/common/home/arup-storage3/u0379426/wgs_pregen_huge
#PREGEN_DIR=/uufs/chpc.utah.edu/common/home/arup-storage3/u0379426/wgs_pregen_halfhuge_lc_bigvars_fpfns
#PREGEN_DIR=/uufs/chpc.utah.edu/common/home/u0379426/storage/wgs_pregen_mqfeat_all
#PREGEN_DIR=/uufs/chpc.utah.edu/common/home/u0379426/storage/wgs_pregen_mqfeat_all_plus_susregions
#PREGEN_DIR=/uufs/chpc.utah.edu/common/home/u0379426/storage/wgs_pregen_smallsus
#PREGEN_DIR=/uufs/chpc.utah.edu/common/home/u0379426/storage/wgs_pregen_mq_lcbig_partial
#PREGEN_DIR=/scratch/general/vast/u0379426/wgs_pregen_mq_lcbig_partial
#PREGEN_DIR=/uufs/chpc.utah.edu/common/home/u0379426/storage/pregen_mq_small
PREGEN_DIR=/uufs/chpc.utah.edu/common/home/arup-storage3/u0379426/pregen_depth200test


LEARNING_RATE=0.00004

CHECKPOINT_FREQ=1

RUN_NAME="depth200test_pt2"
RUN_NOTES="test depth 200"

set -x


cd $ROOT_DIR


mkdir -p $RUN_NAME
cd $RUN_NAME

git clone $REPO_BASE

cd dnaseq2seq
git checkout $GIT_BRANCH
ds2s=$(readlink -f dnaseq2seq/main.py)
COMMIT=$(git rev-parse HEAD)


cd ..

echo "Branch: $GIT_BRANCH \n commit: $COMMIT \n" >> git_info.txt

export ENABLE_WANDB=

export JV_LOGLEVEL=INFO; $PYTHON $ds2s train \
    -d $PREGEN_DIR \
    --val-dir $VAL_DIR \
    -n 25 \
    --batch-size 512 \
    --learning-rate $LEARNING_RATE \
    --checkpoint-freq $CHECKPOINT_FREQ \
    -o ${RUN_NAME}.model \
    --threads 16 \
    --max-decomp-batches 8 \
    --samples-per-epoch 5000000 \
    --wandb-run-name $RUN_NAME \
    --wandb-notes "$RUN_NOTES"

echo "Script is exiting"

