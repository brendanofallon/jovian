#!/bin/bash

##SBATCH --account=notchpeak-gpu
##SBATCH --partition=notchpeak-gpu

##SBATCH --account=arup-gpu-np
##SBATCH --partition=arup-gpu-np
##SBATCH --mem=128G
##SBATCH --cpus-per-task=16
##SBATCH --time=8-0
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=brendan.ofallon@aruplab.com
##SBATCH --gres=gpu:2 --constraint="a6000|a100"


module load gcc/11.2.0 # Required for recent version of glibc / libstdc++ (GLIBCXXX errors)
module load cuda/11.6.2
PYTHON=$HOME/storage/miniconda3/envs/jv2/bin/python


CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/chpc_conf3.yaml

#VAL_DIR=$HOME/storage/pregen_depth200_chr21and22
VAL_DIR=/uufs/chpc.utah.edu/common/home/u0379426/storage/pregen_lcbigmap_d150_chrs21and22

#PREGEN_DIR=/scratch/general/vast/u0379426/wgs_pregen_mq_lcsus
#PREGEN_DIR=/uufs/chpc.utah.edu/common/home/u0379426/storage/pregen_mq_small
PREGEN_DIR=/scratch/general/vast/u0379426/pregen_lcbigmap_d150/


LEARNING_RATE=0.00004

CHECKPOINT_FREQ=1

DDP_VARS=$( python /uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/scripts/ddp_slurm_setup.py "$@" )
echo "DDP Vars: $DDP_VARS"
eval $DDP_VARS

set -x

export ENABLE_WANDB=0


export JV_LOGLEVEL=INFO; $PYTHON dnaseq2seq/dnaseq2seq/main.py train \
    -d $PREGEN_DIR \
    --val-dir $VAL_DIR \
    -n 25 \
    --batch-size 512 \
    --learning-rate $LEARNING_RATE \
    --checkpoint-freq $CHECKPOINT_FREQ \
    -o ${RUN_NAME}.model \
    --threads 16 \
    --max-decomp-batches 8 \
    --samples-per-epoch 10000000 \
    --wandb-run-name $RUN_NAME \
    --wandb-notes "$RUN_NOTES"

echo "Script is exiting"

