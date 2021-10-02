#!/bin/bash

#SBATCH --account=kingspeak-gpu
#SBATCH --partition=kingspeak-gpu
#SBATCH --time=1-0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brendan.ofallon@aruplab.com
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB


module load cuda/11.3


PYTHON=$HOME/miniconda3/envs/ds2s/bin/python

ds2s=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/main.py

#CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/chpc_conf.yaml
CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/chpc_conf3.yaml


PREGEN_DIR=/uufs/chpc.utah.edu/common/home/arup-storage3/u0379426/pregen_all_upsample5k/batch3

LEARNING_RATE=0.0005

CHECKPOINT_FREQ=5


$PYTHON $ds2s train \
    -c $CONF \
    -d $PREGEN_DIR \
    -n 100 \
    --learning-rate $LEARNING_RATE \
    --checkpoint-freq $CHECKPOINT_FREQ \
    -o /uufs/chpc.utah.edu/common/home/arup-storage3/u0379426/test_batch3_gpu_2.model \
    --threads 16 \
    --max-decomp-batches 60

