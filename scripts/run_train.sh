#!/bin/bash

#SBATCH --account=kingspeak-gpu
#SBATCH --partition=kingspeak-gpu
#SBATCH --nodes=1
#SBATCH --time=1-0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brendan.ofallon@aruplab.com
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB


module load cuda/11.3


PYTHON=$HOME/miniconda3/envs/ds2s/bin/python

ds2s=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/main.py

#CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/chpc_conf.yaml
CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/chpc_conf3.yaml


PREGEN_DIR=/uufs/chpc.utah.edu/common/home/arup-storage3/u0379426/pregen_round3_partial

LEARNING_RATE=0.0005

CHECKPOINT_FREQ=10   # Save the model every X epochs


$PYTHON $ds2s train \
    -c $CONF \
    -d $PREGEN_DIR \
    -n 100 \ # Number of epochs to train for
    --learning-rate $LEARNING_RATE \
    --checkpoint-freq $CHECKPOINT_FREQ \
    -o /uufs/chpc.utah.edu/common/home/arup-storage3/u0379426/r3partialtest_gpu2.model 

