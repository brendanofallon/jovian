#!/bin/bash


#not this SBATCH --account owner-gpu-guest
#not this SBATCH --partition kingspeak-gpu-guest
#SBATCH --account=kingspeak-gpu
#SBATCH --partition=kingspeak-gpu
#SBATCH --time=2-0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brendan.ofallon@aruplab.com
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB


module load cuda/11.3


PYTHON=$HOME/miniconda3/envs/ds2s/bin/python

ds2s=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/main.py

#CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/chpc_conf.yaml
CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/chpc_conf3.yaml

VAL_DIR=/uufs/chpc.utah.edu/common/home/arup-storage3/u0379426/pregen_no_chr20_21_subsample/
PREGEN_DIR=/uufs/chpc.utah.edu/common/home/arup-storage4/u6004674/dnaseq2seq/onc_giabs_capturewide/all_chr_except_20_21/pregen_all_chr_except_20_21/


ALTPREDICTOR=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/altpredictor_pleasant-dew-54-299.sd

LEARNING_RATE=0.0005

CHECKPOINT_FREQ=5


$PYTHON $ds2s train \
    -c $CONF \
    -d $PREGEN_DIR \
    --val-dir $VAL_DIR \
    -n 100 \
    -ap $ALTPREDICTOR \
    --learning-rate $LEARNING_RATE \
    --checkpoint-freq $CHECKPOINT_FREQ \
    -o /uufs/chpc.utah.edu/common/home/arup-storage3/u0379426/ninefeats_nochr2021_pleasant-dew-ap_notrain_ap.model \
    --threads 16 \
    --max-decomp-batches 60

