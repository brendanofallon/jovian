#!/bin/bash

MASTER_ADDR=$1
MASTER_PORT=$2
RUN_NAME=$3
RUN_NOTES="$4"

VAL_DIR=/uufs/chpc.utah.edu/common/home/u0379426/storage/pregen_lcbigmap_d150_chrs21and22
PREGEN_DIR=/scratch/general/vast/u0379426/pregen_lcbigmap_d150/


RUN_SCRIPT=$HOME/src/dnaseq2seq/scripts/run_train_clone.sh

RUNCMD="dnaseq2seq/dnaseq2seq/main.py train \
    -d $PREGEN_DIR \
    --val-dir $VAL_DIR \
    -n 25 \
    --batch-size 128 \
    --learning-rate 0.00003 \
    --checkpoint-freq 1 \
    -o ${RUN_NAME}.model \
    --threads 16 \
    --max-decomp-batches 8 \
    --samples-per-epoch 10000000 \
    --wandb-run-name $RUN_NAME"

echo "Full run cmd: $RUNCMD"
echo "Master addr: $MASTER_ADDR, master port: $MASTER_PORT"

export ENABLE_WANDB=1

torchrun --nnodes=2 --nproc_per_node=2 --rdzv_id=$SLURM_JOBID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT $RUNCMD

