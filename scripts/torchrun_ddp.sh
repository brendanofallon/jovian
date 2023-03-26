#!/bin/bash

MASTER_ADDR=$1
MASTER_PORT=$2
RUN_NAME=$3
RUN_NOTES="$4"

VAL_DIR=/uufs/chpc.utah.edu/common/home/u0379426/storage/pregen_lcbigmap_d150_chrs21and22
PREGEN_DIR=/scratch/general/vast/u0379426/pregen_lcbigmap_d150/



RUNCMD="jovian/dnaseq2seq/main.py train \
    -d $PREGEN_DIR \
    --val-dir $VAL_DIR \
    -n 100 \
    --batch-size 256 \
    --learning-rate 0.00005 \
    --checkpoint-freq 1 \
    -o ${RUN_NAME}.model \
    --threads 16 \
    -i /uufs/chpc.utah.edu/common/home/arup-storage3/u0379426/variant_transformer_runs/96M_rampLR_lcbigmap2X_fixalltheissues_cont/96M_rampLR_lcbigmap2X_fixalltheissues_cont_epoch28_unwrapped.model \
    --max-decomp-batches 8 \
    --samples-per-epoch 1000000 \
    --wandb-run-name $RUN_NAME"

echo "Full run cmd: $RUNCMD"
echo "Master addr: $MASTER_ADDR, master port: $MASTER_PORT"

export ENABLE_WANDB=1

$HOME/miniconda3/envs/jv/bin/torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=$SLURM_JOBID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT $RUNCMD

