#!/bin/bash

MASTER_ADDR=$1
MASTER_PORT=$2
RUN_NAME=$3
RUN_NOTES="$4"



RUNCMD="jovian/dnaseq2seq/main.py train \
    --conf $HOME/src/jovian/train_conf_50M.yaml \
    -o ${RUN_NAME}.model \
    --input-model /uufs/chpc.utah.edu/common/home/arup-storage4/brendan/variant_transformer_runs/d128_testconf_50M/d128_testconf_50M_epoch120.model \
    --run-name $RUN_NAME"

echo "Full run cmd: $RUNCMD"
echo "Master addr: $MASTER_ADDR, master port: $MASTER_PORT"


export ENABLE_COMET=1
export COMET_GIT_DIRECTORY=jovian/

$HOME/miniconda3/envs/py3/bin/torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=$SLURM_JOBID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT $RUNCMD

