#!/bin/bash

MASTER_ADDR=$1
MASTER_PORT=$2
RUN_NAME=$3
RUN_NOTES="$4"



RUNCMD="jovian/dnaseq2seq/main.py train \
    --conf $HOME/src/jovian/train_conf_100M_bigenc.yaml \
    -o ${RUN_NAME}.model \
    --run-name $RUN_NAME"

echo "Full run cmd: $RUNCMD"
echo "Master addr: $MASTER_ADDR, master port: $MASTER_PORT"


export ENABLE_COMET=
export COMET_GIT_DIRECTORY=jovian/

$HOME/miniconda3/envs/py3/bin/torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=$SLURM_JOBID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT $RUNCMD

