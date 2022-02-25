#!/bin/bash

#SBATCH --account=arupbio-kp
#SBATCH --partition=arup-kp
#SBATCH --nodes=1
#SBATCH --time=2-0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brendan.ofallon@aruplab.com



# Cant activate a conda env non-interactively, so just set the  python binary
# to the right spot - seems to work
PYTHON=$HOME/miniconda3/envs/ds2s/bin/python

ds2s=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/main.py

#CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/multindel_novaav_conf.yaml
CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/wgs_multindel_conf.yaml


DEST=/uufs/chpc.utah.edu/common/home/arup-storage3/u0379426/pregen_wgs_readwindowfix_w150_chr21and22/

BATCH_SIZE=128

mkdir -p $DEST

$PYTHON $ds2s pregen -c $CONF -d $DEST --threads 16 --batch-size $BATCH_SIZE > $DEST/stdout.log 

