#!/bin/bash

#SBATCH --account=arupbio-kp
#SBATCH --partition=arup-kp
#SBATCH --nodes=1
#SBATCH --time=8-0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brendan.ofallon@aruplab.com



# Cant activate a conda env non-interactively, so just set the  python binary
# to the right spot - seems to work
PYTHON=$HOME/miniconda3/envs/ds2s/bin/python

ds2s=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/dnaseq2seq/bin/main.py

#CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/multindel_novaav_conf.yaml
#CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/wgs_multindel_conf.yaml
CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/wgs_bigvars_conf.yaml

DEST=/uufs/chpc.utah.edu/common/home/arup-storage3/u0379426/wgs_pregen_bigvars

BATCH_SIZE=512

mkdir -p $DEST

$PYTHON $ds2s pregen -c $CONF -d $DEST --threads 24 --batch-size $BATCH_SIZE > $DEST/stdout.log 

