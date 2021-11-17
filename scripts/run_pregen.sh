#!/bin/bash

#SBATCH --account=arupbio-kp
#SBATCH --partition=arup-kp
#SBATCH --nodes=1
#SBATCH --time=2-0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brendan.ofallon@aruplab.com


module load cuda/11.3

# Cant activate a conda env non-interactively, so just set the  python binary
# to the right spot - seems to work
PYTHON=$HOME/miniconda3/envs/ds2s/bin/python

ds2s=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/main.py

#CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/chpc_conf.yaml
CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/conf_nochr20_21_tns.yaml

DEST=/uufs/chpc.utah.edu/common/home/arup-storage3/u0379426/pregen_9feats_confvpc_tns/

VALS_PER_CLASS=500

mkdir -p $DEST

$PYTHON $ds2s pregen -c $CONF -d $DEST --threads 10 --vals-per-class $VALS_PER_CLASS > $DEST/stdout.log 

