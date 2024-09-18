#!/bin/bash

#SBATCH --account=arupbio-kp
#SBATCH --partition=arup-kp
#SBATCH --nodes=1
#SBATCH --time=8-0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brendan.ofallon@aruplab.com


module load gcc/11.2.0

# Cant activate a conda env non-interactively, so just set the  python binary
# to the right spot - seems to work
PYTHON=$HOME/miniconda3/envs/py3/bin/python

set -x

ds2s=/uufs/chpc.utah.edu/common/home/u0379426/src/jovian/src/dnaseq2seq/main.py

#CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/wgs_lcbig_sus_chrs21and22.yaml
#CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/wgs_lcbig_sus_5more_chrsE.yaml
#CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/dnaseq2seq/decoder_fpfn_chr1_conf.yaml

CONF=/uufs/chpc.utah.edu/common/home/u0379426/src/jovian/pregen_confs/ont_first4_hard.yaml
DEST=/scratch/general/vast/u0379426/pregen/ont_first4_hard


BATCH_SIZE=512
READ_DEPTH=150
JITTER=0

mkdir -p $DEST
cp $CONF $DEST/

$PYTHON $ds2s pregen -c $CONF -d $DEST --threads 24 --jitter $JITTER --batch-size $BATCH_SIZE --read-depth $READ_DEPTH > $DEST/stdout.log 

