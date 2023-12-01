#!/bin/bash

#SBATCH --account=arupbio-kp
#SBATCH --partition=arup-shared-kp
#SBATCH --ntasks=8
#SBATCH --mem=16G
#SBATCH --time=3-0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brendan.ofallon@aruplab.com



# Cant activate a conda env non-interactively, so just set the  python binary
# to the right spot - seems to work
PYTHON=$HOME/miniconda3/envs/py3/bin/python


$PYTHON /uufs/chpc.utah.edu/common/home/u0379426/src/jovian/scripts/emit_tnsus_multi.py $@
