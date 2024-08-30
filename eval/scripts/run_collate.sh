#!/bin/bash -l

#SBATCH --account=arupbio-kp
#SBATCH --partition=arup-kp
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=0-12

COLLATE="${HOME}/src/jovian/eval/collate.py"

TAG_A=$1
TAG_B=$2

python $COLLATE $TAG_A *_${TAG_A}_sorted_happyresults > collated_${TAG_A}-${TAG_B}.csv
python $COLLATE $TAG_B *_${TAG_B}_sorted_happyresults | tail -n +2 >> collated_${TAG_A}-${TAG_B}.csv


