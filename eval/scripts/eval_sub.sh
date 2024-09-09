#!/bin/bash

set -ex

INPUT_BED=$(readlink -f $1)

TAG_A=$2
TAG_B=$3

JOVROOT="${HOME}/src/jovian/" 
RUN_SCRIPT="${JOVROOT}/eval/scripts/call_eval.sh"
COLLATE_SCRIPT="${JOVROOT}/eval/scripts/run_collate.sh"
RESULT_ROOT="$HOME/storage/jovian/eval_results/"
CRAMS="$HOME/src/jovian/eval/afewcrams.txt"


RANDO=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 5)

RESULT_DIR="${RESULT_ROOT}/eval-${TAG_A}-${TAG_B}-$RANDO"


mkdir -p $RESULT_DIR
cp $INPUT_BED $RESULT_DIR/
cd $RESULT_DIR

REPO_ROOT="jenever"
SRC_REPO="https://github.com/brendanofallon/jovian"
#SRC_REPO="https://github.com/ARUP-NGS/jenever.git "

REPO_A="$REPO_ROOT-$TAG_A"
REPO_B="$REPO_ROOT-$TAG_B"


GIT_LFS_SKIP_SMUDGE=1 git clone -b $TAG_A --filter=blob:none $SRC_REPO "$REPO_A"
GIT_LFS_SKIP_SMUDGE=1 git clone -b $TAG_B --filter=blob:none $SRC_REPO "$REPO_B"


readarray -t files < "$CRAMS"

# Array to store job IDs
job_ids=()

# Loop over each file and submit a job
for file in "${files[@]}"; do
    job_id=$(sbatch -M notchpeak --parsable "$RUN_SCRIPT" "$REPO_B" "$INPUT_BED" "$file")
    echo "Submitted job for $file with ID: $job_id"
    job_ids+=("$job_id")
done

#for file in "${files[@]}"; do
#    job_id=$(sbatch -M notchpeak --parsable "$RUN_SCRIPT" "$REPO_A" "$INPUT_BED" "$file")
#    echo "Submitted job for $file with ID: $job_id"
#    job_ids+=("$job_id")
#done



# Create a string of job IDs separated by colons for the dependency
job_dependency=$(IFS=:; echo "${job_ids[*]}")

# Submit the dependent job
dependent_job=$(sbatch --dependency=afterok:$job_dependency --parsable "${COLLATE_SCRIPT}")
echo "Submitted collation job with ID: $dependent_job"

