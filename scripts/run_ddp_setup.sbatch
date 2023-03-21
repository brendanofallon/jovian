#!/bin/bash -l


#SBATCH --account=arup-gpu-np
#SBATCH --partition=arup-gpu-np
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10-0

# --- Setup In Master ---
# Master will be the one running this bash script (SLURM runs this only once)
# Get hostname and port on first node first process
# For the port see: https://unix.stackexchange.com/questions/55913/whats-the-easiest-way-to-find-an-unused-local-port
export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
#echo "Master addr: $master_addr, port: $master_port"


ROOT_DIR=/uufs/chpc.utah.edu/common/home/arup-storage3/u0379426/variant_transformer_runs/
REPO_BASE=/uufs/chpc.utah.edu/common/home/u0379426/src/jovian/

GIT_BRANCH="decoder_big"

export RUN_NAME="grrrrfixbazbar"
export RUN_NOTES="Test AdamW optimizer and ramping up LR"

cd $ROOT_DIR
mkdir -p $RUN_NAME
cd $RUN_NAME

git clone $REPO_BASE

cd jovian
git checkout $GIT_BRANCH
COMMIT=$(git rev-parse HEAD)

cd ..

echo "Branch: $GIT_BRANCH \n commit: $COMMIT \n" >> git_info.txt


TORCHRUN_SCRIPT=$HOME/src/jovian/scripts/torchrun_ddp.sh

# --- Call the Script which the User will Edit ---
# With srun this will be run on all nodes for all processes
#srun --gres=gpu:2 $RUN_SCRIPT --master_addr $master_addr --master_port $master_port
srun --nodes=1 $TORCHRUN_SCRIPT $MASTER_ADDR $MASTER_PORT $RUN_NAME "$RUN_NOTES"
