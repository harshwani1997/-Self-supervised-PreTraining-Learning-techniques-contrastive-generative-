#!/bin/bash

#SBATCH --job-name=DLS_SWaV
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4:p40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END
#SBATCH --mem=96G
#SBATCH --signal=SIGUSR1@90

. ~/.bashrc
conda activate DLS

echo $PWD
which python
echo $PYTHONPATH

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_LL_THRESHOLD=0


python SWaV.py --gpus 4
