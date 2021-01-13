#!/bin/bash

#SBATCH --job-name=Finetune_Pascal
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END
#SBATCH --mem=32G
#SBATCH --signal=SIGUSR1@90

. ~/.bashrc
conda activate DLS

echo $PWD
which python
echo $PYTHONPATH

for ckpt in 9 19 29 39 49 59 69 79 89 99
do
  python finetune_pascal.py --backbone_checkpoint $ckpt
done
