#!/bin/bash
#SBATCH --account=cs_gy_6923-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out

source ~/venvs/dl-hw2/bin/activate
cd ~/project1/code
nvidia-smi
python train.py "$@"