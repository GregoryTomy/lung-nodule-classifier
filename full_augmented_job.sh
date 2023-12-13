#!/bin/bash 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --time=12:00:00
#SBATCH --partition=aa100
#SBATCH --output=train_full_rotate_scale_job.%j.out
#SBATCH --job-name=train_full_rotate_scale

module purge
module load anaconda
conda activate ctdl
python -m src.training --num-workers 36 --batch-size 32 --epochs 52 --balanced --augment-rotate --augment-scale --tb-dir full rotate-scale
