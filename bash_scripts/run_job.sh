#!/bin/bash 
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=03:00:00
#SBATCH --partition=aa100
#SBATCH --output=training_job.%j.out
#SBATCH --job-name=training_augmentation

module purge
module load anaconda
conda activate ctdl
python -m src.training --num-workers 16 --epochs 10 --balanced --augment-offset --tb-dir small bal_offset
python -m src.training --num-workers 16 --epochs 10 --balanced --augment-scale --tb-dir small bal_scale
python -m src.training --num-workers 16 --epochs 10 --balanced --augment-rotate --tb-dir small bal_rotate
python -m src.training --num-workers 16 --epochs 10 --balanced --augment-noise --tb-dir small bal_noise
python -m src.training --num-workers 16 --epochs 10 --balanced --augment --tb-dir small bal_all
python -m src.training --num-workers 16 --epochs 10 --balanced --augment-flip --tb-dir small bal_flipped

