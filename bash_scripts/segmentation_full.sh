#!/bin/bash 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --time=12:00:00
#SBATCH --partition=aa100
#SBATCH --output=train_segmentation_job.%j.out
#SBATCH --job-name=train_segmentation

module purge
module load anaconda
conda activate ctdl3

python -m src_segmentation.training_seg --num-workers 36 --batch-size 16 --epochs 20 --augment-all final_seg
