#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --time=1:30:00
#SBATCH --partition=aa100
#SBATCH --output=run_segmentation.%j.out
#SBATCH --job-name=run_segmentation

module purge
module load anaconda
conda activate ctdl3
python -m src_analysis.grouping \
    --num-workers 34 \
    --run-validation