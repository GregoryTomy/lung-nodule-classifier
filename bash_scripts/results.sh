#!/bin/bash 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=30
#SBATCH --time=2:00:00
#SBATCH --partition=aa100
#SBATCH --output=results_job2.%j.out
#SBATCH --job-name=results

module purge
module load anaconda
conda activate ctdl
python -m src.evaluate --num-workers 24 --batch-size 32 --init-weights-path model_checkpoints/final.pth --tb-dir results