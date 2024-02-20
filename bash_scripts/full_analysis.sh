#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --time=1:30:00
#SBATCH --partition=aa100
#SBATCH --output=full_analysis_redo.%j.out
#SBATCH --job-name=full_analysis_redo

module purge
module load anaconda
conda activate ctdl3
python -m src_analysis.grouping \
    --num-workers 34 \
    --run-validation  \
    -sm final_models/seg_redo_adam.best.state \
    --malignancy-model-path final_models/mal_2024-02-12_17.24_augment_all_4000000.best.state