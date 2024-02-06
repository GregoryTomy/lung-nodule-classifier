#!/bin/bash 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=30
#SBATCH --time=5:00:00
#SBATCH --partition=aa100
#SBATCH --output=malignant_classification.%j.out
#SBATCH --job-name=malignant_classification

module purge
module load anaconda
conda activate ctdl3
python -m src_classification.training_cls \
    --malignant \
    --dataset MalignantLunaDataset \
    --finetune-depth 2 \
    --finetune \
    --num-workers 26 \
    --batch-size 24 \
    --epochs 10 \
    --balanced \
    --augment-all \
    --tb-dir malignant-finetune augment_all
