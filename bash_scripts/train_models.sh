#!/bin/bash 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --time=2:00:00
#SBATCH --partition=aa100
#SBATCH --output=trials_job.%j.out
#SBATCH --job-name=trials


python -m src.training --num-workers 24 --epochs 20 --balanced --init-weights-path inital_weights.pth  --augment-offset --augment-flip --tb-dir final_test offset-flip
python -m src.training --num-workers 24 --epochs 20 --balanced --init-weights-path inital_weights.pth  --augment-offset --augment-scale --tb-dir final_test offset-scale
python -m src.training --num-workers 24 --epochs 20 --balanced --init-weights-path inital_weights.pth  --augment-flip --augment-scale --tb-dir final_test flip-scale
python -m src.training --num-workers 24 --epochs 20 --balanced --init-weights-path inital_weights.pth  --augment-rotate --augment-scale --tb-dir final_test rotate-scale
python -m src.training --num-workers 24 --epochs 20 --balanced --init-weights-path inital_weights.pth  --augment-offset --tb-dir final_test offset
python -m src.training --num-workers 24 --epochs 20 --balanced --init-weights-path inital_weights.pth  --augment-scale --tb-dir final_test scale
python -m src.training --num-workers 24 --epochs 20 --balanced --init-weights-path inital_weights.pth  --augment-rotate --tb-dir final_test rotate
python -m src.training --num-workers 24 --epochs 20 --balanced --init-weights-path inital_weights.pth  --augment-flip --tb-dir final_test flip
python -m src.training --num-workers 24 --epochs 20 --balanced --init-weights-path inital_weights.pth  --augment-all --tb-dir final_test all 
python -m src.training --num-workers 24 --epochs 20 --balanced --init-weights-path inital_weights.pth  --tb-dir final_test base

