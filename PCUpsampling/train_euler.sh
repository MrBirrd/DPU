#!/bin/bash

#SBATCH -n 8
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=8
#SBATCH --gres=gpumem:20g
#SBATCH -A ls_drzrh

python train_upsampling.py --distribution_type multi --config configs/st_arkit.yml --save_dir /cluster/scratch/matvogel/DPU/ --model_path /cluster/scratch/matvogel/DPU/st_arkit/pretrain.pth