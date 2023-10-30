#!/bin/bash

#SBATCH -n 2
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --gpus=2
#SBATCH --gres=gpumem:30g
#SBATCH -A ls_polle

python train_upsampling.py --distribution_type multi --config configs/scannet_cut_mink.yml --save_dir /cluster/scratch/matvogel/DPU/
