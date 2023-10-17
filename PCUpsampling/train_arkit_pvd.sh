#!/bin/bash

#SBATCH -n 4
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=4
#SBATCH --gres=gpumem:20g
#SBATCH -A ls_drzrh

module load python_gpu/3.10.4

python train_upsampling.py --name pvd_arkit_pc2pc --distribution_type multi --config configs/pvd_arkit.yml --save_dir /cluster/scratch/matvogel/DPU/