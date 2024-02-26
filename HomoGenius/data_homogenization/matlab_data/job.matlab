#!/bin/bash
#SBATCH -J Matlab
#SBATCH -p cnall
#SBATCH -N 1   
#SBATCH -o stdout.%j
#SBATCH -e stderr.%j
#SBATCH --no-requeue
#SBATCH --ntasks-per-node=56

module load soft/matlab/v2023a
matlab -r  "calcu_homo_3D_main601_1200;quit"
