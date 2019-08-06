#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
module load mpich
mpirun -n 8 ./julia_modes static 12096
mpirun -n 8 ./julia_modes dynamic 12096 4096