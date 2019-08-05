#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
module load mpich
mpirun -n 4 ./julia_modes static 12096
mpirun -n 4 ./julia_modes dynamic 12096 4096