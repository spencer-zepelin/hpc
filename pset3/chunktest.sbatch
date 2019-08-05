#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --partition=broadwl
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=28
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
module load mpich
mpirun -n 56 ./julia_modes dynamic 12096 128
mpirun -n 56 ./julia_modes dynamic 12096 256
mpirun -n 56 ./julia_modes dynamic 12096 512
mpirun -n 56 ./julia_modes dynamic 12096 1024
mpirun -n 56 ./julia_modes dynamic 12096 2048
