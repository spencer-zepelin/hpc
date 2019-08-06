#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --partition=broadwl
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=28
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
module load mpich
mpirun -n 56 ./julia_modes dynamic 12096 4096
mpirun -n 56 ./julia_modes dynamic 12096 8192
mpirun -n 56 ./julia_modes dynamic 12096 16384
mpirun -n 56 ./julia_modes dynamic 12096 32768
mpirun -n 56 ./julia_modes dynamic 12096 65536
