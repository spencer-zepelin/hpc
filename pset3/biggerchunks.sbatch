#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --partition=broadwl
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=28
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
module load mpich
mpirun -n 56 ./julia_modes dynamic 12096 262144
mpirun -n 56 ./julia_modes dynamic 12096 524288
mpirun -n 56 ./julia_modes dynamic 12096 1048576
mpirun -n 56 ./julia_modes dynamic 12096 2097152
mpirun -n 56 ./julia_modes dynamic 12096 4194304
