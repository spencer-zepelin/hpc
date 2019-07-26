#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --partition=broadwl
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
module load mpich
mpiexec -n 4 ./mpirun 10000 200 1.0 1.0e3 5.0e-7 2.85e-7 16 hybrid
