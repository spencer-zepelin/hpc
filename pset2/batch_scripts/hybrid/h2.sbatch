#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --partition=broadwl
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
module load mpich
mpiexec -n 16 ./mpirun 10000 200 1.0 1.0e3 5.0e-7 2.85e-7 4 hybrid
