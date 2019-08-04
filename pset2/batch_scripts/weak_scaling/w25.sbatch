#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=25
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
module load mpich
mpiexec -n 25 ./mpirun 4000 300 1.0 1.0e3 5.0e-7 2.85e-7 1 mpi_blocking
mpiexec -n 25 ./mpirun 4000 300 1.0 1.0e3 5.0e-7 2.85e-7 1 mpi_non_blocking