#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --partition=broadwl
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=18
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
module load mpich
mpiexec -n 36 ./mpirun 4800 300 1.0 1.0e3 5.0e-7 2.85e-7 1 mpi_blocking
mpiexec -n 36 ./mpirun 4800 300 1.0 1.0e3 5.0e-7 2.85e-7 1 mpi_non_blocking