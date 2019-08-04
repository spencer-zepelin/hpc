#! / bin /bash
#SBATCH --time =00:05:00
#SBATCH --partition=broadwl
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
module load mpich
mpirun -n 2 ./latencybandwidth