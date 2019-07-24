parallel: advection_parallel.c
	gcc -std=c99 -Wall -pedantic -Werror advection_parallel.c -o advection_parallel -fopenmp -lm

run: parallel
	./advection_parallel 400 20000 1.0 1.0e6 5.0e-7 2.85e-7 4

test: mpirun.c
	mpicc -o mpirun mpirun.c -fopenmp

try: test
	mpiexec -n 9 ./mpirun 30 10 1.0 1.0e6 5.0e-7 2.85e-7 1 mpi_non_blocking

mpi: test.c
	mpicc -o testblock test.c -fopenmp

check: testblock
	mpiexec -n 2 ./testblock

.PHONY: clean
clean:
	rm -f advection_parallel f0.txt f1.txt f2.txt mpirun
