test: advection_modes.c
	mpicc -std=c99 -O3 -o mpirun advection_modes.c -fopenmp -lm

try: test
	mpiexec -n 36 ./mpirun 900 300 1.0 1.0e3 5.0e-7 2.85e-7 1 mpi_blocking

run: test
	./mpirun 90 300 1.0 1.0e6 5.0e-7 2.85e-7 2 threads	

.PHONY: clean
clean:
	rm -f f0.bin f1.bin f2.bin mpirun