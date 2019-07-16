parallel: advection_parallel.c
	gcc -std=c99 -Wall -pedantic -Werror advection_parallel.c -o advection_parallel -fopenmp -lm

run: parallel
	./advection_parallel 400 20000 1.0 1.0e6 5.0e-7 2.85e-7 4

.PHONY: clean
clean:
	rm -f advection_parallel fp0.txt fp1.txt fp2.txt
