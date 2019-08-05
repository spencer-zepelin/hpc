test: julia_serial.c julia_modes.c
	gcc -std=c99 -o julia_serial julia_serial.c
	mpicc -std=c99 -o julia_modes julia_modes.c

mpi: test
	mpiexec -n 2 ./julia_modes static 
	mpiexec -n 2 ./julia_modes dynamic 10000 10000

chunk: test
	mpiexec -n 2 ./julia_modes dynamic 10000 10000

serial: test
	./julia 1000

trials: latencybandwidth.c
	mpicc -std=c99 -o latencybandwidth latencybandwidth.c

latband: trials
	mpiexec -n 2 ./latencybandwidth

.PHONY: clean
clean:
	rm -f data.bin data_mpi_static.bin data_mpi_dynamic.bin julia_serial julia_modes latencybandwidth