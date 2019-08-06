prog: julia_serial.c julia_modes.c
	gcc -std=c99 -o julia_serial julia_serial.c
	mpicc -std=c99 -o julia_modes julia_modes.c

trials: latencybandwidth.c
	mpicc -std=c99 -o latencybandwidth latencybandwidth.c

serial: prog
	./julia_serial 1000

mpi: prog
	mpiexec -n 2 ./julia_modes static 1000
	mpiexec -n 2 ./julia_modes dynamic 1000 10000

latband: trials
	mpiexec -n 2 ./latencybandwidth

.PHONY: clean
clean:
	rm -f data.bin data_mpi_static.bin data_mpi_dynamic.bin julia_serial julia_modes latencybandwidth