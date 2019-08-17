compile: raytrace.c
	gcc -std=c99 -Wall -pedantic -Werror raytrace.c -o raytrace

try: compile
	./raytrace 1000 1000000000

parallel: raytrace_cuda.cu
	nvcc -arch=sm_37 raytrace_cuda.cu -o raytrace_cuda

run: parallel
	./raytrace_cuda 1000 1000000000

.PHONY: clean
clean:
	rm -f raytrace raytrace_cuda data.bin
