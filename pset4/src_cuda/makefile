compile: raytrace_cuda.cu
	nvcc -arch=sm_37 raytrace_cuda.cu -o raytrace_cuda -lm

run: compile
	./raytrace_cuda 1000 100000000

.PHONY: clean
clean:
	rm -f raytrace_cuda data.bin
