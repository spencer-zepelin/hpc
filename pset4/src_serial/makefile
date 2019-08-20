compile: raytrace.c
	gcc -std=c99 -Wall -pedantic -Werror raytrace.c -o raytrace -lm

run: compile
	./raytrace 1000 100000000

.PHONY: clean
clean:
	rm -f raytrace data.bin
