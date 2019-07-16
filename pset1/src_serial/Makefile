serial: advection.c
	gcc -std=c99 -Wall -pedantic -Werror advection.c -o advection -lm

run: serial
	./advection 400 20000 1.0 1.0e6 5.0e-7 2.85e-7

.PHONY: clean
clean:
	rm -f advection f0.txt f1.txt f2.txt