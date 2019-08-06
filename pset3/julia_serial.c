#include <stdlib.h>
#include <stdio.h>

#define MAX 1000

int main(int agrc, char ** args){
	// Initialize necessary variables

	// Grid dimension
	int N = atoi(args[1]);
	float cx = -0.7; 
	float cy = 0.26; 
	float delt_x = 3.0/N;
	float delt_y = 2.0/N; 

	// Allocate grid
	int * P = malloc(sizeof(int) * N * N);
	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			float zx = -1.5 + (delt_x * i);
			float zy = -1.0 + (delt_y * j);
			int iteration = 0;

			while (((zx * zx) + (zy * zy)) < 4.0 && iteration < MAX ){
				float tmp = (zx * zx) - (zy * zy);
				zy = 2 * zx *zy + cy;
				zx = tmp + cx;
				iteration++;
			}
			P[i * N + j] = iteration;
		}
	}

	// Write to file
	FILE * f0 = fopen("data.bin", "wb"); 
	fwrite(P, sizeof(int), N * N, f0);

	return EXIT_SUCCESS;
}