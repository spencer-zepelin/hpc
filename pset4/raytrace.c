#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

/** Hardcoded Variables **/
#define WMAX 10.0
#define RADIUS 6.0
#define WINDOW_DISTANCE 10.0

/** Used for vector indexing **/
#define X 0
#define Y 1
#define Z 2


double dot3(double * vec1, double * vec2);
double mag3(double * vec);
void scale3(double scalar, double * in_vec, double * out);
void subvec3(double * vec1, double * vec2, double * diff);
int gridindex(double * vec, int grid_dim, double window_dim);
void sample_vec(double * vec, uint64_t * seed);
double LCG_random_double(uint64_t * seed);
// double rand_val(double min, double max);


/***
NOTE: for the purpose of this assignment, 
all vectors are assumed to have dimensionality of 3

TODO

TIMING

seed random?
Validate random elements working correctly

random ray selection
finding position of ray in terms of gridspace

for the while coditional on the window check
should it be >= or just >
prompt has different specs:

section 3 -- find intersection of view with window: wx > wmax
section 4 -- algorithm: wx < wmax and wz < wmax


INPUTS for various starting positions of sphere and source
INPUTS for radius of sphere and Window Dimension (wmax)
INPUTS for grid dimension and number of rays


Cleanup variable declaration/definition


Convert arrays to structs??

Adding timing functionality



***/


int main(int argc, char ** args){

	// arg check
	if (argc != 3){
		printf("---Incorrect Arguments---\nProgram should be run with the following:\n    EXECUTABLE GRIDDIMENSION NUMBER-OF-RAYS\n");
		return EXIT_SUCCESS;
	}

	// Initialize timer
    clock_t stopwatch;
    // Start time
    stopwatch = clock();

	// user-defined grid dimension
	int n = atoi(args[1]);
	// user-defined number of rays
	int num_rays = atoi(args[2]);
	// Hardcoded variables
	double wmax = WMAX;
	double r = RADIUS;
	double L[3] =  {4.0, 4.0, -1.0};//{0.0, 0.0, 0.0};
	double c[3] = {0.0, 12.0, 0.0};

	// Undefined Variable Declarations
	double v[3];
	double I[3];
	double N_pt1[3];
	double N[3];
	double S_pt1[3];
	double S[3];
	double t_test, t, scalar, brightness;

	// allocate memory for the grid and initialize to 0
	double * G = (double *) calloc(n * n, sizeof(double)); // calloc initializes allocated memory to 0

	// Seed random number generator
	uint64_t seed = 424242;


	for (int i = 0; i < num_rays; i ++){
		// Define window intersection outside acceptable bounds
		double w[3] = {wmax + 1, WINDOW_DISTANCE, wmax + 1}; // todo might not need plus 1 depending on >= or >

		while ( fabs(w[X]) >= wmax || fabs(w[Z]) >= wmax || t_test <= 0 ){
			// Randomly select a new ray and store values in v
			sample_vec(v, &seed);
			// Calculate scalar value
			scalar = w[Y]/v[Y];
			// Calculate scalar w from vector v
			scale3( scalar, v, w );
			// Calculate t component necessary for realness test
			t_test = pow(dot3(v, c), 2) + (r * r) - dot3(c, c);
		}
		t = dot3(v, c) - sqrt(t_test);
		scale3( t, v, I);
		// I - c stored in N_pt1
		subvec3(I, c, N_pt1);
		// Divide difference by magnitude of difference; store in N
		scale3( 1/mag3(N_pt1), N_pt1, N);
		// L - I stored in S_pt1
		subvec3(L, I, S_pt1);
		// Divide difference by magnitude of difference; store in S
		scale3( 1/mag3(S_pt1), S_pt1, S);
		// Calculate brightness; if brightness < 0, use 0
		brightness = fmax(0, dot3(S, N));
		int index = gridindex(w, n, wmax);
		// G[index] = brightness;
		G[index] = G[index] + brightness;
	}

	FILE * f0 = fopen("data.bin", "wb"); 
	fwrite(G, sizeof(double), n * n, f0);
	fclose(f0);
	free(G);

	// Calculate elapsed time
    double elapsed = ((double) (clock() - stopwatch)) / CLOCKS_PER_SEC;
    
    // Print total runtime
    printf("Total execution time: %.2f seconds\n", elapsed);
    return EXIT_SUCCESS;
}

/*** Helper Functions ***/
// Calculates 3d dot product
double dot3(double * vec1, double * vec2){
	double out = 0;
	for (int i = 0; i < 3; i++){
		out = out + (vec1[i] * vec2[i]);
	}
	return out;
}

double mag3(double * vec){
	double out = sqrt( (vec[0] * vec[0]) + (vec[1] * vec[1]) + (vec[2] * vec[2]) );
	return out;
}

void scale3(double scalar, double * in_vec, double * out){
	for (int i = 0; i < 3; i++){
		out[i] = scalar * in_vec[i];
	}
}

void subvec3(double * vec1, double * vec2, double * diff){
	for (int i = 0; i < 3; i++){
		diff[i] = vec1[i] - vec2[i];
	}
}


int gridindex(double * vec, int grid_dim, double window_dim){
	// Correct offset to make positive
	vec[X] = vec[X] + window_dim;
	vec[Z] = vec[Z] + window_dim;
	// Scale between grid dimension and window dimension
	double ratio = grid_dim / (2 * window_dim);
	// Scale window dimension to grid dimension
	// int row = round(vec[X] * ratio);
	// int column = round(vec[Z] * ratio);
	//TODO testing different rounding function
	int row = vec[X] * ratio;
	int column = vec[Z] * ratio;
	// NOTE: column is now correct; row is flipped
	// Row n is row zero and row zero is row n
	// Below corrects
	row = abs(row - grid_dim);
	return (row * grid_dim) + column;
}

void sample_vec(double * vec, uint64_t * seed){
	double phi = LCG_random_double(seed) * 2 * M_PI;
	double cos_theta = -1.0 + (2.0 * LCG_random_double(seed));
	double sin_theta = sqrt(1 - pow(cos_theta, 2));
	vec[X] = sin_theta * cos(phi);
	vec[Y] = sin_theta * sin(phi);
	vec[Z] = cos_theta;
}

double LCG_random_double(uint64_t * seed){
	// LCG parameters
	const uint64_t m = 9223372036854775808ULL; // 2^63
	const uint64_t a = 2806196910506780709ULL;
	const uint64_t c = 1ULL;
	*seed = (a * (*seed) + c) % m;
	return (double) (*seed) / (double) m;
}



// double rand_val(double min, double max){
// 	double ratio = RAND_MAX / (max-min);
// 	return min + (rand() / ratio);


