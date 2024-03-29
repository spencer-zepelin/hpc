#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <cuda.h> // cuda/6.5  

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif
    

/** Hardcoded Variables **/
#define WMAX 10.0
#define RADIUS 6.0
#define WINDOW_DISTANCE 10.0

/** Used for vector indexing **/
#define X 0
#define Y 1
#define Z 2

#define THREADSPERBLOCK 128

__global__ void ray_thread(double *G, int *n, double *wmax, double *r, double *L, double *c);
__device__ double dot3(double * vec1, double * vec2);
__device__ double mag3(double * vec);
__device__ void scale3(double scalar, double * in_vec, double * out);
__device__ void subvec3(double * vec1, double * vec2, double * diff);
__device__ int gridindex(double * vec, int grid_dim, double window_dim);
__device__ void sample_vec(double * vec, uint64_t * seed);
__device__ double LCG_random_double(uint64_t * seed);
__device__ uint64_t fast_forward_LCG(uint64_t seed, uint64_t n);
static __inline__ __device__ double atomicAdd(double *address, double val);

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

	// user-defined number of rays
	int num_rays = atoi(args[2]);

	// Calculate number of blocks based on threads per and total rays
	int blocks = (num_rays + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
	
	// Below will all be passed to kernel
	// user-defined grid dimension
	int n = atoi(args[1]);
	// Hardcoded variables
	double wmax = WMAX;
	double r = RADIUS;
	double L[3] =  {4.0, 4.0, -1.0};
	double c[3] = {0.0, 12.0, 0.0};

	// Allocate space for device variables
	// grid dimension
	int *d_n;
	cudaMalloc((void **)&d_n, sizeof(int));
	cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);
	// window dimension
	double *d_wmax;
	cudaMalloc((void **)&d_wmax, sizeof(double));
	cudaMemcpy(d_wmax, &wmax, sizeof(double), cudaMemcpyHostToDevice);
	// radius
	double *d_r;
	cudaMalloc((void **)&d_r, sizeof(double));
	cudaMemcpy(d_r, &r, sizeof(double), cudaMemcpyHostToDevice);
	// light source coordinates
	double *d_L;
	cudaMalloc((void **)&d_L, 3 * sizeof(double));
	cudaMemcpy(d_L, L, 3 * sizeof(double), cudaMemcpyHostToDevice);
	// sphere coordinates
	double *d_c;
	cudaMalloc((void **)&d_c, 3 * sizeof(double));
	cudaMemcpy(d_c, c, 3 * sizeof(double), cudaMemcpyHostToDevice);

	// Host and device grids
	double *h_G;
	double *d_G;

	// allocate host memory for the grid and initialize to 0
	h_G = (double *) calloc(n * n, sizeof(double)); // calloc initializes allocated memory to 0

    // allocate device memory
    cudaMalloc((void**)&d_G, n * n * sizeof(double));
    // Set CUDA memory to 0
    cudaMemset(d_G, 0, n * n * sizeof(double));

    // Initialize the kernel and run GPU code
    ray_thread<<<blocks,THREADSPERBLOCK>>>(d_G, d_n, d_wmax, d_r, d_L, d_c);

    // Copy device grid back to host
    cudaMemcpy(h_G, d_G, n * n * sizeof(double), cudaMemcpyDeviceToHost);

    // Write data to file
	FILE * f0 = fopen("data.bin", "wb"); 
	fwrite(h_G, sizeof(double), n * n, f0);
	
	// Close file and free memory
	fclose(f0);
	free(h_G);
	cudaFree(d_n);
	cudaFree(d_wmax);
	cudaFree(d_r);
	cudaFree(d_L);
	cudaFree(d_c);
	cudaFree(d_G);

	// Calculate elapsed time
    double elapsed = ((double) (clock() - stopwatch)) / CLOCKS_PER_SEC;
    
    // Print total runtime
    printf("Total execution time: %.2f seconds\n", elapsed);
    return EXIT_SUCCESS;
}



/*** Helper Functions ***/

__global__ void ray_thread(double *G, int *n, double *wmax, double *r, double *L, double *c){
	// Undefined Variable Declarations
	// Declared here so each thread has own copy
	double v[3];
	double I[3];
	double N_pt1[3];
	double N[3];
	double S_pt1[3];
	double S[3];
	double t_test = 1.0; // dummy initial value to avoid triggering conditional erroneously
	double t, scalar, brightness;
	// Thread ID
	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	// Seed random number generator
	uint64_t seed = 424242;
	// Fast forward PRNG based on thread id
	seed = fast_forward_LCG(seed, 200 * tid);
	// Define window intersection outside acceptable bounds
	double w[3] = {*wmax + 1, WINDOW_DISTANCE, *wmax + 1};

	while ( fabs(w[X]) >= *wmax || fabs(w[Z]) >= *wmax || t_test <= 0 ){
		// Randomly select a new ray and store values in v
		sample_vec(v, &seed);
		// Calculate scalar value
		scalar = w[Y]/v[Y];
		// Calculate scalar w from vector v
		scale3( scalar, v, w );
		// Calculate t component necessary for realness test
		t_test = pow(dot3(v, c), 2) + (*r * *r) - dot3(c, c);
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
	int index = gridindex(w, *n, *wmax);
	atomicAdd( &G[index], brightness);
}


// Calculates 3d dot product
__device__ double dot3(double * vec1, double * vec2){
	double out = 0;
	for (int i = 0; i < 3; i++){
		out = out + (vec1[i] * vec2[i]);
	}
	return out;
}

// Calculates magnitude of 3d vector
__device__ double mag3(double * vec){
	double out = sqrt( (vec[0] * vec[0]) + (vec[1] * vec[1]) + (vec[2] * vec[2]) );
	return out;
}

// Calculates scalar from vector and scalar value
__device__ void scale3(double scalar, double * in_vec, double * out){
	for (int i = 0; i < 3; i++){
		out[i] = scalar * in_vec[i];
	}
}

// Perform vector subtraction
__device__ void subvec3(double * vec1, double * vec2, double * diff){
	for (int i = 0; i < 3; i++){
		diff[i] = vec1[i] - vec2[i];
	}
}

// Convert window position to grid index
__device__ int gridindex(double * vec, int grid_dim, double window_dim){
	// Correct offset to make positive
	vec[X] = vec[X] + window_dim;
	vec[Z] = vec[Z] + window_dim;
	// Scale between grid dimension and window dimension
	double ratio = grid_dim / (2 * window_dim);
	// Scale window dimension to grid dimension
	int row = vec[X] * ratio;
	int column = vec[Z] * ratio;
	// NOTE: column is now correct; row is flipped
	// Row n is row zero and row zero is row n
	// Below corrects
	row = abs(row - grid_dim);
	return (row * grid_dim) + column;
}

// Randomly sample rays
__device__ void sample_vec(double * vec, uint64_t * seed){
	double phi = LCG_random_double(seed) * 2 * M_PI;
	double cos_theta = -1.0 + (2.0 * LCG_random_double(seed));
	double sin_theta = sqrt(1 - pow(cos_theta, 2));
	vec[X] = sin_theta * cos(phi);
	vec[Y] = sin_theta * sin(phi);
	vec[Z] = cos_theta;
}

// PRNG
__device__ double LCG_random_double(uint64_t * seed){
	// LCG parameters
	const uint64_t m = 9223372036854775808ULL; // 2^63
	const uint64_t a = 2806196910506780709ULL;
	const uint64_t c = 1ULL;
	// update seed
	*seed = (a * (*seed) + c) % m;
	return (double) (*seed) / (double) m;
}

// Fast forward PRNG
__device__ uint64_t fast_forward_LCG(uint64_t seed, uint64_t n){
	const uint64_t m = 9223372036854775808ULL; // 2^63
	uint64_t a = 2806196910506780709ULL;
	uint64_t c = 1ULL;
	n = n % m;
	uint64_t a_new = 1;
	uint64_t c_new = 0;
	while (n > 0){
		if ( n & 1 ){
			a_new *= a;
			c_new = c_new * a + c;
		}
		c *= (a + 1);
		a *= a;
		n >>= 1;
	}
	return (a_new * seed + c_new) % m;
}

// Atomic addition function
static __inline__ __device__ double atomicAdd(double *address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    if (val==0.0)
        return __longlong_as_double(old);
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}


