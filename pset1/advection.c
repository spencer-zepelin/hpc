#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>
void init_blob(double * array, int nx, int ny, double lx, double ly);
/* TODO
*/

int main(int argc, char ** args){
    
    // Check correct number of command line arguments
    if (argc != 7) {
        perror("Improper number of command line arguments.\nRequires N, NT, L, T, u, and v");
        return EXIT_FAILURE;
    }
    
    // Initialize timer
    clock_t stopwatch;
    // Start time
    stopwatch = clock();
    
    // ESTABLISH INPUT VARIABLES
    // Matrix dimension
    int N = atoi(args[1]);
    // Number of timesteps
    int NT = atoi(args[2]);
    // Physical Cartesian Domain Length
    double L = atof(args[3]); // todo does this need to be a double?
    // Total Physical Timespan
    double T = atof(args[4]);
    // X Velocity Scalar
    double u = atof(args[5]);
    // Y Velocity Scalar
    double v = atof(args[6]);
    printf("TEST PARAMETERS\n---\nMatrix Dimension:  %d\nNumber of Timesteps:  %d\nPhysical Cartesian Domain Length:  %f\nTotal Physical Timespan:  %.2e\nX velocity Scalar:  %.2e\nY velocity Scalar:  %.2e\n", N, NT, L, T, u, v);
    // we have two matrices of N x N doubles
    printf("Estimated Memory Use:  %.2e KB\n", sizeof(double) * 2 * N * N / 1024.0);
    
    // Allocate N x N grid for C^n_{i,j}
    double * current_step = (double *) malloc(sizeof(double) * N * N);
    // Allocate N x N grid for C^{n+1}_{i,j}
    double * next_step = (double *) malloc(sizeof(double) * N * N);
    
    // Delta x = L/N
    double delt_x = L / N;
    // Delta t = T/NT
    double delt_t = T / NT;
    
    // Courant stability condition
    double courant_right = delt_x / sqrt((pow(u, 2) + pow(v, 2)) * 2);
    // Asserts Courant Stability is met
    assert(delt_t <= courant_right);
    
    // Calculate Gaussian pulse initial condition
    init_blob(current_step, N, N, L, L);
    
    // File initialization
    FILE * f0 = fopen("f0.txt", "w"); // Initial
    FILE * f1 = fopen("f1.txt", "w"); // Midway
    FILE * f2 = fopen("f2.txt", "w"); // Final

    // Save initial values
    for ( int i = 0; i < N * N; i++){
        fprintf(f0, "%.2e ", current_step[i]);
        if ((i + 1) % N == 0){
            fprintf(f0, "\n");
        }
    }
    
    // Triple nested
    // Timestep Loop
    for (int n = 1; n <= NT; n++){
        
        // Save middle step results
        if (n == NT/2){
            for ( int i = 0; i < N * N; i++){
                fprintf(f1, "%.2e ", current_step[i]);
                if ((i + 1) % N == 0){
                    fprintf(f1, "\n");
                }
            }
        }
        
        // Row Loop
        for (int i = 0; i < N; i++){
            // Column Loop
            for (int j = 0; j < N; j++){
                
                // Determine array indices for neighbors
                int left = i * N + j - 1;
                int right = i * N + j + 1;
                int up = (i-1) * N + j;
                int down = (i+1) * N + j;

                // Correct indices as necessary for wraparound
                if (j == 0){
                    left += N;
                }
                if (j == (N - 1)){
                    right -= N;
                }
                if (i == 0){
                    up += N * N;
                }
                if (i == (N - 1)){
                    down -= N * N;
                }
                
                // LAX CALCULATION
                // left expression
                double lax1 = (current_step[left] + current_step[right] + current_step[up] + current_step[down])/4.0;
                // right expression
                double lax2 = ( (u * (current_step[down] - current_step[up])) + (v * (current_step[right] - current_step[left])) ) * delt_t / (2 * delt_x);
                // left minus right
                double lax_final = lax1 - lax2;
                // store result in "next step" array
                next_step[i * N + j] = lax_final;
            }
        }
        // swap pointers to prep for next time step
        double * temp = current_step;
        current_step = next_step;
        next_step = temp;
    }
    
    // Print results after final step to file
    for ( int i = 0; i < N * N; i++){
        fprintf(f2, "%.2e ", current_step[i]);
        if ((i + 1) % N == 0){
            fprintf(f2, "\n");
        }
    }
    
    // Close files
    fclose(f0);
    fclose(f1);
    fclose(f2);
    
    // Calculate elapsed time
    double elapsed = ((double) (clock() - stopwatch)) / CLOCKS_PER_SEC;
    
    // Print total runtime
    printf("Total execution time: %.2f seconds\n", elapsed);
    return EXIT_SUCCESS;
}

// BLOB INITIALIZATION HELPER FUNCTION
void init_blob(double * array, int nx, int ny, double lx, double ly){
    double x,y;
    double dx = lx/nx;
    double dy = ly/ny;

    for (int i = 0; i <= nx + 1; i++){
        x = -lx/2 + dx*i;
        for (int j=0;j<=ny+1;j++){
            y = -ly/2 + dy*j;
            array[i * nx + j] = exp(-(x*x + y*y)/(2*lx/16));
        }
    }
    return;
}
