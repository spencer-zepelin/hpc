#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>

#define SERIAL 0
#define THREADS 1
#define MPI_BLOCKING 2
#define MPI_NON_BLOCKING 3
#define HYBRID 4

void init_blob(double * array, int nx, int ny, double lx, double ly);
void update_step(double * current, double * next_one, int N);
/* TODO
 * change back to generic timers
 *ifndef for pragmas
 */


/*
 PARALLEL REGIONS
 * blob initialization
 * update next step from current step via lax
 */

int main(int argc, char ** args){
    
    // Check correct number of command line arguments
    if (argc != 9) {
        perror("Improper number of command line arguments.\nRequires N, NT, L, T, u, v, nt, and run type.\n");
        return EXIT_FAILURE;
    }
    
    // Initialize timer
    double start_time = omp_get_wtime();
    
    // ESTABLISH INPUT VARIABLES
    // Matrix dimension
    int N = atoi(args[1]);
    // Number of timesteps
    int NT = atoi(args[2]);
    // Physical Cartesian Domain Length
    double L = atof(args[3]);
    // Total Physical Timespan
    double T = atof(args[4]);
    // X Velocity Scalar
    double u = atof(args[5]);
    // Y Velocity Scalar
    double v = atof(args[6]);
    // Number of Threads
    int nt = atoi(args[7]);
    
    // TODO add string compare to command line for run type and a control variable
    // This variable controls the type of run
    int runtype;
    if (! strcmp(args[8], "serial")){
        runtype = SERIAL;
    } else if (! strcmp(args[8], "threads"){
        runtype = THREADS;
    }else if (! strcmp(args[8], "mpi_blocking"){
        runtype = MPI_BLOCKING;
    }else if (! strcmp(args[8], "mpi_non_blocking"){
        runtype = MPI_NON_BLOCKING;
    }else if (! strcmp(args[8], "hybrid"){
        runtype = HYBRID;
    } else {
        perror("ERROR: Improper run type: \nRun type should be one of the following options:\n\nserial\nthreads\nmpi_blocking\nmpi_non_blocking\nhybrid\n\n");
        return EXIT_FAILURE;
    }
    
    int nprocs; // number of processes (aka "procs") used in this invocation
    int mype; // my process id (from 0 .. nprocs-1)
    int stat; // used as error code for MPI calls
              
    MPI_Init(&argc, &argv); // do this first to init MPI
              
    stat = MPI_Comm_size(MPI_COMM_WORLD, &nprocs); // get number of procs
    assert(stat == MPI_SUCCESS); // Check to make sure call worked
              
    stat = MPI_Comm_rank(MPI_COMM_WORLD, &mype); // get my integer proc id
    assert(stat == MPI_SUCCESS); // Check to make sure call worked
              
    printf("\nTEST PARAMETERS\n---\nMatrix Dimension:  %d\nNumber of Timesteps:  %d\nPhysical Cartesian Domain Length:  %f\nTotal Physical Timespan:  %.2e\nX velocity Scalar:  %.2e\nY velocity Scalar:  %.2e\nRun type:  %s\nNumber of processes:  %d\nNumber of threads:  %d\n", N, NT, L, T, u, v, args[8], nprocs, nt);
    // 16 because size of double is 8 and we have two matrices of N x N
    
    // TODO modify this line depending on which case is running
    // If MPI type, show memory per node
    printf("Estimated Memory Use: %.2e KB\n", sizeof(double) * 2 * N * N / 1024.0);
    // Vanity display of max threads available on system
    printf("Total Available Threads:  %d\n", omp_get_max_threads());
    
              
    // MPI Cartesian Grid Creation
               // TODO need to cast double to int??
    int nprocs_per_dim = sqrt(nprocs); // sqrt or cube root for 2D, 3D
    int dims[nprocs_per_dim], periodic[nprocs_per_dim], coords[nprocs_per_dim];
    MPI_Comm comm1d;
    dims[0] = nprocs_per_dim; // Number of MPI ranks in each dimension
    periodic[0] = 1; // Turn on/off periodic boundary conditions for each dimension
              
              
    // Create Cartesian Communicator
    MPI_Cart_create( MPI_COMM_WORLD, // Starting communicator we are going to draw from
                  nprocs_per_dim, // MPI grid n-dimensionality
                  dims, // Array holding number of MPI ranks in each dimension
                  periodic, // Array indicating if we want to use periodic BC's or not
                  1, // Yes/no to reordering (allows MPI to re-organize for better perf)
                  &comm1d ); // Pointer to our new Cartesian Communicator object
              
    // Extract this MPI rank's N-dimensional coordinates from its place in the MPI Cartesian grid
    MPI_Cart_coords(comm1d, // Our Cartesian Communicator Object
                  mype, // Our process ID
                  DIMENSION, // The n-dimensionality of our problem
                  coords); // An n-dimensional array that we will write this rank's MPI coords to

              
    // Determine 1D neighbor ranks for this MPI rank
    int left, right, up, down;
    // left and right neighbors
    MPI_Cart_shift(comm1d, // Our Cartesian Communicator Object
                 0, // Which dimension we are shifting
                 1, // Direction of the shift
                 &left, // Tells us our Left neighbor
                 &right); // Tells us our Right neighbor
    // top and bottom neighbors
    MPI_Cart_shift(comm1d, // Our Cartesian Communicator Object
                 1, // Which dimension we are shifting
                 1, // Direction of the shift
                 &up, // Tells us our Left neighbor
                 &down); // Tells us our Right neighbor
    
              
              
    // Allocate Local Matrices
    
    int N_l = (N / nprocs_per_dim);
    // Add 2 to N_1 for length and width for ghost cells
    double * data = (double *) malloc((N_l+2) * (N_1+2) * sizeof(double));
    double * data_new = (double *) malloc((N_l+2) * (N_1+2) * sizeof(double));
    // Initialize starting conditions....
              
    // TODO gaussian initialization for splits!
              
              
    // Set number of threads for the run
    omp_set_num_threads(nt);
    
    // Allocate N x N grid for C^n_{i,j}
    double * current_step = (double *) malloc(sizeof(double) * N * N); // todo need to cast?
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
    FILE * f0 = fopen("fp0.txt", "w"); // Initial
    FILE * f1 = fopen("fp1.txt", "w"); // Midway
    FILE * f2 = fopen("fp2.txt", "w"); // Final
    
    // Check that files open correctly
    if (f0 == NULL || f1 == NULL || f2 == NULL) {
        perror("ERROR: Files not opened correctly.");
        return EXIT_FAILURE;
    }
    
    // Save initial values
    for ( int i = 0; i < N * N; i++){
        fprintf(f0, "%.2e ", current_step[i]);
        if ((i + 1) % N == 0){
            fprintf(f0, "\n");
        }
    }
    
    // Triple nested
    // DO NOT PARALLELIZE TIMESTEPS
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

#pragma omp parallel for default(none) shared(delt_x,delt_t,u,v,N,current_step,next_step) schedule(static)
        for (int i = 0; i < N; i++){
            for (int j = 0; j < N; j++){
                
                // Determine array indices for neighbors
                int left = i * N + j - 1;
                int right = i * N + j + 1;
                int up = (i - 1) * N + j;
                int down = (i + 1) * N + j;

                // Correct indices as necessary for wraparound
                if (j == 0){
                    left += N;
                }
                if (j == (N-1)){
                    right -= N;
                }
                if (i == 0){
                    up += N * N;
                }
                if (i == (N-1)){
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
    
              
    MPI_Finalize(); // required to terminate cleanly
              
    // Print total runtime
    printf("Total execution time: %.4f seconds\n\n", omp_get_wtime() - start_time);
    return EXIT_SUCCESS;
}

// BLOB INITIALIZATION HELPER FUNCTION
void init_blob(double * array, int nx, int ny, double lx, double ly){
    double dx = lx/nx;
    double dy = ly/ny;
    // Parallelized loop: shared data structure, but no race conditions
#pragma omp parallel for default(none) shared(dx,dy,nx,ny,lx,ly,array) schedule(static)
    for (int i = 0; i < nx; i++){
        double x = -lx/2 + dx*i;
        for (int j = 0; j < ny; j++){
            double y = -ly/2 + dy*j;
            array[i * nx + j] = exp(-(x*x + y*y)/(2*lx/16));
        }
    }
    return;
}
