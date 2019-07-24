#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include <string.h>

#define SERIAL 0
#define THREADS 1
#define MPI_BLOCKING 2
#define MPI_NON_BLOCKING 3
#define HYBRID 4

#define DIMENSION 2
#define TOP 0
#define BOTTOM 1
#define LEFT 2
#define RIGHT 3
#define SEND 4
#define WRITE 4

void init_blob_ghost(double * array, int n, double l, int startrow, int startcolumn, int procdim);
void timesteps_mpi(double * current_step, double * next_step, double ** ghosts, int procdim, double u, double v, double delt_x, double delt_t);
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
    } else if (! strcmp(args[8], "threads")){
        runtype = THREADS;
    }else if (! strcmp(args[8], "mpi_blocking")){
        runtype = MPI_BLOCKING;
    }else if (! strcmp(args[8], "mpi_non_blocking")){
        runtype = MPI_NON_BLOCKING;
    }else if (! strcmp(args[8], "hybrid")){
        runtype = HYBRID;
    } else {
        perror("ERROR: Improper run type: \nRun type should be one of the following options:\n\nserial\nthreads\nmpi_blocking\nmpi_non_blocking\nhybrid\n\n");
        return EXIT_FAILURE;
    }
    
    // Delta x = L/N
    double delt_x = L / N;
    // Delta t = T/NT
    double delt_t = T / NT;
    
    int nprocs; // number of processes (aka "procs") used in this invocation
    int mype; // my process id (from 0 .. nprocs-1)
    int stat; // used as error code for MPI calls
    MPI_Status status; // MPI specific error variable
    
    MPI_Init(&argc, &args); // do this first to init MPI
    
    stat = MPI_Comm_size(MPI_COMM_WORLD, &nprocs); // get number of procs
    assert(stat == MPI_SUCCESS); // Check to make sure call worked
    
    stat = MPI_Comm_rank(MPI_COMM_WORLD, &mype); // get my integer proc id
    assert(stat == MPI_SUCCESS); // Check to make sure call worked
    if (mype == 0){
        printf("\nTEST PARAMETERS\n---\nMatrix Dimension:  %d\nNumber of Timesteps:  %d\nPhysical Cartesian Domain Length:  %f\nTotal Physical Timespan:  %.2e\nX velocity Scalar:  %.2e\nY velocity Scalar:  %.2e\nRun type:  %s\nNumber of processes:  %d\nNumber of threads:  %d\n", N, NT, L, T, u, v, args[8], nprocs, nt);
        // 16 because size of double is 8 and we have two matrices of N x N
        
        // TODO modify this line depending on which case is running
        // If MPI type, show memory per node
        printf("Estimated Memory Use: %.2e KB\n", sizeof(double) * 2 * N * N / 1024.0);
        // Vanity display of max threads available on system
        printf("Total Available Threads:  %d\n", omp_get_max_threads());
    }
    
    // MPI Cartesian Grid Creation
    // TODO need to cast double to int??
    int nprocs_per_dim = sqrt(nprocs); // sqrt or cube root for 2D, 3D
    int dims[DIMENSION], periodic[DIMENSION], coords[DIMENSION];
    MPI_Comm comm2d;
    for (int i = 0; i < nprocs_per_dim; i++){
        dims[i] = nprocs_per_dim; // Number of MPI ranks in each dimension
        periodic[i] = 1; // Turn on/off periodic boundary conditions for each dimension
    }
    
    // Create Cartesian Communicator
    MPI_Cart_create( MPI_COMM_WORLD, // Starting communicator we are going to draw from
                    DIMENSION, // MPI grid n-dimensionality
                    dims, // Array holding number of MPI ranks in each dimension
                    periodic, // Array indicating if we want to use periodic BC's or not
                    1, // Yes/no to reordering (allows MPI to re-organize for better perf)
                    &comm2d ); // Pointer to our new Cartesian Communicator object
    
    // Extract this MPI rank's N-dimensional coordinates from its place in the MPI Cartesian grid
    MPI_Cart_coords(comm2d, // Our Cartesian Communicator Object
                    mype, // Our process ID
                    DIMENSION, // The n-dimensionality of our problem
                    coords); // An n-dimensional array that we will write this rank's MPI coords to
    
    // Determine 2D neighbor ranks for this MPI rank
    int left, right, up, down;
    // top and bottom neighbors
    MPI_Cart_shift(comm2d, // Our Cartesian Communicator Object
                   0, // Which dimension we are shifting
                   1, // Direction of the shift
                   &up, // Tells us our Left neighbor
                   &down); // Tells us our Right neighbor
    // left and right neighbors
    MPI_Cart_shift(comm2d, // Our Cartesian Communicator Object
                   1, // Which dimension we are shifting
                   1, // Direction of the shift
                   &left, // Tells us our Left neighbor
                   &right); // Tells us our Right neighbor
    
    printf("\nFrom process %d\n Coords:(%d,%d)\n Left:  %d\n Right: %d\n Up:    %d\n Down:  %d\n", mype, coords[0], coords[1], left, right, up, down);
    
    // Allocate Local Matrices
    // Dimension of proc array
    int N_proc = (N / nprocs_per_dim);
    // Allocate arrays
    double * data = (double *) malloc(N_proc * N_proc * sizeof(double));
    double * data_new = (double *) malloc(N_proc * N_proc * sizeof(double));
    // Allocate ghost cells
    double * top_ghosts = (double *) malloc(N_proc * sizeof(double));
    double * bottom_ghosts = (double *) malloc(N_proc * sizeof(double));
    double * left_ghosts = (double *) malloc(N_proc * sizeof(double));
    double * right_ghosts = (double *) malloc(N_proc * sizeof(double));
    double * send_buf = (double *) malloc(N_proc * sizeof(double));
    // Pointer holder
    double * ghosts[5];
    ghosts[0] = top_ghosts;
    ghosts[1] = bottom_ghosts;
    ghosts[2] = left_ghosts;
    ghosts[3] = right_ghosts;
    ghosts[4] = send_buf;
    // Initialize starting conditions....
    init_blob_ghost(data, N, L, coords[0] * N_proc, coords[1] * N_proc, N_proc);
    
    printf("before file\n");
    
    //    /* write to file*/
    //    MPI_File f0;
    //    if (mype == 0){
    //        MPI_File_open(MPI_COMM_WORLD, "fb0", MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &f0);
    //    }
    
    //    MPI_Barrier(MPI_COMM_WORLD);
    printf("after file\n");
    for (int n = 0; n < nprocs_per_dim; n++){
        for (int j = 0; j < N_proc; j++){
            for (int m = 0; m < nprocs_per_dim; m++){
                int sender;
                // coordinates of current sender
                int sender_coords[2] = {n, m};
                // function determining sender's rank based on coordinates
                MPI_Cart_rank(comm2d, sender_coords, &sender);
                
                // Send command
                /*
                 if (mype == sender){
                 MPI_Send(&data[N_proc * j], // pointer to row of data
                 N_proc, // Number of elements I am sending
                 MPI_DOUBLE, // datatype
                 0, // destination --> proc 0
                 0, // tag
                 comm2d); // communicator
                 }
                 */
                // Receive command
                if (mype == 0){
                    /* MPI_Recv(ghosts[WRITE], // pointer to buffer
                     N_proc, // number of elements
                     MPI_DOUBLE, // datatype
                     sender,
                     0, // tag
                     comm2d, // communicator
                     &status);
                     // write data to file
                     MPI_File_write(f0, ghosts[WRITE], N_proc, MPI_DOUBLE, &status);*/
                    printf("\nRank: %d\n", sender);
                }
                //            MPI_Barrier(MPI_COMM_WORLD);
            }
        }
    }
    //    if (mype == 0){
    //        MPI_File_close(&f0);
    //    }
    /*
     // loop through rows of procs
     for (int n = 0; n < nprocs_per_dim; n++){
     if (coords[0] == n || mype == 0){ // coords[0] represents row number
     // loop through rows internal to a proc
     for (int j = 0; j < N_proc; j++){
     // loop through columns of procs
     for (int m = 0; m < nprocs_per_dim; m++){
     
     // Send command
     if (coords[1] == m){ // coords[1] represents column number
     MPI_Send(&data[N_proc * j], // pointer to row of data
     N_proc, // Number of elements I am sending
     MPI_DOUBLE, // datatype
     0, // destination --> proc 0
     0, // tag
     // TODO is this the correct communicator vvv
     comm2d); // communicator
     }
     
     // Receive command
     if (mype == 0){
     int sender;
     // coordinates of current sender
     int sender_coords[2] = {n, m};
     // function determining sender's rank based on coordinates
     MPI_Cart_rank(comm2d, sender_coords, &sender);
     printf("\nRank: %d\n", sender);
     MPI_Recv(ghosts[WRITE], // pointer to buffer
     N_proc, // number of elements
     MPI_DOUBLE, // datatype
     sender,
     0, // tag
     comm2d, // communicator
     &status);
     // write data to file
     //MPI_File_write(f0, ghosts[WRITE], N_proc, MPI_DOUBLE, &status);
     }
     }
     }
     }
     MPI_Barrier(comm2d);
     }
     //MPI_File_close(&f0);
     
     // Write to file --- TODO
     */
    
    /*
     MPI_File file;
     
     MPI_File_open(MPI_COMM_WORLD, "data0.txt",
     MPI_MODE_CREATE|MPI_MODE_WRONLY,
     MPI_INFO_NULL, &file);
     //TODO do I need this vvvv
     //MPI_File_set_view(file, 0,  MPI_CHAR, localarray, "native", MPI_INFO_NULL);
     // blocks not on right edge of grid
     if (! coords[1] == (N_proc - 1)){
     double * onerow = (double *) malloc(N_proc * sizeof(double))
     } else {
     double * onerow = (double *) malloc(N_proc * sizeof(double))
     }
     
     
     
     
     for (int i = 0; i < N_proc; i++){
     for (int j = 0; j < N_proc; j++){
     onerow[i] = data[(i+1) * N_proc + j + 1]
     }
     if
     MPI_File_write_all(file, data_as_txt, N_proc, num_as_string, &status);
     }
     */
    /*
     for (int i = 0; i < NT; i++){
     // Handles calculation for current timestep
     timesteps_mpi(data, data_new, ghosts, N_proc, u, v, delt_x, delt_t);
     
     // swap pointers to prep for message passing and next timestep
     double * temp = data;
     data = data_new;
     data_new = temp;*/
    
    /*** MESSAGE PASSING ***/
    // send up and recieve from below
    /*
     MPI_Sendrecv(&data[0], // Data I am sending -- FIRST ROW
     N_proc, // Number of elements to send
     MPI_DOUBLE, // Type I am sending
     up, // Who I am sending to
     99, // Tag (I don't care)
     ghosts[BOTTOM], // Data buffer to receive to
     N_proc, // How many elements I am receiving
     MPI_DOUBLE, // Type
     down, // Who I am receiving from
     99, // Tag (I don't care)
     comm2d, // Our MPI Cartesian Communicator object
     &status); // Status Variable
     
     // send down and recieve from above
     MPI_Sendrecv(&data[N_proc * (N_proc - 1)], // Data I am sending -- LAST ROW
     N_proc, // Number of elements to send
     MPI_DOUBLE, // Type I am sending
     down, // Who I am sending to
     99, // Tag (I don't care)
     ghosts[TOP], // Data buffer to receive to
     N_proc, // How many elements I am receiving
     MPI_DOUBLE, // Type
     up, // Who I am receiving from
     99, // Tag (I don't care)
     comm2d, // Our MPI Cartesian Communicator object
     &status); // Status Variable
     
     // send left and recieve from right
     // load data into buffer
     for (int i = 0; i < N_proc; i++){
     ghosts[SEND][i] = data[i * N_proc];
     }
     // Send and receive
     MPI_Sendrecv(ghosts[SEND], // Data I am sending
     N_proc, // Number of elements to send
     MPI_DOUBLE, // Type I am sending
     left, // Who I am sending to
     99, // Tag (I don't care)
     ghosts[RIGHT], // Data buffer to receive to
     N_proc, // How many elements I am receiving
     MPI_DOUBLE, // Type
     right, // Who I am receiving from
     99, // Tag (I don't care)
     comm2d, // Our MPI Cartesian Communicator object
     &status); // Status Variable
     
     MPI_Barrier(comm2d);
     
     // send left and recieve from right
     // load data into buffer
     for (int i = 1; i <= N_proc; i++){
     ghosts[SEND][i] = data[(i * N_proc) - 1];
     }
     // Send and receive
     MPI_Sendrecv(ghosts[SEND], // Data I am sending
     N_proc, // Number of elements to send
     MPI_DOUBLE, // Type I am sending
     right, // Who I am sending to
     99, // Tag (I don't care)
     ghosts[LEFT], // Data buffer to receive to
     N_proc, // How many elements I am receiving
     MPI_DOUBLE, // Type
     left, // Who I am receiving from
     99, // Tag (I don't care)
     comm2d, // Our MPI Cartesian Communicator object
     &status); // Status Variable
     }
     */
    //MPI_File_close(&file);
    
    // Print total runtime
    printf("Total execution time on thread %d: %.4f seconds\n\n", mype, omp_get_wtime() - start_time);
    
    MPI_Finalize(); // required to terminate cleanly
    return EXIT_SUCCESS;
}




// BLOB INITIALIZATION HELPER FUNCTION
void init_blob_ghost(double * array, int n, double l, int startrow, int startcolumn, int procdim){
    double dx = l/n;
    double dy = l/n;
    // Parallelized loop: shared data structure, but no race conditions
    //#pragma omp parallel for default(none) shared(dx,dy,nx,ny,lx,ly,array) schedule(static)
    for (int i = 0; i < procdim; i++){
        double x = -l/2 + dx*startrow;
        for (int j = 0; j < procdim; j++){
            double y = -l/2 + dy*startcolumn;
            array[i * procdim + j] = exp(-(x*x + y*y)/(2*l/16));
        }
    }
    return;
}


void timesteps_mpi(double * current_step, double * next_step, double ** ghosts, int procdim, double u, double v, double delt_x, double delt_t){
    for (int i = 0; i < procdim; i++){
        for (int j = 0; j < procdim; j++){
            
            double up, down, left, right;
            
            // Determine values for neighbors and account for borders
            if (i == 0){
                up = ghosts[TOP][j];
            } else{
                up = current_step[(i - 1) * procdim + j];
            }
            if (i == (procdim-1)){
                down = ghosts[BOTTOM][j];
            }else{
                down = current_step[(i + 1) * procdim + j];
            }
            if (j == 0){
                left = ghosts[LEFT][i];
            }else{
                left = current_step[i * procdim + j - 1];
            }
            if (j == (procdim-1)){
                right = ghosts[RIGHT][i];
            }else{
                right = current_step[i * procdim + j + 1];
            }
            
            // LAX CALCULATION
            // left expression
            double lax1 = (left + right + up + down)/4.0;
            // right expression
            double lax2 = ( (u * (down - up)) + (v * (right - left)) ) * delt_t / (2 * delt_x);
            // left minus right
            double lax_final = lax1 - lax2;
            // store result in "next step" array
            next_step[i * procdim + j] = lax_final;
            
        }
    }
    return;
}

