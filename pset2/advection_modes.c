#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <string.h>

/*** LABELS FOR RUN MODE ***/
#define SERIAL 0
#define THREADS 1
#define MPI_BLOCKING 2
#define MPI_NON_BLOCKING 3
#define HYBRID 4

/*** LABELS FOR GHOST CELLS AND SEND BUFFERS ***/
#define TOP 0
#define BOTTOM 1
#define LEFT 2
#define RIGHT 3
#define SEND 4
#define WRITE 4
// Send buffer gets reused for the write portions
#define SENDLEFT 4
#define SENDRIGHT 5


void init_blob(double * array, int nx, int ny, double lx, double ly);
void init_blob_threads(double * array, int nx, int ny, double lx, double ly);
void init_blob_ghost(double * array, int n, double l, int startrow, int startcolumn, int procdim);
void init_blob_ghost_threads(double * array, int n, double l, int startrow, int startcolumn, int procdim);
void timesteps_nonmpi(double * current_step, double * next_step, int N, double u, double v, double delt_x, double delt_t);
void timesteps_nonmpi_threads(double * current_step, double * next_step, int N, double u, double v, double delt_x, double delt_t);
void timesteps_mpi(double * current_step, double * next_step, double ** ghosts, int procdim, double u, double v, double delt_x, double delt_t);
void timesteps_mpi_threads(double * current_step, double * next_step, double ** ghosts, int procdim, double u, double v, double delt_x, double delt_t);
//void file_write_mpi(FILE ** fh, int nprocs_per_dim, int N_proc, int mype, double ** ghosts, double * data, MPI_Status status);
void nonblocking_message_pass(int N_proc, double ** ghosts, double * data, int up, int down, int left, int right);
void blocking_message_pass(int N_proc, double ** ghosts, double * data, int up, int down, int left, int right, MPI_Status status);
void allocate_ghosts(double ** ghosts, int N_proc);
void mpi_neighbors(int nprocs, int mype, int * up, int * down, int * left, int * right);

/* TODO
 * enforce number of procs is square
 * close files AND FREE MEMORY
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
    
    // Delta x = L/N
    double delt_x = L / N;
    // Delta t = T/NT
    double delt_t = T / NT;
    
    // Courant stability condition
    double courant_right = delt_x / sqrt((pow(u, 2) + pow(v, 2)) * 2);
    // Asserts Courant Stability is met
    assert(delt_t <= courant_right);
    
    // File initialization
    FILE * f0 = fopen("f0.bin", "wb"); // Initial
    FILE * f1 = fopen("f1.bin", "wb"); // Midway
    FILE * f2 = fopen("f2.bin", "wb"); // FinaL
    
    // Check that files open correctly
    if (f0 == NULL || f1 == NULL || f2 == NULL) {
        perror("ERROR: Files not opened correctly.");
        return EXIT_FAILURE;
    }
    
    // Set number of threads for the run
    omp_set_num_threads(nt);
    
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
    
/*** NON-MPI BRANCHES ***/
    if (runtype == SERIAL || runtype == THREADS){
        printf("\nTEST PARAMETERS\n---\nMatrix Dimension:  %d\nNumber of Timesteps:  %d\nPhysical Cartesian Domain Length:  %f\nTotal Physical Timespan:  %.2e\nX velocity Scalar:  %.2e\nY velocity Scalar:  %.2e\nRun type:  %s\nNumber of processes:  1\nNumber of threads:  %d\n", N, NT, L, T, u, v, args[8], nt);
        printf("Estimated Memory Use: %.2e KB\n", sizeof(double) * 2 * N * N / 1024.0);
        // Vanity display of max threads available on system
        printf("Total Available Threads:  %d\n", omp_get_max_threads());
        // Allocate memory for data structures
        double * current_step = (double*) malloc(sizeof(double) * N * N);
        double * next_step = (double*) malloc(sizeof(double) * N * N);
/*** SERIAL BRANCH ***/
        if (runtype == SERIAL){
            init_blob(current_step, N, N, L, L);
//            fwrite(current_step, sizeof(double), N * N, f0);
            for (int n = 1; n <= NT; n++){
                timesteps_nonmpi(current_step, next_step, N, u, v, delt_x, delt_t);
                double * temp = current_step;
                current_step = next_step;
                next_step = temp;
//                if (n == NT/2){
//                    fwrite(current_step, sizeof(double), N * N, f1);
//                }
            }
//            fwrite(current_step, sizeof(double), N * N, f2);

        }
/*** OMP PARALLELISM BRANCH ***/
        else if (runtype == THREADS){
            init_blob_threads(current_step, N, N, L, L);
//            fwrite(current_step, sizeof(double), N * N, f0);
            for (int n = 1; n <= NT; n++){
                timesteps_nonmpi_threads(current_step, next_step, N, u, v, delt_x, delt_t);
                double * temp = current_step;
                current_step = next_step;
                next_step = temp;
//                if (n == NT/2){
//                    fwrite(current_step, sizeof(double), N * N, f1);
//                }
            }
//            fwrite(current_step, sizeof(double), N * N, f2);
        }
        //TODO CLOSE FILES AND FREE MEMORY
        printf("Total execution time: %.4f seconds\n\n", omp_get_wtime() - start_time);
    }
/*** MPI BRANCHES ***/
    else {
        // Initialize relevant MPI variables
        int nprocs; // number of processes (aka "procs") used in this invocation
        int mype; // my process id (from 0 .. nprocs-1)
        int stat; // used as error code for MPI calls
        MPI_Status status; // MPI specific error variable
        MPI_Init(&argc, &args); // do this first to init MPI
        stat = MPI_Comm_size(MPI_COMM_WORLD, &nprocs); // get number of procs
        assert(stat == MPI_SUCCESS); // Check to make sure call worked
        stat = MPI_Comm_rank(MPI_COMM_WORLD, &mype); // get my integer proc id
        assert(stat == MPI_SUCCESS); // Check to make sure call worked
        // Check that given number of procs is perfect square
        double square_check = sqrt(nprocs);
        assert(square_check - floor(square_check) == 0);
        // procs per dimension of proc grid
        int nprocs_per_dim = sqrt(nprocs);
        // Dimension of proc array
        int N_proc = (N / nprocs_per_dim);
        if (mype == 0){
            printf("\nTEST PARAMETERS\n---\nMatrix Dimension:  %d\nNumber of Timesteps:  %d\nPhysical Cartesian Domain Length:  %f\nTotal Physical Timespan:  %.2e\nX velocity Scalar:  %.2e\nY velocity Scalar:  %.2e\nRun type:  %s\nNumber of processes:  %d\nNumber of threads:  %d\n", N, NT, L, T, u, v, args[8], nprocs, nt);
            // Memory per node
            printf("Estimated Memory Use per Node: %.2e KB\n", ((2 * N_proc * N_proc) + (6 * N_proc)) * sizeof(double) / 1024.0);
            // Vanity display of max threads available on each node
            printf("Total Available Threads per Node:  %d\n", omp_get_max_threads());
        }
        // Determine identity of neighbors
        int up, down, left, right;
        mpi_neighbors(nprocs, mype, &up, &down, &left, &right);
        int grid_row = mype / nprocs_per_dim;
        int grid_column = mype % nprocs_per_dim;
        // Allocate memory for data structures
        double * current_step = (double*) malloc(sizeof(double) * N_proc * N_proc);
        double * next_step = (double*) malloc(sizeof(double) * N_proc * N_proc);
        double * ghosts[6];
        allocate_ghosts(ghosts, N_proc);
/*** MPI BLOCKING AND NON-BLOCKING BRANCH ***/
        if (runtype == MPI_BLOCKING || runtype == MPI_NON_BLOCKING){
            for (int i = 0; i < nprocs; i++){
                if (mype == i){
                    int startrow = grid_row * N_proc;
                    int startcolumn = grid_column * N_proc;
                    init_blob_ghost(current_step, N, L, startrow, startcolumn, N_proc);
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
//            file_write_mpi(&f0, nprocs_per_dim, N_proc, mype, ghosts, current_step, status);
            for (int n = 1; n <= NT; n++){
                if (n % 1000 == 0){
                    if (mype == 0){
                        printf("Completed %d timesteps.\n", n);
                    }
                }
                if (runtype == MPI_BLOCKING){
                    blocking_message_pass(N_proc, ghosts, current_step, up, down, left, right, status);
                } else {
                    nonblocking_message_pass(N_proc, ghosts, current_step, up, down, left, right);
                }
                timesteps_mpi(current_step, next_step, ghosts, N_proc, u, v, delt_x, delt_t);
                double * temp = current_step;
                current_step = next_step;
                next_step = temp;
//                if (n == NT/2){
//                    MPI_Barrier(MPI_COMM_WORLD);
//                    file_write_mpi(&f1, nprocs_per_dim, N_proc, mype, ghosts, current_step, status);
//                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
//            file_write_mpi(&f2, nprocs_per_dim, N_proc, mype, ghosts, current_step, status);
        }
/*** MPI-OMP HYBRID BRANCH ***/
        else if (runtype == HYBRID){
            for (int i = 0; i < nprocs; i++){
                if (mype == i){
                    int startrow = grid_row * N_proc;
                    int startcolumn = grid_column * N_proc;
                    init_blob_ghost_threads(current_step, N, L, startrow, startcolumn, N_proc);
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
//            file_write_mpi(&f0, nprocs_per_dim, N_proc, mype, ghosts, current_step, status);
            for (int n = 1; n <= NT; n++){
                if (n % 1000 == 0){
                    if (mype == 0){
                        printf("Completed %d timesteps.\n", n);
                    }
                }
                nonblocking_message_pass(N_proc, ghosts, current_step, up, down, left, right);
                timesteps_mpi_threads(current_step, next_step, ghosts, N_proc, u, v, delt_x, delt_t);
                double * temp = current_step;
                current_step = next_step;
                next_step = temp;
//                if (n == NT/2){
//                    MPI_Barrier(MPI_COMM_WORLD);
//                    file_write_mpi(&f1, nprocs_per_dim, N_proc, mype, ghosts, current_step, status);
//                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
//            file_write_mpi(&f2, nprocs_per_dim, N_proc, mype, ghosts, current_step, status);
        }
        // TODO CLOSE FILES AND FREE MEMORY
        // Print total runtime
        MPI_Barrier(MPI_COMM_WORLD);
        if (mype == 0){
            printf("Total execution time: %.4f seconds\n\n", omp_get_wtime() - start_time);
        }
        MPI_Finalize();
    }
/*** END OF MODE BRANCHES ***/
    return EXIT_SUCCESS;
}



/*** HELPER FUNCTIONS ***/

void init_blob(double * array, int nx, int ny, double lx, double ly){
    double x,y;
    double dx = lx/nx;
    double dy = ly/ny;
    
    for (int i = 0; i < nx; i++){
        x = -lx/2 + dx*i;
        for (int j = 0; j < ny;j++){
            y = -ly/2 + dy*j;
            array[i * nx + j] = exp(-(x*x + y*y)/(2*lx/16));
        }
    }
    return;
}

void init_blob_threads(double * array, int nx, int ny, double lx, double ly){
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

void init_blob_ghost(double * array, int n, double l, int startrow, int startcolumn, int procdim){
    double dx = l/n;
    double dy = l/n;
    for (int i = 0; i < procdim; i++){
        double x = -l/2 + dx*(startrow+i);
        for (int j = 0; j < procdim; j++){
            double y = -l/2 + dy*(startcolumn+j);
            array[i * procdim + j] = exp(-(x*x + y*y)/(2*l/16));
        }
    }
    return;
}

void init_blob_ghost_threads(double * array, int n, double l, int startrow, int startcolumn, int procdim){
    double dx = l/n;
    double dy = l/n;
    // Parallelized loop: shared data structure, but no race conditions
#pragma omp parallel for default(none) shared(dx,dy,procdim,startrow,startcolumn,l,array) schedule(static)
    for (int i = 0; i < procdim; i++){
        double x = -l/2 + dx*(startrow+i);
        for (int j = 0; j < procdim; j++){
            double y = -l/2 + dy*(startcolumn+j);
            array[i * procdim + j] = exp(-(x*x + y*y)/(2*l/16));
        }
    }
    return;
}

void timesteps_nonmpi(double * current_step, double * next_step, int N, double u, double v, double delt_x, double delt_t){
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
    return;
}


void timesteps_nonmpi_threads(double * current_step, double * next_step, int N, double u, double v, double delt_x, double delt_t){
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

void timesteps_mpi_threads(double * current_step, double * next_step, double ** ghosts, int procdim, double u, double v, double delt_x, double delt_t){
#pragma omp parallel for default(none) shared(delt_x,delt_t,u,v,ghosts,procdim,current_step,next_step) schedule(static)
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

//void file_write_mpi(FILE ** fh, int nprocs_per_dim, int N_proc, int mype, double ** ghosts, double * data, MPI_Status status){
//    for (int n = 0; n < nprocs_per_dim; n++){
//        for (int j = 0; j < N_proc; j++){
//            for (int m = 0; m < nprocs_per_dim; m++){
//                int sender = n * nprocs_per_dim + m;
//                MPI_Barrier(MPI_COMM_WORLD);
//                // Send command
//                if (mype == sender && sender != 0){
//                    MPI_Send(&data[N_proc * j], // pointer to row of data
//                             N_proc, // Number of elements I am sending
//                             MPI_DOUBLE, // datatype
//                             0, // destination --> proc 0
//                             99, // tag
//                             MPI_COMM_WORLD); // communicator
//                }
//
//                // Receive command
//                if (mype == 0 && sender != 0){
//                    MPI_Recv(ghosts[WRITE], // pointer to buffer
//                             N_proc, // number of elements
//                             MPI_DOUBLE, // datatype
//                             sender,
//                             99, // tag
//                             MPI_COMM_WORLD, // communicator
//                             &status);
//                }
//                // write data to file
//                if (mype == 0){
//                    if (sender == 0){
//                        fwrite(&data[N_proc * j], sizeof(double), N_proc, *fh);
//                    }else{
//                        fwrite(ghosts[WRITE], sizeof(double), N_proc, *fh);
//                    }
//                }
//            }
//        }
//    }
//    return;
//}


void nonblocking_message_pass(int N_proc, double ** ghosts, double * data, int up, int down, int left, int right){

    /*** MESSAGE PASSING ***/
    // load left and right column data into buffers
    for (int i = 0; i < N_proc; i++){
        ghosts[SENDLEFT][i] = data[i * N_proc];
        ghosts[SENDRIGHT][i] = data[((i+1) * N_proc) - 1];
    }

    MPI_Request request[8];
    // send up and recieve from below
    MPI_Isend(&data[0], N_proc, MPI_DOUBLE, up, 99, MPI_COMM_WORLD, &request[0]);
    MPI_Irecv(ghosts[BOTTOM], N_proc, MPI_DOUBLE, down, 99, MPI_COMM_WORLD, &request[1]);
    // send down and recieve from above
    MPI_Isend(&data[N_proc * (N_proc - 1)], N_proc, MPI_DOUBLE, down, 99, MPI_COMM_WORLD, &request[2]);
    MPI_Irecv(ghosts[TOP], N_proc, MPI_DOUBLE, up, 99, MPI_COMM_WORLD, &request[3]);
    // send left and receive from right
    MPI_Isend(ghosts[SENDLEFT], N_proc, MPI_DOUBLE, left, 99, MPI_COMM_WORLD, &request[4]);
    MPI_Irecv(ghosts[RIGHT], N_proc, MPI_DOUBLE, right, 99, MPI_COMM_WORLD, &request[5]);
    // send right and receive from left
    MPI_Isend(ghosts[SENDRIGHT], N_proc, MPI_DOUBLE, right, 99, MPI_COMM_WORLD, &request[6]);
    MPI_Irecv(ghosts[LEFT], N_proc, MPI_DOUBLE, left, 99, MPI_COMM_WORLD, &request[7]);

    // Ensure proper message passing conclusion
    MPI_Waitall(8, request, MPI_STATUSES_IGNORE);
    return;
}

void blocking_message_pass(int N_proc, double ** ghosts, double * data, int up, int down, int left, int right, MPI_Status status){
    // send up and recieve from below
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
                 MPI_COMM_WORLD, // Our MPI Cartesian Communicator object
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
                 MPI_COMM_WORLD, // Our MPI Cartesian Communicator object
                 &status); // Status Variable

    // send left and receive from right
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
                 MPI_COMM_WORLD, // Our MPI Cartesian Communicator object
                 &status); // Status Variable

    MPI_Barrier(MPI_COMM_WORLD);

    // send right and receive from left
    // load data into buffer
    for (int i = 0; i < N_proc; i++){
        ghosts[SEND][i] = data[((i+1) * N_proc) - 1];
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
                 MPI_COMM_WORLD, // Our MPI Cartesian Communicator object
                 &status); // Status Variable
    return;
}

void allocate_ghosts(double ** ghosts, int N_proc){
    // Allocate ghost cells
    double * top_ghosts = (double *) malloc(N_proc * sizeof(double));
    double * bottom_ghosts = (double *) malloc(N_proc * sizeof(double));
    double * left_ghosts = (double *) malloc(N_proc * sizeof(double));
    double * right_ghosts = (double *) malloc(N_proc * sizeof(double));
    double * send_buf = (double *) malloc(N_proc * sizeof(double));
    double * send_buf_right = (double *) malloc(N_proc * sizeof(double));
    // store in pointer holder
    ghosts[0] = top_ghosts;
    ghosts[1] = bottom_ghosts;
    ghosts[2] = left_ghosts;
    ghosts[3] = right_ghosts;
    ghosts[4] = send_buf;
    ghosts[5] = send_buf_right;
    return;
}

void mpi_neighbors(int nprocs, int mype, int * up, int * down, int * left, int * right){
      /*** DETERMINE ORTHOGONAL NEIGHBORS ***/
      int nprocs_per_dim = sqrt(nprocs); // sqrt or cube root for 2D, 3D
    
      int grid_row = mype / nprocs_per_dim;
      int grid_column = mype % nprocs_per_dim;
    
      if (grid_row != 0){ // not first row
          *up = mype - nprocs_per_dim;
      } else {
          *up = mype + nprocs - nprocs_per_dim;
      }
      if (grid_row != (nprocs_per_dim-1)){ // not last row
          *down = mype + nprocs_per_dim;
      } else {
          *down = mype + nprocs_per_dim - nprocs;
      }
      if (grid_column != 0){ // not left column
          *left = mype - 1;
      } else {
          *left = mype + nprocs_per_dim - 1;
      }
      if (grid_column != (nprocs_per_dim-1)){ // not right column
          *right = mype + 1;
      } else {
          *right = mype + 1 - nprocs_per_dim;
      }
    return;
}
