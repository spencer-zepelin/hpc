#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <assert.h>
#include <string.h>

#define MAX 1000
#define CHUNKSIZE 10000

/*** LABELS FOR RUN MODE ***/
#define STATIC 0
#define DYNAMIC 1

/*
TODO
check for correct file open? 
different send/recv types?
macros for hardcoded values
check > < conditionals
correct number of args?

make sure timer is correct
barrier to ensure correct runtime?
*/


int main(int argc, char ** args){

	// Parse runtype from command line
	int runtype;
	if (! strcmp(args[1], "static")){
		runtype = STATIC;
	} else if (! strcmp(args[1], "dynamic")){
    	runtype = DYNAMIC;
    } else { // Invalid argument
        printf("ERROR: Improper run type: %s\nRun type should be one of the following options:\n\nstatic\ndynamic\n\n", args[1]);
        return EXIT_FAILURE;
   	}

	/*** MPI Setup ***/
    // Initialize relevant MPI variables
    int nprocs; // number of processes (aka "procs") used in this invocation
    int mype; // my process id (from 0 .. nprocs-1)
    int stat; // used as error code for MPI calls
    MPI_Status status; // MPI specific error variable
    MPI_Init(&argc, &args); // do this first to init MPI
    // Start timer
	double starttime = MPI_Wtime();
    stat = MPI_Comm_size(MPI_COMM_WORLD, &nprocs); // get number of procs
    assert(stat == MPI_SUCCESS); // Check to make sure call worked
    stat = MPI_Comm_rank(MPI_COMM_WORLD, &mype); // get my integer proc id
    assert(stat == MPI_SUCCESS); // Check to make sure call worked

    // Necessary variables
	MPI_File fh;
	int N = atoi(args[2]);
	float cx = -0.7; // args[2]
	float cy = 0.26; // args[3]
	float delt_x = 3.0/N; // 3 hardcoded?
	float delt_y = 2.0/N; // 2 hardcoded?

    /*** STATIC DECOMPOSITION BRANCH ***/
    if (runtype == STATIC){
	    int start_index, end_index, worksize; // Note START INCLUSIVE, END EXCLUSIVE
		// Fractional amount of work for each rank
		float frac_chunk  = (N * N) / ((float) nprocs);
	   
	    // Determine initial and final indices and size of data needed
	    for (int i = 0; i < nprocs; i++){
	    	if (mype == i){
	    		start_index = (int)(frac_chunk * mype);
	    		end_index = (int)(frac_chunk * (mype+1));
	    		worksize = end_index - start_index;
	    	}
	    }

	    // Allocate grid
	    int * P = malloc(sizeof(int) * worksize); 

	    if (mype == 0){
	    	printf("\n--Running Static Decomposition--\n%d processors each calculating approximately %d cells\n", nprocs, worksize);
	    	printf("Estimated memory use per processor: %.2e KB\n\n", sizeof(int) * worksize / 1024.0);
	    }

	    // All ranks perform calculations
		for (int i = 0; i < worksize; i++){
			int row = (start_index + i) / N;
			int column = (start_index + i) % N;
			float zx = -1.5 + (delt_x * row);
			float zy = -1.0 + (delt_y * column);
			int iteration = 0;
			while (((zx * zx) + (zy * zy)) < 4.0 && iteration < MAX ){
				float tmp = (zx * zx) - (zy * zy);
				zy = 2 * zx * zy + cy;
				zx = tmp + cx;
				iteration++;
			}
			// Data stored in personal data structure
			P[i] = iteration;
		}

		// File IO performed collectively
		// Write to file
	    MPI_File_open( MPI_COMM_WORLD, "data_mpi_static.bin", 
	    	MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);
	    // Set view for chunk of work
		MPI_File_set_view(fh, start_index *sizeof(int), MPI_INT, MPI_INT, "native", MPI_INFO_NULL);
		// Collective write
		MPI_File_write_all(fh, P, worksize, MPI_INT, &status);
		// Close file
		MPI_File_close(&fh);
    } 

    /*** DYNAMIC BRANCH ***/
    else if (runtype == DYNAMIC){
    	// Parse chunksize from command line
    	int chunk = atoi(args[3]);
    	// boss will use this for progress; workers for starting index
        int current_index = 0; 
	    // Everyone gets a work buffer
	    int * work_buffer = malloc(sizeof(int) * chunk);

	    if (mype == 0){
	    	printf("\n--Running Dynamic Load Balancing--\nTotal processors: %d\nChunk size: %d cells\n", nprocs, chunk);
	    	printf("Estimated memory use per processor: %.2e KB\n\n", sizeof(int) * chunk / 1024.0);
	    }

	    // Log of who is working on what 
		int log[nprocs];
		// How many workers still working
		int procs_working = nprocs - 1;

	 	// Send out initial assignments
	    for (int i = 1; i < nprocs; i++){
	    	if (mype == 0){
	    		// SEND position
	    		MPI_Send(&current_index, 1, MPI_INT, i, 99, MPI_COMM_WORLD);
	    		// Record current position of worker
	    		log[i] = current_index;
	    		// Advance Position
	    		current_index = current_index + chunk;
	    	}
	    	if (mype == i){
	    		// RECEIVE position
	    		MPI_Recv(&current_index, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	    	}
	    }

	    // Only boss needs to open file
	    if (mype == 0){
	    	MPI_File_open(MPI_COMM_SELF, "data_mpi_dynamic.bin", 
	    		MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);
	    }
	    		
	    // Code for worker nodes
	    if (mype != 0){
	    	// Proc will terminate when there is no more work to be done
	    	while (current_index < N * N){
	    		int loops = chunk;
	    		// Conditional for final chunk to prevent unnecessary work
	    		if ( (current_index + chunk) > (N * N) ){
	    			loops = (N * N) - chunk;
	    		}
	    		// Make calculations
		    	for (int i = 0; i < loops; i++){
					int row = (current_index + i) / N;
					int column = (current_index + i) % N;
					float zx = -1.5 + (delt_x * row);
					float zy = -1.0 + (delt_y * column);
					int iteration = 0;
					while (((zx * zx) + (zy * zy)) < 4.0 && iteration < MAX ){
						float tmp = (zx * zx) - (zy * zy);
						zy = 2 * zx * zy + cy;
						zx = tmp + cx;
						iteration++;
					}
					// Store in buffer
					work_buffer[i] = iteration;
				}
				// Send results to boss
				MPI_Send(work_buffer, chunk, MPI_INT, 0, mype, MPI_COMM_WORLD); 
				// Receive next assignement from boss
				MPI_Recv(&current_index, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			}
		}

	    // Code for boss
	    else {
	    	// Ensure there is still work to be done
			while (procs_working != 0){
				// Receive results from anyone
				MPI_Recv(work_buffer, chunk, MPI_INT, MPI_ANY_SOURCE, 
					MPI_ANY_TAG, MPI_COMM_WORLD, &status);
				// Sender's Identity
				int sender = status.MPI_SOURCE;
				// Give new assignment to sender
				MPI_Send(&current_index, 1, MPI_INT, sender, 99, MPI_COMM_WORLD);
	    		// If no more work for that proc
	    		if (current_index >= N * N){
	    			// Decrement number of procs still working
	    			procs_working--;
	    		}
				// Advance file pointer to write position
				MPI_File_seek(fh, log[sender] * sizeof(int), MPI_SEEK_SET);
				// If writing final chunk, only write to terminal position, not beyond 
				if ( (log[sender] + chunk) > (N * N) ){
					MPI_File_write(fh, work_buffer, (N * N) - log[sender], MPI_INT, &status);
				// Otherwise, write full buffer
				} else { 
					MPI_File_write(fh, work_buffer, chunk, MPI_INT, &status);
				}
	    		// Record current position of worker
	    		log[sender] = current_index;
	    		// Advance position
	    		current_index = current_index + chunk;
			}
	    }
	    // Close file
		if (mype == 0){
			MPI_File_close(&fh);
		}
	}
	/*** Finalize Both Branches ***/
	// Print total runtime
    if (mype == 0){
    	printf("Total runtime: %f s\n\n", MPI_Wtime() - starttime);
    }
	MPI_Finalize();
	return EXIT_SUCCESS;
}










