#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <assert.h>

#define NUM_LATENCY 100000

/*
TODO
check for correct file open? 
*/


int main(int argc, char ** args){
	

	// MPI Setup
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

/*** LATENCY ***/
   int * data_buf = malloc(sizeof(int) * NUM_LATENCY);

	double latency_start = MPI_Wtime();
	for (int i = 0; i < NUM_LATENCY; i++){
		if (mype == 0){
			MPI_Send(&data_buf[i], 1, MPI_INT, 1, 99, MPI_COMM_WORLD);
			MPI_Recv(&data_buf[i], 1, MPI_INT, 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		}
		if (mype == 1){
			MPI_Recv(&data_buf[i], 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			MPI_Send(&data_buf[i], 1, MPI_INT, 0, 99, MPI_COMM_WORLD);
		}
	}
	double latency_end = MPI_Wtime();
	double latency_avg_micro = 1000000 * (latency_end - latency_start) / NUM_LATENCY;
	if (mype == 0){
		printf("\nAverage latency is %f microseconds\n", latency_avg_micro);
	}

	free(data_buf);

/*** BANDWIDTH ***/
	// Allocate a little over 1 GB of memory
	int gig_size = 1024 * 1024 * 1024;
	int meg_size = 1024 * 1024;
	int one_gig_plus = gig_size + 100;
	char * band_buf = malloc(sizeof(char) * one_gig_plus);
	
	// KB test
	double kb_start = MPI_Wtime();
	for (int i = 0; i < 1000; i++){
		if (mype == 0){
			MPI_Send(&band_buf[i * 1024], 1024, MPI_BYTE, 1, 99, MPI_COMM_WORLD);
		}
		if (mype == 1){
			MPI_Recv(&band_buf[i * 1024], 1024, MPI_BYTE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	double kb_end = MPI_Wtime();
	double total_byes_sent = 1024 * 1000.0;
	double total_gigs_sent = total_byes_sent / gig_size;
	double secs_elapsed = kb_end - kb_start;
	double gbs = total_gigs_sent / secs_elapsed;
	if (mype == 0){
		printf("\nGB/s for KB packets: %f\n", gbs);
	}
	

	double mb_start = MPI_Wtime();
	for (int i = 0; i < 100; i++){
		if (mype == 0){
			MPI_Send(&band_buf[i * meg_size], meg_size, MPI_BYTE, 1, 99, MPI_COMM_WORLD);
		}
		if (mype == 1){
			MPI_Recv(&band_buf[i * meg_size], meg_size, MPI_BYTE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	double mb_end = MPI_Wtime();
	total_byes_sent = meg_size * 100;
	total_gigs_sent = total_byes_sent / gig_size;
	secs_elapsed = mb_end - mb_start;
	gbs = total_gigs_sent / secs_elapsed;
	if (mype == 0){
		printf("\nGB/s for MB packets: %f\n", gbs);
	}

	double gb_start = MPI_Wtime();
	for (int i = 0; i < 60; i++){
		if (mype == 0){
			MPI_Send(&band_buf[i], gig_size, MPI_BYTE, 1, 99, MPI_COMM_WORLD);
		}
		if (mype == 1){
			MPI_Recv(&band_buf[i], gig_size, MPI_BYTE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	double gb_end = MPI_Wtime();
	total_byes_sent = gig_size * 60.0;
	total_gigs_sent = total_byes_sent / gig_size;
	secs_elapsed = gb_end - gb_start;
	gbs = total_gigs_sent / secs_elapsed;
	if (mype == 0){
		printf("\nGigs send: %f\n", total_gigs_sent);
		printf("Secs elapsed: %f\n", secs_elapsed);
		printf("\nGB/s for GB packets: %f\n\n", gbs);
	}

	free(band_buf);

    MPI_Finalize();
    return EXIT_SUCCESS;
}