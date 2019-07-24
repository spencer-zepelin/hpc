#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char ** args){
    
    printf("number of args: %d\n", argc);
    
    for (int i = 0; i < argc; i++){
        printf("Arg %d:   %s\n", i, args[i]);
    }
    
    if (!0){
        printf("evaluates\n");
    }
    
    printf("sizeof double: %lu bytes\n", sizeof(double));
    
    char buf[100];
    int j;
    
    for (int i = 0; i<5; i++){
        j = snprintf(buf, 3, "%d ", i);
    }
    
    printf("\nstring:\n%s\ncharacter count = %d\n", buf, j);
    
    MPI_File f0;
    if (mype == 0){
        MPI_File_open(comm2d, "f0.txt", MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &f0);
    }
    
    
    
    return EXIT_SUCCESS;
}
