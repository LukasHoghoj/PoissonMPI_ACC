#include "mat2file.h"

void mat2file(double *A, char filename[], int nx, int ny, int nz, int rank, int size){
    int nstar = 1, nend = nz - 1;
    FILE * f;
    // Find indices that should be printed
    if(rank == 0){
        nstar = 0;
    }
    if(rank == size-1){
        nend = nz;
    }
    // Open file
    if(rank == 0){
        f = fopen(filename,"w");
    }else{
        f = fopen(filename,"a");
    }
    for(int i = nstar; i < nend; i++){
        for(int j = 0; j < ny; j++){
            for(int k = 0; k < nx; k++){
                fprintf(f,"%lf ",A[IDX_3D(i,j,k,nz,ny,nx)]);
                //printf("A(%d,%d,%d) = %lf\n",k,j,i,A[IDX_3D(i,j,k,nz,ny,nx)]);
            }
            fprintf(f,"\n");
        }
        fprintf(f,"\n");
    }
    // Close file
    fclose(f);
    printf("[%d] Results successfully written to file %s\n",rank, filename);
}
