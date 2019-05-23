#include "header.h"

#include "ST.h"
#include "setBC_IC.h"
#include "poisson3D_serial.h"
#include "poisson3D_MPI1.h"
#include "poisson3D_MPI2.h"
#include "poisson3D_ACC1.h"
#include "poisson3D_ACC2.h"
#include "poisson3D_hyb1.h"
#include "poisson3D_hyb2.h"
#include "poisson3D_hyb3.h"
#include "poisson3D_hyb4.h"
#include "poisson3D_hyb5.h"
#include "mat2file.h"
#include "printres.h"
#include "initACC.h"

int main(int argc, char** argv){
    int rank, size;

    MPI_Init(&argc, &argv);

    // Check input arguments
    if(rank==0 && argc < 6){
        printf("Too few arguments, required input:\n");
        printf("1: Experiment type\n");
        printf("    0:  Serial version\n");
        printf("    1:  OpenACC version 1\n");
        printf("    2:  OpenACC version 2\n");
        printf("    10: MPI version 1\n");
        printf("    11: MPI version 2\n");
        printf("    20: MPI + OpenACC version 1\n");
        printf("    21: MPI + OpenACC version 2\n");
        printf("    22: MPI + OpenACC version 3\n");
        printf("    23: MPI + OpenACC version 4\n");
        printf("    24: MPI + OpenACC version 5 --- NOT WORKING CORRECTLY\n");
        printf("2: Nx\n");
        printf("3: Ny\n");
        printf("4: Nz\n");
        printf("5: Number of iterations\n");
        printf("\nOptional input:\n");
        printf("-o ""solution output file""\n");
        printf("-p ""timing output file (append)""\n");
        return 1;
    }
    // Divide number of elements on different ranks
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int etype = atoi(argv[1]);

    if(etype == 2 | etype >= 20){
        initACC(rank);
    }

    // Global number of elements
    int nxg, nyg, nzg;
    nxg = atoi(argv[2]);
    nyg = atoi(argv[3]);
    nzg = atoi(argv[4]);
    // Number of iterations
    int Niter = atoi(argv[5]);
    // Local number of elements, separating alog z elements
    int nx, ny, nz;
    nx = nxg;
    ny = nyg;
    nz = nzg/size;
    int nzstart = nz * rank;
    double dx, dy, dz;
    dx = 2.0 / (((double) nxg) - 1);
    dy = 2.0 / (((double) nyg) - 1);
    dz = 2.0 / (((double) nzg) - 1);
    if(nzg%size>rank){
        nz = nz + 1;
        nzstart += rank;
    }else{
        nzstart += nzg%size;
    }
    if(rank > 0) nzstart -= 1;
    // Adjust nz to have BC buffer
    if(size != 1){
        if(rank == 0 || rank == size - 1){
            nz += 1;
        }else{
            nz += 2;
        }
    }
   
    // Allocate local data
    double* A = malloc(nx * ny * nz * sizeof(double));
    double* Anew = malloc(nx * ny * nz * sizeof(double));
    double* S = malloc(nx * ny * nz * sizeof(double));

    // Set Boundary condition
    setBC_IC(&A,&Anew,rank,size,nx,ny,nz);

    // Initialize source term
    ST(&S, rank, size, nx, ny, nz, nzstart, dx, dy, dz);

    // Print message
    for(int i = 0; i < size; i++){
        MPI_Barrier(MPI_COMM_WORLD);
        if(i == rank){
            printf("[%d] Matrices of dimensions %d x %d x %d initiated \n",rank, nx, ny, nz);
        }
    }
   

    // Select correct jacobi function
    MPI_Barrier(MPI_COMM_WORLD);
    double wt = MPI_Wtime();
    if(etype == 0){
        poisson3D_serial(A, Anew, S, nx, ny, nz, dx, dy, dz, Niter);
    }else{
    if(etype == 1){
        poisson3D_ACC1(A, Anew, S, nx, ny, nz, dx, dy, dz, Niter);
    }else{
    if(etype == 2){
        poisson3D_ACC2(A, Anew, S, nx, ny, nz, dx, dy, dz, Niter);
    }else{
    if(etype == 10){
        poisson3D_MPI1(A, Anew, S, nx, ny, nz, dx, dy, dz, Niter, rank, size);
    }else{
    if(etype == 11 || etype == 1){
        poisson3D_MPI2(A, Anew, S, nx, ny, nz, dx, dy, dz, Niter, rank, size);
    }else{
    if(etype == 20){
        poisson3D_hyb1(A, Anew, S, nx, ny, nz, dx, dy, dz, Niter, rank, size);
    }else{
    if(etype == 21){
        poisson3D_hyb2(A, Anew, S, nx, ny, nz, dx, dy, dz, Niter, rank, size);
    }else{
    if(etype == 22){
        poisson3D_hyb3(A, Anew, S, nx, ny, nz, dx, dy, dz, Niter, rank, size);
    }else{
    if(etype == 23 || etype == 2){
        poisson3D_hyb4(A, Anew, S, nx, ny, nz, dx, dy, dz, Niter, rank, size);
    }else{
    if(etype == 24){
        poisson3D_hyb5(A, Anew, S, nx, ny, nz, dx, dy, dz, Niter, rank, size);
    }else{
        if(rank == 0){
            printf("Invalid experiment type\n");
            return 1;
        }
    }}}}}}}}}}

    wt = MPI_Wtime() - wt;
    

    // Print walltime
    for(int i = 0; i < size; i++){
        MPI_Barrier(MPI_COMM_WORLD);
        if(i == rank){
            printf("[%d] Walltime = %lf\n",rank, wt);
        }
     }    


    // Print results to file
    for(int i = 0; i < argc - 1; i++){
        if(strcmp(argv[i],"-o") == 0){
            // Complete solution
            for(int j = 0; j < size; j++){
                MPI_Barrier(MPI_COMM_WORLD);
                if(j == rank){
                    mat2file(A, argv[i+1], nx, ny, nz, rank, size);
                }
            }
            
        }

        // Timing
        if(strcmp(argv[i],"-p") == 0){
            double tmean, tmin, tmax;
            MPI_Reduce(&wt,&tmean,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
            MPI_Reduce(&wt,&tmin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
            MPI_Reduce(&wt,&tmax,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
            tmean = tmean / ((double) size);
            if(rank == 0){
                printres(argv[i+1],etype,tmean,tmin,tmax,nx,ny,nzg,Niter,size);
            }
        }
    }

    free(S);
    free(A);
    free(Anew);
    MPI_Finalize();
    return 0;
}
