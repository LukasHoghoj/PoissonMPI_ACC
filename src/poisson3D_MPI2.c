#include "poisson3D_MPI2.h"


void poisson3D_MPI2(double *A, double *Anew, double *S, int nx, int ny, int nz, double dx, double dy, double dz, int Niter, int rank, int size){
    double dxs, dys, dzs;
    dxs = dx*dx; dys = dy*dy; dzs = dz*dz;
    double dsum = dxs + dys + dzs;
    double *tmp;
    double Fnorm, Fnormold=0, Fdiff,Fdiffred;
    
    MPI_Request *req_low, *req_high;
    req_low = malloc(2*sizeof(MPI_Request));
    req_high = malloc(2*sizeof(MPI_Request));
    for(int N = 0; N < Niter; N++){
        // Itteration step computation
        Fnorm = 0;
        // Compute Boundaries
        int ii = 0;
        for(int i = 1; i < nz - 1; i += nz - 3){
        ii+=1;
            for(int j = 1; j < ny - 1; j++){
                for(int k = 1; k < nx - 1; k++){
                    Anew[IDX_3D(i,j,k,nz,ny,nx)] = (((double) 1)/((double) 6))*(
                                               A[IDX_3D(i+1,j,k,nz,ny,nx)] +
                                               A[IDX_3D(i-1,j,k,nz,ny,nx)] +
                                               A[IDX_3D(i,j+1,k,nz,ny,nx)] +
                                               A[IDX_3D(i,j-1,k,nz,ny,nx)] +
                                               A[IDX_3D(i,j,k+1,nz,ny,nx)] +
                                               A[IDX_3D(i,j,k-1,nz,ny,nx)])
                                               + (((double) 1)/((double) 3)) * dsum *
                                                    S[IDX_3D(i,j,k,nz,ny,nx)];
                    Fnorm += Anew[IDX_3D(i,j,k,nz,ny,nx)];
                }
            }
            if(ii==2) break;
        }
        
        // Communicate Ghost layers/BC
        if(rank > 0){
            MPI_Isend(Anew+IDX_3D(1,0,0,nz,ny,nx),nx*ny,MPI_DOUBLE,rank-1,1,MPI_COMM_WORLD, &req_low[0]);
            MPI_Irecv(Anew,nx*ny,MPI_DOUBLE,rank-1,0,MPI_COMM_WORLD, &req_low[1]);
         }
        if(rank < size - 1){
            MPI_Isend(Anew+IDX_3D((nz-2),0,0,nz,ny,nx),nx*ny,MPI_DOUBLE,rank+1,0,MPI_COMM_WORLD,&req_high[0]);
            MPI_Irecv(Anew+IDX_3D((nz-1),0,0,nz,ny,nx),nx*ny,MPI_DOUBLE,rank+1,1,MPI_COMM_WORLD,&req_high[1]);
        }

        // Make inner computations
        for(int i = 2; i < nz - 2; i++){
            for(int j = 1; j < ny - 1; j++){
                for(int k = 1; k < nx - 1; k++){
                    Anew[IDX_3D(i,j,k,nz,ny,nx)] = (((double) 1)/((double) 6))*(
                                               A[IDX_3D(i+1,j,k,nz,ny,nx)] +
                                               A[IDX_3D(i-1,j,k,nz,ny,nx)] +
                                               A[IDX_3D(i,j+1,k,nz,ny,nx)] +
                                               A[IDX_3D(i,j-1,k,nz,ny,nx)] +
                                               A[IDX_3D(i,j,k+1,nz,ny,nx)] +
                                               A[IDX_3D(i,j,k-1,nz,ny,nx)])
                                               + (((double) 1)/((double) 3)) * dsum *
                                                    S[IDX_3D(i,j,k,nz,ny,nx)];
                    Fnorm += Anew[IDX_3D(i,j,k,nz,ny,nx)];
                }
            }
        }
        Fnorm = sqrt(Fnorm);
        Fdiff = Fnorm - Fnormold;
        Fnormold = Fnorm;
	MPI_Reduce(&Fdiff,&Fdiffred,1,MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
        // Barriers for non - blocking communication
        if(rank > 0){
            MPI_Waitall(2,req_low,MPI_STATUSES_IGNORE);
        }
        if(rank < size - 1){
            MPI_Waitall(2,req_high,MPI_STATUSES_IGNORE);
        }

        // Swap pointers
        tmp = A;
        A = Anew;
        Anew = tmp;

    }
    free(req_low);
    free(req_high);
}
