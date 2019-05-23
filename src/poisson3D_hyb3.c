#include "poisson3D_hyb3.h"

void poisson3D_hyb3(double *A, double *Anew, double *S, int nx, int ny, int nz, double dx, double dy, double dz, int Niter, int rank, int size){
    double dxs, dys, dzs;
    dxs = dx*dx; dys = dy*dy; dzs = dz*dz;
    double dsum = dxs + dys + dzs;
    double *tmp;
    double Fnorm, Fnormold=0, Fdiff;
    #pragma acc data copy(A[0:nx*ny*nz],Anew[0:nx*ny*nz], S[0:nx*ny*nz])
    {
    for(int N = 0; N < Niter; N++){
        // Itteration step computation change to acc kernel and kernel async to have make comm while compting inner domain
        Fnorm = 0;
                
        #pragma acc parallel loop present(A[0:nx*ny*nz], S[0:nx*ny*nz],Anew[0:nx*ny*nz]) reduction(+:Fnorm) tile(32,4,4)
        for(int i = 1; i < nz - 1; i++){
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
        // Swap pointers
        tmp = A;
        A = Anew;
        Anew = tmp;
	// Change to only the required data
      
        if(rank > 0){
            // Halo comm at lower boundary
            #pragma acc update host(A[IDX_3D(1,0,0,nz,ny,nx):nx*ny]) 
            MPI_Sendrecv(A+IDX_3D(1,0,0,nz,ny,nx),nx*ny,MPI_DOUBLE,rank-1,1,A,nx*ny,MPI_DOUBLE,rank-1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            #pragma acc update device(A[0:nx*ny])
        }
        if(rank < size - 1){
            // Halo comm at upper boundary
            #pragma acc update host(A[IDX_3D(nz-2,0,0,nz,ny,nx):nx*ny])
            MPI_Sendrecv(A+IDX_3D((nz-2),0,0,nz,ny,nx),nx*ny,MPI_DOUBLE,rank+1,0,A+IDX_3D((nz-1),0,0,nz,ny,nx),nx*ny,MPI_DOUBLE,rank+1,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            #pragma acc update device(A[IDX_3D(nz-1,0,0,nz,ny,nx):nx*ny])
        } 
    }
}
}
