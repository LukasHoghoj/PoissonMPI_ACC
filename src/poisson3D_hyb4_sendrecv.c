#include "poisson3D_hyb4.h"

void poisson3D_hyb4(double *A, double *Anew, double *S, int nx, int ny, int nz, double dx, double dy, double dz, int Niter, int rank, int size){
    double dxs, dys, dzs;
    dxs = dx*dx; dys = dy*dy; dzs = dz*dz;
    double dsum = dxs + dys + dzs;
    double *tmp;
    double Fnorm, Fnormold=0, Fdiff;
    double third, sixth;
    third = ((double) 1)/((double) 3); sixth = third / ((double) 2);

    #pragma acc data copy(A[0:nx*ny*nz],Anew[0:nx*ny*nz], S[0:nx*ny*nz],dsum[0:1],third, sixth)
    {
    for(int N = 0; N < Niter; N++){
        // Itteration step computation change to acc kernel and kernel async to have make comm while compting inner domain
        Fnorm = 0;
        for(int i = 1; i < nz - 1; i = i + nz - 3){
            #pragma acc parallel loop present(A[0:nx*ny*nz], S[0:nx*ny*nz],Anew[0:nx*ny*nz],dsum,third, sixth) reduction(+:Fnorm) tile(32,32) async(1)
            for(int j = 1; j < ny - 1; j++){
                for(int k = 1; k < nx - 1; k++){
                    Anew[IDX_3D(i,j,k,nz,ny,nx)] = sixth*(
                                               A[IDX_3D(i+1,j,k,nz,ny,nx)] +
                                               A[IDX_3D(i-1,j,k,nz,ny,nx)] +
                                               A[IDX_3D(i,j+1,k,nz,ny,nx)] +
                                               A[IDX_3D(i,j-1,k,nz,ny,nx)] +
                                               A[IDX_3D(i,j,k+1,nz,ny,nx)] +
                                               A[IDX_3D(i,j,k-1,nz,ny,nx)])
                                               + third * dsum *
                                                    S[IDX_3D(i,j,k,nz,ny,nx)];
                    Fnorm += Anew[IDX_3D(i,j,k,nz,ny,nx)];
                }
            }
        }

       
        #pragma acc parallel loop present(A[0:nx*ny*nz], S[0:nx*ny*nz],Anew[0:nx*ny*nz],dsum,third, sixth) reduction(+:Fnorm) tile(32,4,4) async(2)
        for(int i = 2; i < nz - 2; i++){
            for(int j = 1; j < ny - 1; j++){
                for(int k = 1; k < nx - 1; k++){
                    Anew[IDX_3D(i,j,k,nz,ny,nx)] = sixth*(
                                               A[IDX_3D(i+1,j,k,nz,ny,nx)] +
                                               A[IDX_3D(i-1,j,k,nz,ny,nx)] +
                                               A[IDX_3D(i,j+1,k,nz,ny,nx)] +
                                               A[IDX_3D(i,j-1,k,nz,ny,nx)] +
                                               A[IDX_3D(i,j,k+1,nz,ny,nx)] +
                                               A[IDX_3D(i,j,k-1,nz,ny,nx)])
                                               + third * dsum *
                                                    S[IDX_3D(i,j,k,nz,ny,nx)];
                    Fnorm += Anew[IDX_3D(i,j,k,nz,ny,nx)];
                }
            }
        }
        Fnorm = sqrt(Fnorm);
        Fdiff = Fnorm - Fnormold;
        Fnormold = Fnorm;




        #pragma acc wait(1)

        if(rank > 0){
            #pragma acc update host(Anew[IDX_3D(1,0,0,nz,ny,nx):nx*ny])
           MPI_Sendrecv(Anew+IDX_3D(1,0,0,nz,ny,nx),nx*ny,MPI_DOUBLE,rank-1,1,Anew,nx*ny,MPI_DOUBLE,rank-1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            #pragma acc update device(Anew[0:nx*ny])
        }
        if(rank < size - 1){
            #pragma acc update host(Anew[IDX_3D(nz-2,0,0,nz,ny,nx):nx*ny])

            MPI_Sendrecv(Anew+IDX_3D((nz-2),0,0,nz,ny,nx),nx*ny,MPI_DOUBLE,rank+1,0,Anew+IDX_3D((nz-1),0,0,nz,ny,nx),nx*ny,MPI_DOUBLE,rank+1,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            #pragma acc update device(Anew[IDX_3D(nz-1,0,0,nz,ny,nx):nx*ny])
        }

        #pragma acc wait(2)

        // Swap pointers
        tmp = A;
        A = Anew;
        Anew = tmp;
    }
}
} // end data region
