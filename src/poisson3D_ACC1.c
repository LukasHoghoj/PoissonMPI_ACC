#include "poisson3D_ACC1.h"

void poisson3D_ACC1(double *A, double *Anew, double *S, int nx, int ny, int nz, double dx, double dy, double dz, int Niter){
    double dxs, dys, dzs;
    dxs = dx*dx; dys = dy*dy; dzs = dz*dz;
    double dsum = dxs + dys + dzs;
    double *tmp;
    double Fnorm, Fnormold=0, Fdiff;

    for(int N = 0; N < Niter; N++){
        Fnorm = 0;
        #pragma acc parallel loop copyin(A[0:nx*ny*nz], S[0:nx*ny*nz]) copyout(Anew[0:nx*ny*nz]) reduction(+:Fnorm) tile(32,4,4)
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
    }
}
