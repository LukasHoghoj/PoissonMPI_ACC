#include "ST.h"

// source term
void ST(double **f, int rank, int size, int nx, int ny, int nz, int nzstart, double dx, double dy, double dz){
    double pi = 2.0 * asin(1.0);
    double x, y, z;
    double zstart = -1.0 + ((double) nzstart) * dz;
    for(int i = 0; i < nz; i++){
        z = ((double) i) * dz + zstart;
        for(int j = 0; j < ny; j++){
            y = ((double) j) * dy;
            for(int k = 0; k < nx; k++){
                x = ((double) k) * dx;
                (*f)[IDX_3D(i,j,k,nz,ny,nx)] = 3*pi*pi*sin(pi*x) * sin(pi*y) * sin(pi*z);
                
            }
        }
    }

} 
