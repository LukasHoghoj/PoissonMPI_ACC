#include "setBC_IC.h"

void setBC_IC(double **A, double **Anew, int rank,int size,int nx,int ny,int nz){
    // Set IC
    for(int i = 0; i < nx*ny*nz; i++){
        (*A)[i] = 0.0;
        (*Anew)[i] = 0.0;
    }
    // Zero boundary condition covered by setting IC
        
}
