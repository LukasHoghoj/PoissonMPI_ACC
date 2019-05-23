#include "printres.h"

void printres(char filename[], int study, double tmean, double tmin, double tmax, int nx, int ny, int nz, int iter, int size){
    FILE *f;
    f = fopen(filename,"r");
    if(f==NULL){
        f = fopen(filename,"w");
        fprintf(f,"Problem ID, Size,      Nx,      Ny,      Nz,  Niter,     tavg,     tmin,     tmax\n");
    }else{
        fclose(f);
        f = fopen(filename,"a");
    }
    fprintf(f,"%2d,           %2d, %7d, %7d, %7d, %6d, %0.6lf, %0.6lf, %0.6lf\n",study,size, nx,ny,nz,iter,tmean,tmin,tmax);
    fclose(f);
}
