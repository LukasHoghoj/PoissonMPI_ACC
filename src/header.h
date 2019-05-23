#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include <openacc.h>

#define IDX_3D(i, j, k, I, J, K) ((i)*(J)*(K) +  (j)*(K) + (k))

