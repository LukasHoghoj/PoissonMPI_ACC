#include "initACC.h"

void initACC(int rank){
    acc_init(acc_device_nvidia);                                 // OpenACC call
    const int num_dev = acc_get_num_devices(acc_device_nvidia);  // #GPUs
    const int dev_id = rank % num_dev;         
    acc_set_device_num(dev_id,acc_device_nvidia); // assign GPU to one MPI process

    printf("MPI process %d  is assigned to GPU %d\n",rank, dev_id);
}
