#include <stdio.h>
#include <stdlib.h>

#define CHECK_ERR(x)                                    \
if (x != cudaSuccess) {                               \
    fprintf(stderr,"%s in %s at line %d\n",             \
    cudaGetErrorString(x),__FILE__,__LINE__);	\
    exit(-1);						\
}

int main() {
  int nDevices;

  CHECK_ERR(cudaGetDeviceCount(&nDevices));
  int i;
  for (i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    CHECK_ERR(cudaGetDeviceProperties(&prop, i));
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
}