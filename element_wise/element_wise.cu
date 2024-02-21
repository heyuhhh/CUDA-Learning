#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <time.h>

#define WARP_SIZE 32

// warp reduce sum
template<const int NUM_WARPS = WARP_SIZE>
__device__ float warp_reduce_sum(float val) {
    for (int i = NUM_WARPS >> 1; i > 0; i >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    return val;
}

template<const int NUM_THREADS=128/4>
__global__ void element_wise_vec4(float* a, float* b, float* y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 4;

    for (int i = idx; i < N; i += 4 * gridDim.x * NUM_THREADS) {
        float4 reg_a, reg_b;
        if (i < N) {
            reg_a = reinterpret_cast<float4*>(a + i)[0];
            reg_b = reinterpret_cast<float4*>(b + i)[0];
        } else {
            reg_a = make_float4(0, 0, 0, 0);
            reg_b = make_float4(0, 0, 0, 0);
        }
        float4 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;
        reinterpret_cast<float4*>(y + i)[0] = reg_c;
    }
}

int main() {
    const int N = 100000000;
    float *x, *y, *z;
    cudaMallocManaged((void**) &x, sizeof(float) * N);
    cudaMallocManaged((void**) &y, sizeof(float) * N);
    cudaMallocManaged((void**) &z, sizeof(float) * N);
    srand(time(NULL));
    float *z_cpu = (float*) malloc(sizeof(float) * N);
    for (int i = 0; i < N; i++) {
        x[i] = rand() / (double) RAND_MAX;
        y[i] = rand() / (double) RAND_MAX;
        z_cpu[i] = x[i] + y[i];
    }
    dim3 blockDim(128 / 4);
    dim3 gridDim(1024);
    element_wise_vec4<<<gridDim, blockDim>>>(x, y, z, N);
    cudaDeviceSynchronize();
    float max_error = 0.0;
    for (int i = 0; i < N; i++) {
        max_error = fmax(max_error, fabs(z[i] - z_cpu[i]));
    }
    printf("max error is: %f\n", max_error);
}