#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>

#define WARP_SIZE 32

// Warp Reduce Sum
template<const int kWarpSize = WARP_SIZE>
__device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// Softmax: x: N, y: N
// y[i] = exp(x_i) / {exp(x_1) + exp(x_2) + ... + exp(x_n)}
template<const int NUM_THREADS=128>
__global__ void softmax(float* x, float* y, float* total, int* count, int N) {
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float smem[NUM_WARPS];
    float val = (idx < N) ? expf(x[idx]) : 0.0f;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    float sum = warp_reduce_sum<WARP_SIZE>(val);

    if (lane == 0) {
        smem[warp] = sum;
    }
    __syncthreads();

    if (warp == 0) {
        sum = (lane < NUM_WARPS) ? smem[lane] : 0;
        sum = warp_reduce_sum<NUM_WARPS>(sum);
        if (lane == 0) {
            atomicAdd(total, sum);
            // 网格级同步，确保写入count在total+=sum之后
            __threadfence();
            if (tid == 0) {
                atomicAdd(count, 1);
            }
        }
    }
    
    // 检测直到所有块都完成写入
    while (atomicAdd(count, 0) != gridDim.x);

    if (idx < N) {
        y[idx] = val / (*total);
    }
}



int main() {
    const int N = 100000;
    float *x, *y, *z;
    int *count;
    cudaMallocManaged((void**) &x, sizeof(float) * N);
    cudaMallocManaged((void**) &y, sizeof(float) * N);
    cudaMallocManaged((void**) &z, sizeof(float));
    cudaMallocManaged((void**) &count, sizeof(int));
    srand(time(NULL));

    // cpu softmax
    float z_sum = 0.0f;
    float f_max = -MAXFLOAT;
    for (int i = 0; i < N; i++) {
        x[i] = rand() / (double) RAND_MAX;
        f_max = fmax(f_max, x[i]);
    }
    // 为了方便起见，减去最大值就不单独写一个kernel了
    for (int i = 0; i < N; i++) {
        // x[i] -= f_max;
        z_sum += expf(x[i]);
    }
    float *y_cpu = (float*) malloc(sizeof(float) * N);
    for (int i = 0; i < N; i++) {
        y_cpu[i] = expf(x[i]) / z_sum;
    }

    dim3 blockDim(128);
    dim3 gridDim((N + 128 - 1) / 128);
    softmax<<<gridDim, blockDim>>>(x, y, z, count, N);
    cudaDeviceSynchronize();
    
    float max_error = 0.0f;
    for (int i = 0; i < N; i++) {
        max_error = fmax(max_error, fabs(y[i] - y_cpu[i]));
        // printf("%d: %f %f\n", i, y_cpu[i], y[i]);
    }
    printf("max error is: %f\n", max_error);
}