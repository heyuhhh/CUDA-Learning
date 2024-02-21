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
__global__ void dot_vec4(float* a, float* b, float* y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 4;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float smem[NUM_WARPS];

    float sum = 0.0f;
    for (int i = idx; i < N; i += 4 * gridDim.x * NUM_THREADS) {
        float4 reg_a, reg_b;
        if (i < N) {
            reg_a = reinterpret_cast<float4*>(a + i)[0];
            reg_b = reinterpret_cast<float4*>(b + i)[0];
        } else {
            reg_a = make_float4(0, 0, 0, 0);
            reg_b = make_float4(0, 0, 0, 0);
        }
        if (i < N) {
            sum += reg_a.x * reg_b.x;
        }
        if (i + 1 < N) {
            sum += reg_a.y * reg_b.y;
        }
        if (i + 2 < N) {
            sum += reg_a.z * reg_b.z;
        }
        if (i + 3 < N) {
            sum += reg_a.w * reg_b.w;
        }
    }
    
    sum = warp_reduce_sum(sum);
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    if (lane == 0) {
        smem[warp] = sum;
    }
    __syncthreads();

    if (warp == 0) { // 小优化：后续只需对 warp0 里面的线程做操作即可
        sum = (threadIdx.x < NUM_WARPS) ? smem[threadIdx.x] : 0.0f;
        sum = warp_reduce_sum<NUM_WARPS>(sum);
        if (tid == 0) {
            atomicAdd(y, sum);
        }
    }
}

int main() {
    const int N = 10000000;
    float *x, *y, *z;
    cudaMallocManaged((void**) &x, sizeof(float) * N);
    cudaMallocManaged((void**) &y, sizeof(float) * N);
    cudaMallocManaged((void**) &z, sizeof(float));
    srand(time(NULL));
    float z_cpu = 0.0f;
    for (int i = 0; i < N; i++) {
        x[i] = rand() / (double) RAND_MAX;
        y[i] = rand() / (double) RAND_MAX;
        z_cpu += x[i] * y[i];
    }
    dim3 blockDim(128 / 4);
    dim3 gridDim(1024);
    dot_vec4<<<gridDim, blockDim>>>(x, y, z, N);
    cudaDeviceSynchronize();
    printf("ans = %f, gpu res = %f\n", z_cpu, z[0]);
}