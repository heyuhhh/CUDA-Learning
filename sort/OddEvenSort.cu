#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>
#include <random>


__global__ void OddEvenSortKernel(float* a, const int N) {
    __shared__ bool sorted;
    if (threadIdx.x == 0) {
        sorted = false;
    }
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // printf("%d\n", sorted);
    while (!sorted) {
        sorted = true;
        // step 1: odd pos
        for (int i = 2 * idx; i + 1 < N; i += gridDim.x * blockDim.x * 2) {
            // printf("%d %d %d\n", idx, i, i + 1);
            if (a[i] > a[i + 1]) {
                float tmp = a[i];
                a[i] = a[i + 1];
                a[i + 1] = tmp;
                sorted = false;
            }
        }
        // __threadfence();
        // step 2: even pos
        for (int i = 2 * idx + 1; i + 1 < N; i += gridDim.x * blockDim.x * 2) {
            if (a[i] > a[i + 1]) {
                float tmp = a[i];
                a[i] = a[i + 1];
                a[i + 1] = tmp;
                sorted = false;
            }
        }
        __syncthreads();
    }
}

void OddEvenSort_gpu(float* a, const int N) {
    float *a_d;
    cudaMalloc((void**) &a_d, sizeof(float) * N);
    cudaMemcpy(a_d, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    dim3 blockDim(1024);
    dim3 gridDim(1);
    OddEvenSortKernel<<<gridDim, blockDim>>>(a_d, N);
    cudaMemcpy(a, a_d, sizeof(float) * N, cudaMemcpyDeviceToHost);
}

int main() {
    const int N = 100000;
    float* a = (float*) malloc(sizeof(float) * N);
    float* b = (float*) malloc(sizeof(float) * N);
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        a[i] = rand() / (float) RAND_MAX;
        b[i] = a[i];
    }
    std::sort(b, b + N);
    OddEvenSort_gpu(a, N);
    for (int i = 0; i < N; i++) {
        if (a[i] != b[i]) {
            printf("比对有差异!\n");
            exit(1);
        }
    }
    // for (int i = 0; i < 256; i++) {
    //     std::cout << a[i] << ' ';
    // } std::cout << std::endl;
    // for (int i = N - 20; i < N; i++) {
    //     std::cout << a[i] << ' ';
    // } std::cout << std::endl;
}