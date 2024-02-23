#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>
#include <random>

__device__ __host__
void merge_seq(float* A, int A_len, float* B, int B_len, float* C) {
    int A_cnt = 0, B_cnt = 0;
    while (A_cnt < A_len && B_cnt < B_len) {
        if (A[A_cnt] < B[B_cnt]) {
            C[A_cnt + B_cnt] = A[A_cnt];
            ++A_cnt;
        } else {
            C[A_cnt + B_cnt] = B[B_cnt];
            ++B_cnt;
        }
    }
    while (A_cnt < A_len) {
        C[A_cnt + B_cnt] = A[A_cnt];
        ++A_cnt;
    }
    while (B_cnt < B_len) {
        C[A_cnt + B_cnt] = B[B_cnt];
        ++B_cnt;
    }
}

__global__
void merge_sort(float* a, float* c, const int N, int stride) {
    int st = (blockDim.x * blockIdx.x + threadIdx.x) * (2 * stride);
    int md = st + stride;
    if (md < N) {
        merge_seq(a + st, stride, a + md, min(stride, N - md), c + st);
        for (int i = st; i < min(N, st + 2 * stride); i++) {
            a[i] = c[i];
        }
    }
}

void mergeSort(float* a, const int n) {
    float *a_d, *tmp;
    cudaMalloc((void**) &a_d, sizeof(float) * n);
    cudaMalloc((void**) &tmp, sizeof(float) * n);
    cudaMemcpy(a_d, a, sizeof(float) * n, cudaMemcpyHostToDevice);
    for (int stride = 1; stride < n; stride <<= 1) {
        int seg_num = (n + 2 * stride - 1) / (2 * stride);
        dim3 blockDim(min(seg_num, 128));
        dim3 gridDim((seg_num + blockDim.x - 1) / blockDim.x);
        merge_sort<<<gridDim, blockDim>>>(a_d, tmp, n, stride);
    }
    cudaMemcpy(a, a_d, sizeof(float) * n, cudaMemcpyDeviceToHost);
    cudaFree(a_d);
    cudaFree(tmp);
}

int main() {
    const int N = 10000000;
    float *a = (float*) malloc(sizeof(float) * N);
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        a[i] = rand() / (float) RAND_MAX;
    }
    mergeSort(a, N);
    for (int i = 0; i < 20; i++) {
        std::cout << a[i] << ' ';
    } std::cout << std::endl;
    for (int i = N - 20; i < N; i++) {
        std::cout << a[i] << ' ';
    } std::cout << std::endl;
}