// nvcc -rdc=true QuickSort.cu
#include <cstdio>
#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include <algorithm>

#define MAX_DEPTH 16
#define INSERTION_SORT 32

// selection sort for data in [left, right]
__device__ void selection_sort(float* data, int left, int right) {
    for (int i = left; i <= right; i++) {
        float min_value = data[i];
        int min_id = i;
        // 能利用缓存
        for (int j = i + 1; j <= right; j++) {
            float t_data = data[j];
            if (t_data < min_value) {
                min_value = t_data;
                min_id = j;
            }
        }
        if (i != min_id) {
            data[min_id] = data[i];
            data[i] = min_value;
        }
    }
}

__global__ void cdp_simple_quicksort(float* data, int left, int right, int depth) {
    // 边界情况，数据较小或者递归层数太深时直接排序
    if (depth >= MAX_DEPTH || right - left <= INSERTION_SORT) {
        selection_sort(data, left, right);
        return;
    }
    float *lptr = data + left;
    float *rptr = data + right;
    float pivot = data[(left + right) / 2];

    while (lptr <= rptr) {
        float lval = *lptr;
        float rval = *rptr;
        while (lval < pivot) {
            ++lptr;
            lval = *lptr;
        }
        while (rval > pivot) {
            --rptr;
            rval = *rptr;
        }
        if (lptr <= rptr) {
            *lptr++ = rval;
            *rptr-- = lval;
        }
    }

    int nright = rptr - data;
    int nleft = lptr - data;

    if (left < nright) {
        // 创建非阻塞的流，互相之间不会影响
        cudaStream_t s0;
        cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking);
        cdp_simple_quicksort<<<1, 1, 0, s0>>>(data, left, nright, depth + 1);
        cudaStreamDestroy(s0);
    }

    if (nleft < right) {
        cudaStream_t s1;
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        cdp_simple_quicksort<<<1, 1, 0, s1>>>(data, nleft, right, depth + 1);
        cudaStreamDestroy(s1);
    }
}

void quick_sort_gpu(float* a, const int N) {
    float *a_d;
    cudaMalloc((void**) &a_d, sizeof(float) * N);
    cudaMemcpy(a_d, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cdp_simple_quicksort<<<1, 1>>>(a_d, 0, N - 1, 0);
    cudaMemcpy(a, a_d, sizeof(float) * N, cudaMemcpyDeviceToHost);
}

int main() {
    const int N = 1000000;
    float* a = (float*) malloc(sizeof(float) * N);
    float* b = (float*) malloc(sizeof(float) * N);
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        a[i] = rand() / (float) RAND_MAX;
        b[i] = a[i];
    }
    quick_sort_gpu(a, N);
    std::sort(b, b + N);
    for (int i = 0; i < N; i++) {
        if (a[i] != b[i]) {
            printf("对比有差异!\n");
            exit(1);
        }
    }
    return 0;
}