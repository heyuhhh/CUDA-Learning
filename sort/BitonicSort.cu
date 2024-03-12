#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>
#include <random>

void bitonic_sort(float* a, const int N, bool desc) {
    // 长为 stride 的序列已经是双调序列了，现在对其排序
    for (int stride = 2; stride <= N; stride <<= 1) {
        int half_stride = stride >> 1;
        // 这里 s 的含义是每次要对长度为 s 的序列从中间断开分为两个双调序列
        for (int s = stride; s >= 2; s >>= 1) {
            int hs = s >> 1;
            // 每次只用处理一半的数据
            for (int i = 0; i < N / 2; i++) {
                // 每 half_stride 个元素，他们的排序规则都保持一致
                // 注意这里，本来是每个 idx / stride 的，但是因为现在 i 表示前一半，所以 stride 也要 / 2
                // (其实找规律发现一下也行，每次都枚举的是前一半)
                bool inner_desc = ((i / half_stride) & 1);
                // j k 为一个 s 范围内相应的两个下标，需要对其进行排序
                int j = (i / hs) * s + i % hs;
                int k = j + hs;
                if ((desc && ((!inner_desc && a[j] < a[k]) || (inner_desc && a[j] > a[k])))
                || (!desc && ((!inner_desc && a[j] > a[k]) || (inner_desc && a[j] < a[k])))
                    ) {
                    std::swap(a[j], a[k]);
                }
            }
        }
    }
}

__global__
void bitonic_sort_kernel(float*  a, const int N, bool desc, int stride, int s) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int half_stride = stride >> 1;
    int hs = s >> 1;
    // 将 cpu 算法中的内层枚举 i 并行化
    for (int i = idx; i < N / 2; i += blockDim.x * gridDim.x) {
        bool inner_desc = (((i / half_stride) & 1) ^ desc);
        int j = (i / hs) * s + i % hs;
        int k = j + hs;
        if ((!inner_desc && a[j] > a[k]) || (inner_desc && a[j] < a[k])) {
            // swap a[j], a[k]
            float tmp = a[k];
            a[k] = a[j];
            a[j] = tmp;
        }
    }
}

void bitonic_sort_gpu(float* a, const int N, bool desc=0) {
    int t = 1;
    while (t < N) {
        t <<= 1;
    }
    float *a_d;
    cudaMalloc((void**) &a_d, sizeof(float) * t);
    cudaMemcpy(a_d, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    if (desc) {
        cudaMemset(a_d + N, 0, sizeof(float) * (t - N));
    } else {
        cudaMemset(a_d + N, 0x7f, sizeof(float) * (t - N));
    }
    dim3 blockDim(128);
    dim3 gridDim(1024);
    // 枚举放在外面，一方面减少warp scheduler压力，另外一方面实现线程块间的同步
    // kernel 涉及全局内存的 swap 操作，如果没有线程块的同步可能会出问题
    for (int stride = 2; stride <= t; stride <<= 1) {
        for (int s = stride; s >= 2; s >>= 1) {
            bitonic_sort_kernel<<<gridDim, blockDim>>>(a_d, t, desc, stride, s);
        }
    }
    cudaMemcpy(a, a_d, sizeof(float) * N, cudaMemcpyDeviceToHost);
}

void bitonic_sort_cpu(float* a, const int N, bool desc=0) {
    int t = 1;
    while (t < N) {
        t <<= 1;
    }
    float *tmp = (float*) malloc(sizeof(float) * t);
    memcpy(tmp, a, sizeof(float) * N);
    for (int i = N; i < t; i++) {
        if (desc) {
            tmp[i] = 0;
        } else {
            tmp[i] = MAXFLOAT;
        }
    }
    bitonic_sort(tmp, t, desc);
    memcpy(a, tmp, sizeof(float) * N);
}

int main() {
    const int N = 10000000;
    
    float* a = (float*) malloc(sizeof(float) * N);
    float* b = (float*) malloc(sizeof(float) * N);
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        a[i] = rand() / (float) RAND_MAX;
        b[i] = a[i];
    }
    bitonic_sort_gpu(a, N, 0);
    bitonic_sort_cpu(b, N, 0);
    for (int i = 0; i < N; i++) {
        if (a[i] != b[i]) {
            printf("比对有差异\n");
            exit(1);
        }
    }
    // bitonic_sort_gpu(a, N, 1);
    // for (int i = N - 20; i < N; i++) {
    //     std::cout << a[i] << ' ';
    // } std::cout << std::endl;
    // bitonic_sort_cpu(a, N, 0);
    // for (int i = N - 20; i < N; i++) {
    //     std::cout << a[i] << ' ';
    // } std::cout << std::endl;
}