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

// SEGMV_K32，假设 K 为 32 的倍数
// blockDim(32, 4)，每个block处理四行，每个warp处理一行
// gridDim(M / 4)，直接定义成一维，比较直观
__global__ void sgemv_k32(float* a, float* x, float* y, const int M, const int K) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    int m = blockIdx.x * blockDim.y + warp;
    if (m < M) {
        float sum = 0.0f;
        const int NUM_WARPS = (K + WARP_SIZE - 1) / WARP_SIZE;
        for (int i = 0; i < NUM_WARPS; i++) {
            int k = i * WARP_SIZE + threadIdx.x;
            if (k < K) {
                sum += a[m * K + k] * x[k];
            }
        }
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            y[m] = sum;
        }
    }
}

// K 为 128 倍数的情况
// 同上每个warp处理一行，进行一行的规约
// 只是现在通过vec4来读取
// blockDim(32, 4)，gridDim(M / 4)
__global__ void sgemv_k128(float* a, float* x, float* y, const int M, const int K) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    int m = blockIdx.x * blockDim.y + warp;
    if (m < M) {
        float sum = 0.0f;
        const int NUM_WARPS = (K + WARP_SIZE * 4 - 1) / (WARP_SIZE * 4);
        for (int i = 0; i < NUM_WARPS; i++) {
            int k = (i * WARP_SIZE + threadIdx.x) * 4;
            if (k < K) {
                float4 a_vec = reinterpret_cast<float4*>(a + m * K + k)[0];
                float4 x_vec = reinterpret_cast<float4*>(x + k)[0];
                sum += a_vec.x * x_vec.x;
                if (k + 1 < K) {
                    sum += a_vec.y * x_vec.y;
                }
                if (k + 2 < K) {
                    sum += a_vec.z * x_vec.z;
                }
                if (k + 3 < K) {
                    sum += a_vec.w * x_vec.w;
                }
            }
        }
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            y[m] = sum;
        }
    }
}   

// K = 16 < 32 的情况
// 一个 warp 处理两行
// blockDim(32, 4)
// gridDim(M / 8)
template<const int ROW_PER_WARP=2>
__global__ void sgemv_k16(float* a, float* x, float* y, const int M, const int K) {
    assert(WARP_SIZE % ROW_PER_WARP == 0);
    constexpr int K_WARP_SIZE = WARP_SIZE / ROW_PER_WARP;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    int m = (blockIdx.x * blockDim.y + warp) * ROW_PER_WARP + lane / K_WARP_SIZE;
    if (m < M) {
        int k = threadIdx.x % K_WARP_SIZE;
        float sum = a[m * K + k] * x[k];
        sum = warp_reduce_sum<K_WARP_SIZE>(sum);
        if (k == 0) {
            y[m] = sum;
        }
    }
}

float testPerformance(const int M, const int K, const int repeat, int algorithm);

int main() {
    // const int M_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    int M_list[15];
    for (int i = 0; i < 15; i++) {
        M_list[i] = (1 << (i + 10));
    }
    const int K_list[3] = {32, 128, 16};
    const int outer_repeat = 10, inner_repeat = 1;
    for (int i = 0; i < 15; i++) {
        for (int j = 0; j < 3; j++) {
            const int M = M_list[i];
            const int K = K_list[j];
            double total_sec = 0.0f;
            for (int k = 0; k < outer_repeat; k++) {
                double this_sec = testPerformance(M, K, inner_repeat, j);
                total_sec += this_sec;
            }
            printf("%lf\n", total_sec);
            double avg_sec = total_sec / outer_repeat;
            double avg_Gflops = (double) M * K * K / 1024 / 1024 / 1024 / avg_sec;
            printf("Alg %d: M = %6d, K = %6d, Time = %12.8lf s, AVG Performance = %10.4lf Gflops\n", j, M, K, avg_sec, avg_Gflops);
        }
    }
}

void cpuSgemv(float* a, float* x, float* y, const int M, const int K) {
    for (int i = 0; i < M; i++) {
        float res = 0;
        for (int j = 0; j < K; j++) {
            res += a[i * K + j] * x[j];
        }
        y[i] = res;
    }
}

float testPerformance(const int M, const int K, const int repeat, int algorithm) {
    float *a, *x, *y;
    cudaMallocManaged((void**) &a, sizeof(float) * M * K);
    cudaMallocManaged((void**) &x, sizeof(float) * K);
    cudaMallocManaged((void**) &y, sizeof(float) * M);
    srand(time(NULL));
    for (int i = 0; i < M * K; i++) {
        a[i] = rand() / (float) RAND_MAX;
    }
    for (int i = 0; i < K; i++) {
        x[i] = rand() / (float) RAND_MAX;
    }
    
    float *y_cpu = (float*) malloc(sizeof(float) * M);
    cpuSgemv(a, x, y_cpu, M, K);

    dim3 blockDim(32, 4);
    dim3 gridDim;
    if (K == 16) {
        gridDim = dim3((M + 7) / 8);
    } else {
        gridDim = dim3((M + 3) / 4);
    }

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // warm up
    if (algorithm == 0) {
        sgemv_k32<<<gridDim, blockDim>>>(a, x, y, M, K);
    } else if (algorithm == 1) {
        sgemv_k128<<<gridDim, blockDim>>>(a, x, y, M, K);
    } else {
        sgemv_k16<<<gridDim, blockDim>>>(a, x, y, M, K);
    }

    cudaEventRecord(start);
    if (algorithm == 0) {
        sgemv_k32<<<gridDim, blockDim>>>(a, x, y, M, K);
    } else if (algorithm == 1) {
        sgemv_k128<<<gridDim, blockDim>>>(a, x, y, M, K);
    } else {
        sgemv_k16<<<gridDim, blockDim>>>(a, x, y, M, K);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000;

    float max_error = 0.0f;
    for (int i = 0; i < M; i++) {
        max_error = fmax(max_error, fabs(y[i] - y_cpu[i]));
    }
    
    printf("Algorithm %d's max error is : %f\n", algorithm, max_error);

    cudaFree(x);
    cudaFree(y);
    cudaFree(a);
    free(y_cpu);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return sec;
}