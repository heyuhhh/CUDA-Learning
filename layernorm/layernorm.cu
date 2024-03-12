#include <cstdio>
#include <random>
#include <float.h>
#include <ctime>
#define FLOAT4(val) (reinterpret_cast<float4*>(&(val))[0])
#define WARP_SIZE 32

template <const int KWARP_SIZE = 32>
__device__ float warp_reduce(float val) {
    for (int i = KWARP_SIZE >> 1; i > 0; i >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    return val;
}

template <const int NUM_THREADS = 256>
__device__ float blockReduce(float val) {
    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
    __shared__ float smem[NUM_WARPS];
    val = warp_reduce<WARP_SIZE>(val);
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    if (lane == 0) {
        smem[warp] = val;
    }
    __syncthreads();
    val = lane < NUM_WARPS ? smem[lane] : 0.0f;
    val = warp_reduce<NUM_WARPS>(val);
    return val;
}

template<const int NUM_THREADS = 256>
__global__ void layer_norm(float* a, float* b, const int N, const int K) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int idx = bid * K + tid * 4;
    __shared__ float mean;
    __shared__ float var;
    extern __shared__ float smem[];
    const float epsilon = 1e-5f;
    float sum = 0.0;
    #pragma unroll
    for (int i = idx; i < (bid + 1) * K; i += 4 * blockDim.x) {
        FLOAT4(smem[i % K]) = FLOAT4(a[i]);
    }
    #pragma unroll
    for (int i = 4 * tid; i < K; i += 4 * blockDim.x) {
        float4 reg = FLOAT4(smem[i]);
        if (i     < K) {
            sum += reg.x;
        }
        if (i + 1 < K) {
            sum += reg.y;
        }
        if (i + 2 < K) {
            sum += reg.z;
        }
        if (i + 3 < K) {
            sum += reg.w;
        }
    }
    sum = blockReduce<NUM_THREADS>(sum);
    if (tid == 0) {
        mean = sum / (float) K;
    }
    __syncthreads();
    
    float var_sum = 0;
    #pragma unroll
    for (int i = 4 * tid; i < K; i += 4 * blockDim.x) {
        float4 reg = FLOAT4(smem[i]);
        if (i   < K) {
            var_sum += (mean - reg.x) * (mean - reg.x);
        }
        if (i + 1 < K) {
            var_sum += (mean - reg.y) * (mean - reg.y);
        }
        if (i + 2 < K) {
            var_sum += (mean - reg.z) * (mean - reg.z);
        }
        if (i + 3 < K) {
            var_sum += (mean - reg.w) * (mean - reg.w);
        }
    }
    var_sum = blockReduce<NUM_THREADS>(var_sum);
    if (tid == 0) {
        var = rsqrtf(var_sum / (float)K + epsilon);
    }
    __syncthreads();
    #pragma unroll
    for (int i = 4 * tid; i < K; i += 4 * blockDim.x) {
        float4 reg = FLOAT4(smem[i]);
        float4 reg_o = make_float4(0, 0, 0, 0);
        if (i   < K) {
            reg_o.x = (reg.x - mean) / var;
        }
        if (i + 1 < K) {
            reg_o.y = (reg.y - mean) / var;
        }
        if (i + 2 < K) {
            reg_o.z = (reg.z - mean) / var;
        }
        if (i + 3 < K) {
            reg_o.w = (reg.w - mean) / var;
        }
        FLOAT4(b[bid * K + i]) = FLOAT4(reg_o);
    }
}

void cpu_layer_norm(float* a, float* b, const int N, const int K) {
    for (int i = 0; i < N; i++) {
        float sum = 0.0;
        for (int j = 0; j < K; j++) {
            sum += a[i * K + j];
        }
        float mean = sum / K;
        
        float var = 0;
        for (int j = 0; j < K; j++) {
            var += (a[i * K + j] - mean) * (a[i * K + j] - mean);
        }
        var = rsqrtf(var / (float) K + 1e-5);
        for (int j = 0; j < K; j++) {
            b[i * K + j] = (a[i * K + j] - mean) / var;
        }
    }
}

int main() {
    const int N = 100000;
    const int K = 256;
    float* a = new float[N * K];
    srand(time(NULL));
    for (int i = 0; i < N * K; i++) {
        a[i] = rand() / (float)RAND_MAX;
    }
    float* a_d;
    cudaMalloc((void**) &a_d, sizeof(float) * N * K);
    cudaMemcpy(a_d, a, sizeof(float) * N * K, cudaMemcpyHostToDevice);
    float* b_d;
    cudaMalloc((void**) &b_d, sizeof(float) * N * K);
    float* res = new float[N * K];
    // 当 K 较大时，考虑每个 block 处理一行，否则可以考虑每个 warp 处理一行
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    dim3 blockDim(256);
    dim3 gridDim(N);
    layer_norm<<<gridDim, blockDim, sizeof(float) * K>>>(a_d, b_d, N, K);
    cudaMemcpy(res, b_d, sizeof(float) * N * K, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    float* b = new float[N * K];
    float esp_time_cpu;
	clock_t start_cpu, stop_cpu;
    start_cpu = clock();
    cpu_layer_norm(a, b, N, K);
    stop_cpu = clock();
    esp_time_cpu = (float)(stop_cpu - start_cpu) / CLOCKS_PER_SEC;
    float max_error = 0;
    for (int i = 0; i < N * K; i++) {
        max_error = fmaxf(max_error, fabs(res[i] - b[i]));
    }
    printf("cpu costs %fs, gpu costs %fs, max error is %f\n", esp_time_cpu, time / 1e3, max_error);
    cudaFree(a_d);
    cudaFree(b_d);
}