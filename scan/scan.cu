#include <cstdio>
#include <random>
#define WARP_SIZE 32

template <const int KWARP_SIZE = 32>
__device__ float warp_scan(float val, int lane) {
    for (int mask = 1; mask <= (KWARP_SIZE >> 1); mask <<= 1) {
        float tmp = __shfl_up_sync(0xffffffff, val, mask);
        if (lane >= mask) {
            val += tmp;
        }
    }
    return val;
}

__device__ int blockCounter;

template <const int THREAD_NUM = 256>
__global__ void inclusive_scan(float* a, float* b, float* aux, float* flag, const int N) {
    __shared__ int bid;
    int tid = threadIdx.x;
    if (tid == 0) {
        bid = atomicAdd(&blockCounter, 1);
    }
    __syncthreads();
    int idx = bid * THREAD_NUM + tid;
    constexpr int NUM_WARPS = THREAD_NUM / WARP_SIZE;
    __shared__ float s_sum[NUM_WARPS];
    __syncthreads();
    int lane = tid % WARP_SIZE;
    int warp = tid / WARP_SIZE;
    
    float val = idx < N ? a[idx] : 0;
    
    val = warp_scan<WARP_SIZE>(val, lane);

    // 最后一个元素为这个 warp 的和
    if (lane == WARP_SIZE - 1) {
        s_sum[warp] = val;
    }
    __syncthreads();
    // 跨 warp 的 scan
    if (warp == 0) {
        float warp_val = lane < NUM_WARPS ? s_sum[lane] : 0;
        warp_val = warp_scan<NUM_WARPS>(warp_val, lane);
        s_sum[lane] = lane < NUM_WARPS ? warp_val : 0;
    }
    __syncthreads();

    if (warp > 0) {
        val += s_sum[warp - 1];
    }
    __syncthreads();
    while (bid > 0 && atomicAdd(&flag[bid - 1], 0) == 0);
    float last_val = 0;
    if (bid > 0) {
        last_val = aux[bid - 1];
    }
    if (idx < N) {
        b[idx] = val + last_val;
    }
    if (tid == THREAD_NUM - 1) {
        aux[bid] = last_val + val;
        __threadfence();
        atomicAdd(&flag[bid], 1);
    }
}

void one_pass_inclu_scan(float* a, float* b, const int N) {
    float *a_d, *b_d;
    cudaMalloc((void**) &a_d, sizeof(float) * N);
    cudaMalloc((void**) &b_d, sizeof(float) * N);
    cudaMemcpy(a_d, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    dim3 blockDim(256);
    int blockCount = (N + 256 - 1) / 256;
    dim3 gridDim(blockCount);
    int tmp = 0;
    cudaMemcpyToSymbol(blockCounter, &tmp, sizeof(int));
    float *aux, *flag;
    cudaMalloc((void**) &aux, sizeof(float) * blockCount);
    cudaMalloc((void**) &flag, sizeof(float) * blockCount);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    inclusive_scan<<<gridDim, blockDim>>>(a_d, b_d, aux, flag, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    printf("gpu costs %f s\n", time / 1e3);
    cudaMemcpy(b, b_d, sizeof(float) * N, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(aux);
    cudaFree(flag);
}

int main() {
    const int N = 1e5;
    float *a = new float[N];
    float *b = new float[N];
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        a[i] = rand() / (float) RAND_MAX;
    }
    one_pass_inclu_scan(a, b, N);
    
    float *c = new float[N];
    float esp_time_cpu;
    clock_t start_cpu, stop_cpu;
    start_cpu = clock();
    c[0] = a[0];
    for (int i = 1; i < N; i++) {
        c[i] = c[i - 1] + a[i];
    }
    stop_cpu = clock();
    esp_time_cpu = (float)(stop_cpu - start_cpu) / CLOCKS_PER_SEC;
    printf("cpu costs %f s\n", esp_time_cpu);

    float max_error = 0.0;
    for (int i = 0; i < N; i++) {
        max_error = max(max_error, fabs(c[i] - b[i]));
    }
    printf("max error is %f\n", max_error);
    return 0;
}