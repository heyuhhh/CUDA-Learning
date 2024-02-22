#include <cstdio>
#include <random>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define WARP_COUNT 6
#define PARTIAL_HISTOGRAM256_COUNT 240
#define HIST_256_BIN 256

template<const int NUM_WARPS = WARP_SIZE>
__device__ uint warp_reduce_sum(uint val) {
    for (int i = NUM_WARPS >> 1; i > 0; i >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    return val;
}
__device__ void addByte(uint* s_WarpHist, uint data) {
    atomicAdd(s_WarpHist + data, 1);
}
__device__ void addWord(uint* s_WarpHist, uint data) {
    // 因为是 256bin，所以每次取 8 位，2^8 = 256
    addByte(s_WarpHist, (data >> 0) & 0xFFU);
    addByte(s_WarpHist, (data >> 8) & 0xFFU);
    addByte(s_WarpHist, (data >> 16) & 0xFFU);
    addByte(s_WarpHist, (data >> 24) & 0xFFU);
}

// 按照 Warp 为单位生成 block histogram
// 通过 smem 合并 block 信息
template<const int BIN=256, const int NUM_THREADS=WARP_COUNT * WARP_SIZE, const int NUM_BLOCKS=PARTIAL_HISTOGRAM256_COUNT>
__global__ void histogram256Kernel(uint* d_PartialHistograms, uint* d_Data, uint dataCount) {
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;
    int warp = tid / WARP_SIZE;
    __shared__ uint s_Hist[WARP_COUNT * BIN];
    uint *s_WarpHist = s_Hist + warp * BIN;
    
    for (int i = threadIdx.x; i < WARP_COUNT * BIN; i += blockDim.x) {
        s_Hist[i] = 0;
    }
    __syncthreads();

    // 读取全局data，记录信息到 warp histogram 中
    for (int i = idx; i < dataCount; i += blockDim.x * gridDim.x) {
        uint data = d_Data[i];
        addWord(s_WarpHist, data);
    }
    __syncthreads();

    // 合并 warp hist 的信息到 block hist 中
    for (int i = tid; i < BIN; i += blockDim.x) {
        uint sum = 0;
        for (int j = 0; j < WARP_COUNT; j++) {
            sum += s_Hist[j * BIN + i];
        }
        d_PartialHistograms[blockIdx.x * BIN + i] = sum;
    }
}

// 合并不同 block 的 hist 信息
// 每个 block 负责 output 的一个 bin
// 每个 thread 负责收集 block hist 对应的信息，然后做 warwp reduce 求和
template<const int BIN=256, const int NUM_THREADS=256, const int NUM_BLOCKS=256>
__global__ void mergeHist256Kernel(uint* d_Histogram, uint* d_PartialHistograms, uint histogramCount) {
    int tid = threadIdx.x;
    uint sum = 0;
    // 每个 thread 读取对应的 block hist 的信息
    // histogramCount 就是 block hist 的数量
    for (int i = tid; i < histogramCount; i += blockDim.x) {
        sum += d_PartialHistograms[blockIdx.x + i * BIN];
    }
    sum = warp_reduce_sum(sum);
    if (tid == 0) {
        d_Histogram[blockIdx.x] = sum;
    }
}

int main() {
    const int N = 2000;
    uint *a, *block_hist, *global_hist;
    cudaMallocManaged((void**) &a, sizeof(uint) * N);
    cudaMallocManaged((void**) &block_hist, sizeof(uint) * HIST_256_BIN * PARTIAL_HISTOGRAM256_COUNT);
    cudaMallocManaged((void**) &global_hist, sizeof(uint) * HIST_256_BIN);

    uint *hist_cpu = (uint*) malloc(sizeof(uint) * HIST_256_BIN);
    for (int i = 0; i < HIST_256_BIN; i++) {
        hist_cpu[i] = 0;
    }
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        a[i] = rand();
        uint data = a[i];
        hist_cpu[(data >> 0) & 0xFFU] += 1;
        hist_cpu[(data >> 8) & 0xFFU] += 1;
        hist_cpu[(data >> 16) & 0xFFU] += 1;
        hist_cpu[(data >> 24) & 0xFFU] += 1;
    }

    dim3 gridDim(PARTIAL_HISTOGRAM256_COUNT);
    dim3 blockDim(WARP_COUNT * WARP_SIZE);
    histogram256Kernel<<<gridDim, blockDim>>>(block_hist, a, N);
    gridDim = dim3(HIST_256_BIN); // 注意块个数必须为 BIN 大小
    blockDim = dim3(HIST_256_BIN);
    mergeHist256Kernel<<<gridDim, blockDim>>>(global_hist, block_hist, PARTIAL_HISTOGRAM256_COUNT);
    cudaDeviceSynchronize();

    for (int i = 0; i < HIST_256_BIN; i++) {
        printf("i = %d: cpu = %d, gpu = %d\n", i, hist_cpu[i], global_hist[i]);
    }
    return 0;
}