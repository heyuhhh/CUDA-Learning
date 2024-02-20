#include <cstdio>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

float testError(void);

float testPerformance(
    void (*gpuGemm) (float*, float*, float*, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat
);

void cpuSgemm(
    float *a, float *b, float *c, const int M, const int N, const int K) {

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = psum;
        }
    }
}

__global__
void Sgemm_v3(
    float* __restrict__ a, float* __restrict__ b, float* __restrict__ c,
    const int M, const int N, const int K) {
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[2][BK][BM];
    __shared__ float s_b[2][BK][BN];

    // 考虑 TM TN 来得到对应的计算行、列
    int load_a_smem_m = (tid >> 1);
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = (tid >> 5);
    int load_b_smem_n = (tid & 31) << 2;

    int load_a_gmem_m = by * BM + load_a_smem_m; // global row of a
    int load_b_gmem_n = bx * BN + load_b_smem_n; // global col of b

    float r_load_a[4];
    float r_load_b[4];
    // 第一次访存，无法避免
    {
        int bk = 0;
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_b_gmem_k = bk * BK + load_b_smem_k;

        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);

        // 按列进行 a 的存取
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

        s_a[0][load_a_smem_k][load_a_smem_m] = r_load_a[0];
        s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];

        // 每个thread在sub-block中每次读取4个元素
        FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);
        __syncthreads();
    }

    float r_c[TM][TN] = {0.0}; // 注意这是 local memory
    float r_a_comp[TM];
    float r_b_comp[TN];
    
    // 后续的访存与计算并行
    for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_b_gmem_k = bk * BK + load_b_smem_k;

        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);

        // global -> local 先进行，用计算隐藏延迟
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

        // 现在每个线程处理四个分块，行来自 m, m + BM / 2，列来自 n, n + BN / 2
        for (int k = 0; k < BK; k++) {
            // 每个矩阵分分块读取，因为一个矩阵读8个，都两次 128bit 就好了
            // 注意现在一个子矩阵大小是 [TM / 2, TN / 2]，左上角的计算得按照这个来
            FLOAT4(r_a_comp[0]) = FLOAT4(s_a[!(bk & 1)][k][ty * TM / 2]);
            FLOAT4(r_a_comp[4]) = FLOAT4(s_a[!(bk & 1)][k][ty * TM / 2 + BM / 2]);
            FLOAT4(r_b_comp[0]) = FLOAT4(s_b[!(bk & 1)][k][tx * TN / 2]);
            FLOAT4(r_b_comp[4]) = FLOAT4(s_b[!(bk & 1)][k][tx * TN / 2 + BN / 2]);
            for (int m = 0; m < TM; m++) {
                for (int n = 0; n < TN; n++) {
                    r_c[m][n] += r_a_comp[m] * r_b_comp[n];
                }
            }
        }

        s_a[bk & 1][load_a_smem_k][load_a_smem_m] = r_load_a[0];
        s_a[bk & 1][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[bk & 1][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[bk & 1][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];

        // 每个thread在sub-block中每次读取4个元素
        FLOAT4(s_b[bk & 1][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);
        __syncthreads();
    }

    // 最后一次计算
    {
        int bk = (K + BK - 1) / BK;
        for (int k = 0; k < BK; k++) {
            // 每个矩阵分分块读取，因为一个矩阵读8个，都两次 128bit 就好了
            // 注意现在一个子矩阵大小是 [TM / 2, TN / 2]，左上角的计算得按照这个来
            FLOAT4(r_a_comp[0]) = FLOAT4(s_a[!(bk & 1)][k][ty * TM / 2]);
            FLOAT4(r_a_comp[4]) = FLOAT4(s_a[!(bk & 1)][k][ty * TM / 2 + BM / 2]);
            FLOAT4(r_b_comp[0]) = FLOAT4(s_b[!(bk & 1)][k][tx * TN / 2]);
            FLOAT4(r_b_comp[4]) = FLOAT4(s_b[!(bk & 1)][k][tx * TN / 2 + BN / 2]);
            for (int m = 0; m < TM; m++) {
                for (int n = 0; n < TN; n++) {
                    r_c[m][n] += r_a_comp[m] * r_b_comp[n];
                }
            }
        }
    }

    // 每个线程将自己的那块 TM * TN 矩阵传输到全局内存中
    // 注意考虑分块的影响，先将上半部分放进去，再放下半部分
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        for (int j = 0; j < TN / 2; j += 4) {
            int store_c_gmem_n = bx * BN + tx * TN / 2 + j;
            int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
            FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][j]);

            store_c_gmem_n = bx * BN + tx * TN / 2 + j + BN / 2;
            store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
            FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][j + TN / 2]);
        }
    }
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i + BM / 2;
        for (int j = 0; j < TN / 2; j += 4) {
            int store_c_gmem_n = bx * BN + tx * TN / 2 + j;
            int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
            FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][j]);

            store_c_gmem_n = bx * BN + tx * TN / 2 + j + BN / 2;
            store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
            FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][j + TN / 2]);
        }
    }
}

int main() {
    float max_error = testError();
    printf("Max Error = %f\n", max_error);

    printf("\nKernal = Sgemm_v3\n");
    const int M_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int N_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int K_list[15] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};
    
    const int outer_repeat = 10, inner_repeat = 1;
    void (*gpuGemm) (float *, float *, float *, const int, const int, const int) = Sgemm_v3;
    const int TESTNUM = 15;
    const int BM = 128, BN = 128;
    const int TM = 8, TN = 8;
    for (int i = 0; i < TESTNUM; i++) {
        const int M = M_list[i], N = N_list[i], K = K_list[i];
        dim3 blockDim(BN / TN, BM / TM);
        dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        for (int j = 0; j < outer_repeat; j++) {
            double this_sec = testPerformance(gpuGemm, gridDim, blockDim, M, N, K, inner_repeat);
            max_sec = max(max_sec, this_sec);
            min_sec = min(min_sec, this_sec);
            total_sec += this_sec;
        }

        double avg_sec = total_sec / outer_repeat;
        // 有效带宽：计算量 / 时间
        // (/ 1024 / 1024 / 1024) FLOPS->GFLOPS
        double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

        printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
    }
    return 0;
}

float testError(void) {
    const int BM = 128, BN = 128;
    const int TM = 8, TN = 8;
    const int M = 512, N = 512, K = 512;
    dim3 blockDim(BN / TN, BM / TM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c, *h_d_c;
    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c = (float *)malloc(size_c);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    h_d_c = (float *)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++)
        h_a[i] = rand() / float(RAND_MAX);
    for (int i = 0; i < K * N; i++)
        h_b[i] = rand() / float(RAND_MAX);
    cudaMemset(d_c, 15, size_c);

    cpuSgemm(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    Sgemm_v3<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs(h_d_c[i] - h_c[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = max(max_error, this_error);
    }

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_d_c);

    return max_error;
}

float testPerformance(
    void (*gpuGemm) (float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat) {
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        gpuGemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return sec;
}