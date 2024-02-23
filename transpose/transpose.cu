#include <cstdio>
#include <algorithm>
#include <iostream>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void transpose_kernel(int* a, int* a_o, const int M, const int N) {
    __shared__ int tile[TILE_SIZE][TILE_SIZE + 1]; // avoid bank conflict
    int row = TILE_SIZE * blockIdx.y + threadIdx.y;
    int col = TILE_SIZE * blockIdx.x + threadIdx.x;
    if (row < M && col < N) {
        // 按列读取，合并访存
        tile[threadIdx.y][threadIdx.x] = a[row * N + col];
    } else {
        tile[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();

    // tile transpose
    int n_row = TILE_SIZE * blockIdx.x + threadIdx.y;
    int n_col = TILE_SIZE * blockIdx.y + threadIdx.x;
    
    // 注意转置后行列发生了变化
    if (n_row < N && n_col < M) {
        // 按列写入，合并访存，读取 smem 的时候也不会发生bank conflict
        a_o[n_row * M + n_col] = tile[threadIdx.x][threadIdx.y];
    }
}

void transpose(int* a, int* a_out, const int M, const int N) {
    int *a_d, *a_d_output;
    cudaMalloc((void**) &a_d, sizeof(int) * M * N);
    cudaMalloc((void**) &a_d_output, sizeof(int) * M * N);
    cudaMemcpy(a_d, a, sizeof(int) * M * N, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((M + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    transpose_kernel<<<gridDim, blockDim>>>(a_d, a_d_output, M, N);
    cudaMemcpy(a_out, a_d_output, sizeof(int) * M * N, cudaMemcpyDeviceToHost);
}

int main() {
    const int M = 50, N = 50;
    int* a = (int*) malloc(sizeof(int) * M * N);
    srand(time(NULL));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = rand() % 10;
        }
    }
    int *a_trans = (int*) malloc(sizeof(int) * M * N);
    transpose(a, a_trans, M, N);
}