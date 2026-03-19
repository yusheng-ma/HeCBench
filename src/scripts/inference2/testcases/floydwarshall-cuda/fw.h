// Auto-generated CUDA Kernel by DeepSeek-R1
#ifndef TEMP_FLOYDWARSHALL_CUDA_2_H
#define TEMP_FLOYDWARSHALL_CUDA_2_H
#include <cuda.h>
__global__ void fw_kernel(
    float* __restrict__ dist,
    const int n) {
    const int BLOCK_SIZE = 16;
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    int i = blockRow * BLOCK_SIZE + threadRow;
    int j = blockCol * BLOCK_SIZE + threadCol;

    for (int k = 0; k < n; ++k) {
        // Load the current tile into shared memory
        for (int row = 0; row < BLOCK_SIZE; ++row) {
            for (int col = 0; col < BLOCK_SIZE; ++col) {
                int globalRow = blockRow * BLOCK_SIZE + row;
                int globalCol = blockCol * BLOCK_SIZE + col;
                if (globalRow < n && globalCol < n) {
                    tile[row][col] = dist[globalRow * n + globalCol];
                } else {
                    tile[row][col] = INF; // Handle out of bounds
                }
            }
        }
        __syncthreads();

        // Update the tile based on k
        if (i < n && j < n) {
            if (i == j) {
                // Ensure diagonal is zero
                tile[threadRow][threadCol] = 0.0f;
            } else {
                float throughK = dist[i * n + k] + dist[k * n + j];
                if (throughK < tile[threadRow][threadCol]) {
                    tile[threadRow][threadCol] = throughK;
                }
            }
        }
        __syncthreads();

        // Write the updated tile back to global memory
        for (int row = 0; row < BLOCK_SIZE; ++row) {
            for (int col = 0; col < BLOCK_SIZE; ++col) {
                int globalRow = blockRow * BLOCK_SIZE + row;
                int globalCol = blockCol * BLOCK_SIZE + col;
                if (globalRow < n && globalCol < n) {
                    dist[globalRow * n + globalCol] = tile[row][col];
                }
            }
        }
        __syncthreads();
    }
}
#endif
