// Auto-generated CUDA Kernel by DeepSeek-R1
#ifndef TEMP_BACKPROP_CUDA_2_H
#define TEMP_BACKPROP_CUDA_2_H
#include <cuda.h>
__global__ void kernel_layerforward(
    const float* __restrict__ input,
    float* __restrict__ input_weights,
    float* __restrict__ hidden_partial_sum,
    const int hid)
{
    // Shared memory for input and weights, combined for better access
    __shared__ float s_data[2 * BLOCK_SIZE][BLOCK_SIZE];

    // Block and thread indices
    int blockIdx_x = blockIdx.x;
    int threadIdx_x = threadIdx.x;
    int threadIdx_y = threadIdx.y;

    // Initialize partial sum
    float sum = 0.0f;

    // Load input and weights into shared memory with coalesced access
    int input_idx = blockIdx_x * BLOCK_SIZE + threadIdx_x;
    int weight_idx = hid * BLOCK_SIZE + threadIdx_y;

    // Load input if within bounds
    if (input_idx < hid) {
        s_data[threadIdx_y][threadIdx_x] = input[input_idx];
    } else {
        s_data[threadIdx_y][threadIdx_x] = 0.0f;
    }

    // Load weights if within bounds
    if (weight_idx < hid * BLOCK_SIZE) {
        s_data[threadIdx_y + BLOCK_SIZE][threadIdx_x] = input_weights[weight_idx];
    } else {
        s_data[threadIdx_y + BLOCK_SIZE][threadIdx_x] = 0.0f;
    }

    // Synchronize to ensure shared memory is loaded
    __syncthreads();

    // Perform matrix multiplication with unrolled loop for better ILP
    for (int k = 0; k < BLOCK_SIZE; k += 8) {
        // Unroll 8 iterations at a time
        sum += s_data[threadIdx_y][k] * s_data[threadIdx_y + BLOCK_SIZE][threadIdx_x];
        sum += s_data[threadIdx_y][k + 1] * s_data[threadIdx_y + BLOCK_SIZE][threadIdx_x + 1];
        sum += s_data[threadIdx_y][k + 2] * s_data[threadIdx_y + BLOCK_SIZE][threadIdx_x + 2];
        sum += s_data[threadIdx_y][k + 3] * s_data[threadIdx_y + BLOCK_SIZE][threadIdx_x + 3];
        sum += s_data[threadIdx_y][k + 4] * s_data[threadIdx_y + BLOCK_SIZE][threadIdx_x + 4];
        sum += s_data[threadIdx_y][k + 5] * s_data[threadIdx_y + BLOCK_SIZE][threadIdx_x + 5];
        sum += s_data[threadIdx_y][k + 6] * s_data[threadIdx_y + BLOCK_SIZE][threadIdx_x + 6];
        sum += s_data[threadIdx_y][k + 7] * s_data[threadIdx_y + BLOCK_SIZE][threadIdx_x + 7];
    }

    // Write the result to global memory with coalesced access
    if (threadIdx_x < hid) {
        hidden_partial_sum[blockIdx_x * BLOCK_SIZE + threadIdx_x] = sum;
    }
}
#endif
