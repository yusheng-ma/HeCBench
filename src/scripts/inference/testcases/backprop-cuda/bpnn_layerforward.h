// ============================================================================
// Auto-generated CUDA Kernel by DeepSeek-R1
// Kernel: kernel_layerforward
// ============================================================================

#ifndef BPNN_LAYERFORWARD_R2_BPNN_LAYERFORWARD_H
#define BPNN_LAYERFORWARD_R2_BPNN_LAYERFORWARD_H

#include <cuda.h>

__global__ void kernel_layerforward(const float* input,
                                    float* input_weights,
                                    float* hidden_partial_sum,
                                    const int hid) {
    __shared__ float input_node[HEIGHT];  // Shared memory for input nodes
    __shared__ float weight_matrix[HEIGHT * WIDTH];  // Shared memory for weights

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int by = blockIdx.y;

    // Load input node into shared memory
    if (input[ty * HEIGHT + tx + 1] < 0) {
        __syncthreads();
        input_node[ty] = input[ty * HEIGHT + tx + 1];
    } else {
        input_node[ty] = input[ty * HEIGHT + tx + 1];
    }

    // Load weights into shared memory
    if (input_weights[(hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1)] < 0) {
        __syncthreads();
        weight_matrix[ty * WIDTH + tx] = input_weights[(hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1)];
    } else {
        weight_matrix[ty * WIDTH + tx] = input_weights[(hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1)];
    }

    // Multiply using shared memory
    weight_matrix[ty * WIDTH + tx] *= input_node[ty];

    // Reduce using shared memory
    __syncthreads();
    for (int txId = 1; txId < tx; txId++) {
        weight_matrix[ty * WIDTH + txId] += weight_matrix[ty * WIDTH + txId - 1];
    }

    // Store the result
    if (tx == 0) {
        hidden_partial_sum[by * hid + ty] = weight_matrix[ty * WIDTH + tx];
    }
}

#endif // BPNN_LAYERFORWARD_R2_BPNN_LAYERFORWARD_H
