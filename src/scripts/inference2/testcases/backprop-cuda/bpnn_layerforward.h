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
    const int hid_size = hid;
    const int input_size = hid_size;

    // Shared memory for input and weights
    __shared__ float s_input[32];
    __shared__ float s_weights[32];

    int hid_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int in_idx = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;

    // Load input and weights into shared memory
    if (hid_idx < hid_size) {
        s_input[threadIdx.x] = input[hid_idx];
    }

    if (in_idx < input_size) {
        s_weights[threadIdx.y] = input_weights[in_idx];
    }

    __syncthreads();

    // Perform matrix multiplication via tree reduction
    for (int k = 0; k < hid_size; k++) {
        sum += s_input[k] * s_weights[k];
    }

    // Store the result
    if (hid_idx < hid_size) {
        hidden_partial_sum[hid_idx] = sum;
    }
}
#endif
