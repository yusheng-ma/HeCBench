// Auto-generated CUDA Kernel by DeepSeek-R1
#ifndef TEMP_SOFTMAX_CUDA_2_H
#define TEMP_SOFTMAX_CUDA_2_H
#include <cuda.h>
__global__ void softmax_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int n, const int dim)
{
    const int BLOCK_SIZE = 256;
    __shared__ float s_data[BLOCK_SIZE];

    int idx;
    if (dim == 1) {
        idx = blockIdx.x * n + threadIdx.x;
    } else {
        idx = blockIdx.x + threadIdx.x * n;
    }

    if (idx >= n * (dim == 1 ? gridDim.x : gridDim.x * n / n)) {
        return;
    }

    float val = input[idx];
    s_data[threadIdx.x] = val;
    __syncthreads();

    float max_val = -FLT_MAX;
    for (int i = 0; i < BLOCK_SIZE; i += 8) {
        max_val = max(max_val, s_data[i]);
        max_val = max(max_val, s_data[i+1]);
        max_val = max(max_val, s_data[i+2]);
        max_val = max(max_val, s_data[i+3]);
        max_val = max(max_val, s_data[i+4]);
        max_val = max(max_val, s_data[i+5]);
        max_val = max(max_val, s_data[i+6]);
        max_val = max(max_val, s_data[i+7]);
    }

    float sum = 0.0f;
    for (int i = 0; i < BLOCK_SIZE; i += 8) {
        sum += exp(s_data[i] - max_val);
        sum += exp(s_data[i+1] - max_val);
        sum += exp(s_data[i+2] - max_val);
        sum += exp(s_data[i+3] - max_val);
        sum += exp(s_data[i+4] - max_val);
        sum += exp(s_data[i+5] - max_val);
        sum += exp(s_data[i+6] - max_val);
        sum += exp(s_data[i+7] - max_val);
    }

    float inv_sum = 1.0f / sum;
    __syncthreads();

    if (threadIdx.x < n) {
        output[idx] = s_data[threadIdx.x] * inv_sum;
    }
}
#endif
