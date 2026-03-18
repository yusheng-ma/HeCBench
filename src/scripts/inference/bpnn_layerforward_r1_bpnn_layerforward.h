// ============================================================================
// Auto-generated CUDA Kernel by DeepSeek-R1
// Kernel: kernel_layerforward
// ============================================================================

#ifndef BPNN_LAYERFORWARD_R1_BPNN_LAYERFORWARD_H
#define BPNN_LAYERFORWARD_R1_BPNN_LAYERFORWARD_H

#include <cuda.h>

__global__ void kernel_layerforward(
  const float* __restrict__ input,
        float* __restrict__ input_weights,
        float* __restrict__ hidden_partial_sum,
  const int hid)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int by = blockIdx.y;

  __shared__ float input_node[HEIGHT];
  __shared__ float weight_matrix[HEIGHT * WIDTH];

  // Load input
  int input_idx = HEIGHT * by + ty;
  if (input_idx < hid) {
    input_node[ty] = input[input_idx];
  } else {
    input_node[ty] = 0.0f;
  }

  // Load weights
  int weight_idx = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx;
  if (weight_idx < (hid + 1) * HEIGHT * WIDTH) {
    weight_matrix[ty * WIDTH + tx] = input_weights[weight_idx];
  } else {
    weight_matrix[ty * WIDTH + tx] = 0.0f;
  }

  __syncthreads();

  // Multiply
  weight_matrix[ty * WIDTH + tx] *= input_node[ty];

  __syncthreads();

  // Reduce
  if (tx == 0) {
    float sum = 0.0f;
    for (int i = 0; i < WIDTH; ++i) {
      sum += weight_matrix[ty * WIDTH + i];
    }
    hidden_partial_sum[by * hid + ty] = sum;
  }
}

#endif // BPNN_LAYERFORWARD_R1_BPNN_LAYERFORWARD_H
