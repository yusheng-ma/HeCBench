// ============================================================================
// Auto-generated CUDA Kernel by DeepSeek-R1
// Kernel: kernel_layerforward
// ============================================================================

#ifndef BPNN_LAYERFORWARD_R2_BPNN_LAYERFORWARD_H
#define BPNN_LAYERFORWARD_R2_BPNN_LAYERFORWARD_H

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

  // Load input with boundary check
  int input_idx = HEIGHT * by + ty;
  input_node[ty] = (input_idx < hid) ? input[input_idx] : 0.0f;

  // Load weights with boundary check
  int weight_idx = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx;
  weight_matrix[ty * WIDTH + tx] = (weight_idx < (hid + 1) * HEIGHT * WIDTH) ? input_weights[weight_idx] : 0.0f;

  __syncthreads();

  // Multiply
  weight_matrix[ty * WIDTH + tx] *= input_node[ty];

  __syncthreads();

  // Reduce with improved data access pattern
  if (tx == 0) {
    float sum = 0.0f;
    for (int i = 0; i < WIDTH; ++i) {
      sum += weight_matrix[ty * WIDTH + i];
    }
    hidden_partial_sum[by * hid + ty] = sum;
  }
}

#endif // BPNN_LAYERFORWARD_R2_BPNN_LAYERFORWARD_H
