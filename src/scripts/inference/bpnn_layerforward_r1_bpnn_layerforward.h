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

  __shared__ float input_node[HEIGHT];
  __shared__ float weight_matrix[HEIGHT * WIDTH];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int by = blockIdx.y;

  // Load Input
  input_node[ty] = input[HEIGHT * by + ty + 1];
  // Load Weights
  weight_matrix[ty * WIDTH + tx] = input_weights[(hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1)];
  // Synchronize
  __syncthreads();
  // Multiply
  weight_matrix[ty * WIDTH + tx] *= input_node[ty];
  // Reduce
  __syncthreads();
  // Store Result
  if (tx == 0) {
    hidden_partial_sum[by * hid + ty] = weight_matrix[ty * WIDTH + tx];
  }

#endif // BPNN_LAYERFORWARD_R1_BPNN_LAYERFORWARD_H
