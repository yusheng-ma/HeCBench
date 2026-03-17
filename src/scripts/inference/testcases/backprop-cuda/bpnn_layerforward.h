// ============================================================================
// Auto-generated CUDA Kernel by DeepSeek-R1
// Kernel: kernel_layerforward
// ============================================================================

#ifndef BPNN_LAYERFORWARD_R2_BPNN_LAYERFORWARD_H
#define BPNN_LAYERFORWARD_R2_BPNN_LAYERFORWARD_H

#include <cuda.h>

__global__ void kernel_layerforward(
  const float* input,
  float* input_weights,
  float* hidden_partial_sum,
  const int hid)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int by = blockIdx.y;

  __shared__ float input_node[HEIGHT];
  __shared__ float weight_matrix[HEIGHT * WIDTH];

  // Load input node
  if (input[ty * HEIGHT + tx + 1] < 0) {
    input_node[ty] = input[ty * HEIGHT + tx + 1];
  } else {
    input_node[ty] = input[ty * HEIGHT + tx + 1];
  }

  // Load weights
  if (input_weights[(hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1)] < 0) {
    weight_matrix[ty * WIDTH + tx] = input_weights[(hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1)];
  } else {
    weight_matrix[ty * WIDTH + tx] = input_weights[(hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1)];
  }

  // Multiply
  weight_matrix[ty * WIDTH + tx] *= input_node[ty];

  // Reduce using tree reduction
  __syncthreads();
  for (int txId = 1; txId < tx; txId++) {
    weight_matrix[ty * WIDTH + txId] += weight_matrix[ty * WIDTH + txId - 1];
  }
}

#endif // BPNN_LAYERFORWARD_R2_BPNN_LAYERFORWARD_H
