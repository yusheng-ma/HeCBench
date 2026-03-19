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
  const int hid) {

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int by = blockIdx.y;

  // Increased shared memory size for better coalescing and reuse
  __shared__ float input_node[HEIGHT * 2];
  __shared__ float weight_matrix[HEIGHT * WIDTH * 2];

  // Load input with coalesced access
  int input_idx = HEIGHT * by + ty;
  if (input_idx < hid) {
    int idx = ty * 2 + (tx % 2);
    input_node[idx] = input[input_idx + (tx % 2)];
  }

  // Load weights with coalesced access
  int weight_idx = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx;
  if (weight_idx < (hid + 1) * HEIGHT * WIDTH) {
    int idx = ty * WIDTH + tx;
    weight_matrix[idx * 2 + (tx % 2)] = input_weights[weight_idx];
  }

  __syncthreads();

  // Multiply with improved coalescing
  if (input_idx < hid) {
    int idx = ty * WIDTH + tx;
    weight_matrix[idx] *= input_node[ty * 2 + (tx % 2)];
  }

  __syncthreads();

  // Parallel reduction using tree reduction
  float sum = 0.0f;
  int stride = 1;
  for (int i = 0; i < WIDTH; i += stride) {
    if (tx == 0) {
      sum += weight_matrix[ty * WIDTH + i];
    }
    stride <<= 1;
  }

  __syncthreads();

  // Store result with coalesced access
  if (tx == 0) {
    int output_idx = by * hid + ty;
    if (output_idx < hid) {
      hidden_partial_sum[output_idx] = sum;
    }
  }
}

#endif // BPNN_LAYERFORWARD_R2_BPNN_LAYERFORWARD_H
