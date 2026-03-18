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

  // Define thread and block indices
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int by = blockIdx.y;

  // Shared memory for input and weights
  __shared__ float input_node[HEIGHT];
  __shared__ float weight_matrix[HEIGHT * WIDTH];

  // Calculate input index
  int input_idx = HEIGHT * by + ty;
  if (input_idx < hid) {
    input_node[ty] = input[input_idx];
  } else {
    input_node[ty] = 0.0f; // Initialize unused elements to avoid garbage
  }

  // Load weights with improved coalescing
  int weight_idx = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx;
  if (weight_idx < (hid + 1) * HEIGHT * WIDTH) {
    weight_matrix[ty * WIDTH + tx] = input_weights[weight_idx];
  } else {
    weight_matrix[ty * WIDTH + tx] = 0.0f; // Initialize unused elements
  }

  __syncthreads();

  // Multiply input and weights
  if (input_idx < hid) {
    weight_matrix[ty * WIDTH + tx] *= input_node[ty];
  }

  __syncthreads();

  // Efficient block-wise reduction
  float sum = 0.0f;
  if (tx < WIDTH) {
    sum = weight_matrix[ty * WIDTH + tx];
  }

  // Block-wise reduction
  for (int s = WIDTH / 2; s > 0; s >>= 1) {
    if (tx < s) {
      sum += weight_matrix[ty * WIDTH + tx + s];
    }
    __syncthreads();
  }

  // Store the result if tx is in the first half of the block
  if (tx == 0) {
    int output_idx = by * hid + ty;
    if (output_idx < hid) {
      hidden_partial_sum[output_idx] = sum;
    }
  }
}

#endif // BPNN_LAYERFORWARD_R2_BPNN_LAYERFORWARD_H
