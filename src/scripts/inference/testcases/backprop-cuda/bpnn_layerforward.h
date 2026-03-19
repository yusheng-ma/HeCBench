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

  // Shared memory allocation
  __shared__ float input_node[HEIGHT];
  __shared__ float weight_matrix[HEIGHT * WIDTH];

  // Load input with coalesced access
  int input_idx = HEIGHT * by + ty;
  if (input_idx < hid) {
    input_node[ty] = input[input_idx];
  } else {
    input_node[ty] = 0.0f;
  }

  // Load weights with coalesced access
  int weight_idx = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx;
  if (weight_idx < (hid + 1) * HEIGHT * WIDTH) {
    weight_matrix[ty * WIDTH + tx] = input_weights[weight_idx];
  } else {
    weight_matrix[ty * WIDTH + tx] = 0.0f;
  }

  // Load Input
  input_node[ty] = input[HEIGHT * by + ty + 1];
  // Load Weights
  weight_matrix[ty * WIDTH + tx] = input_weights[(hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1)];
  // Synchronize
  __syncthreads();

  // Multiply with improved data locality
  if (input_idx < hid) {
    float input_val = input_node[ty];
    for (int i = 0; i < WIDTH; ++i) {
      weight_matrix[ty * WIDTH + i] *= input_val;
    }
  }

  __syncthreads();

  // Reduce with better warp utilization
  float sum = 0.0f;
  if (tx == 0) {
    for (int i = 0; i < WIDTH; ++i) {
      sum += weight_matrix[ty * WIDTH + i];
    }
  }

  __syncthreads();

  // Store result with coalesced access
  if (tx == 0) {
    int output_idx = by * hid + ty;
    if (output_idx < hid) {
      hidden_partial_sum[output_idx] = sum;
    }
  }

#endif // BPNN_LAYERFORWARD_R2_BPNN_LAYERFORWARD_H
