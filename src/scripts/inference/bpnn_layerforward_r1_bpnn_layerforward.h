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
<<<<<<< HEAD
  const int hid) {

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int by = blockIdx.y;
=======
  const int hid)
>>>>>>> e2a610f27a322cea279a14b7de1560980a8d4a8f

  __shared__ float input_node[HEIGHT];
  __shared__ float weight_matrix[HEIGHT * WIDTH];

<<<<<<< HEAD
  // Load input
  int input_idx = HEIGHT * by + ty;
  if (input_idx < hid) {
    input_node[ty] = input[input_idx];
  }

  // Load weights
  int weight_idx = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx;
  if (weight_idx < (hid + 1) * HEIGHT * WIDTH) {
    weight_matrix[ty * WIDTH + tx] = input_weights[weight_idx];
  }
=======
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int by = blockIdx.y;
>>>>>>> e2a610f27a322cea279a14b7de1560980a8d4a8f

  // Load Input
  input_node[ty] = input[HEIGHT * by + ty + 1];
  // Load Weights
  weight_matrix[ty * WIDTH + tx] = input_weights[(hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1)];
  // Synchronize
  __syncthreads();
  // Multiply
  if (input_idx < hid) {
    weight_matrix[ty * WIDTH + tx] *= input_node[ty];
  }

  __syncthreads();

  // Reduce
  float sum = 0.0f;
  if (tx == 0) {
    for (int i = 0; i < WIDTH; ++i) {
      sum += weight_matrix[ty * WIDTH + i];
    }
  }

  __syncthreads();

  // Store result
  if (tx == 0) {
    int output_idx = by * hid + ty;
    if (output_idx < hid) {
      hidden_partial_sum[output_idx] = sum;
    }
  }

#endif // BPNN_LAYERFORWARD_R1_BPNN_LAYERFORWARD_H
