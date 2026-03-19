// Auto-generated CUDA Kernel by DeepSeek-R1
#ifndef TEMP_CONVOLUTION3D_CUDA_2_H
#define TEMP_CONVOLUTION3D_CUDA_2_H
#include <cuda.h>
__global__ void conv3d_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    int dimx, int dimy, int dimz, int kdimx, int kdimy, int kdimz) {
    
    // Shared memory for input tile and kernel
    __shared__ float s_input[BLOCK_X * BLOCK_Y * BLOCK_Z];
    __shared__ float s_kernel[KERNEL_BLOCK_X * KERNEL_BLOCK_Y * KERNEL_BLOCK_Z];
    
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    
    // Block indices
    int bx = blockIdx.x * BLOCK_X;
    int by = blockIdx.y * BLOCK_Y;
    int bz = blockIdx.z * BLOCK_Z;
    
    // Load input tile into shared memory
    for (int i = 0; i < BLOCK_X; ++i) {
        for (int j = 0; j < BLOCK_Y; ++j) {
            for (int k = 0; k < BLOCK_Z; ++k) {
                int idx = i + j * BLOCK_X + k * BLOCK_X * BLOCK_Y;
                int x = bx + tx + i;
                int y = by + ty + j;
                int z = bz + tz + k;
                if (x < dimx && y < dimy && z < dimz) {
                    s_input[idx] = input[x * dimy * dimz + y * dimz + z];
                } else {
                    s_input[idx] = 0.0f;
                }
            }
        }
    }
    
    // Load kernel tile into shared memory
    for (int i = 0; i < KERNEL_BLOCK_X; ++i) {
        for (int j = 0; j < KERNEL_BLOCK_Y; ++j) {
            for (int k = 0; k < KERNEL_BLOCK_Z; ++k) {
                int idx = i + j * KERNEL_BLOCK_X + k * KERNEL_BLOCK_X * KERNEL_BLOCK_Y;
                int x = tx + i;
                int y = ty + j;
                int z = tz + k;
                if (x < kdimx && y < kdimy && z < kdimz) {
                    s_kernel[idx] = kernel[x * kdimy * kdimz + y * kdimz + z];
                } else {
                    s_kernel[idx] = 0.0f;
                }
            }
        }
    }
    
    // Synchronize to ensure shared memory is loaded
    __syncthreads();
    
    // Compute convolution
    float result = 0.0f;
    for (int i = 0; i < KERNEL_BLOCK_X; ++i) {
        for (int j = 0; j < KERNEL_BLOCK_Y; ++j) {
            for (int k = 0; k < KERNEL_BLOCK_Z; ++k) {
                int idx_input = (tx + i) + (ty + j) * BLOCK_X + (tz + k) * BLOCK_X * BLOCK_Y;
                int idx_kernel = i + j * KERNEL_BLOCK_X + k * KERNEL_BLOCK_X * KERNEL_BLOCK_Y;
                result += s_input[idx_input] * s_kernel[idx_kernel];
            }
        }
    }
    
    // Write result to output
    int x = bx + tx;
    int y = by + ty;
    int z = bz + tz;
    if (x < dimx && y < dimy && z < dimz) {
        output[x * dimy * dimz + y * dimz + z] = result;
    }
}
#endif
