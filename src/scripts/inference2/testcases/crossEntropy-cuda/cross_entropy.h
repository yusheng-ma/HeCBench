// Auto-generated CUDA Kernel by DeepSeek-R1
#ifndef TEMP_CROSSENTROPY_CUDA_2_H
#define TEMP_CROSSENTROPY_CUDA_2_H
#include <cuda.h>
__global__ void cross_entropy_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int n)
{
    // Shared memory for reduction
    __shared__ float sdata[256];
    
    // Thread index
    int tid = threadIdx.x;
    
    // Compute log probabilities and store in shared memory
    float log_prob = logf(input[tid] + 1e-8f); // Larger epsilon for stability
    sdata[tid] = log_prob;
    
    // Parallel reduction
    for (int stride = 1; stride < blockDim.x; stride <<= 1)
    {
        if (tid >= stride)
        {
            sdata[tid] += sdata[tid - stride];
        }
    }
    
    // Write the result to output
    if (tid == 0)
    {
        output[blockIdx.x] = -sdata[0];
    }
}
#endif
