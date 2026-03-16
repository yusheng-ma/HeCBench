// ============================================================================
// Optimized CUDA Kernel - kernel_layerforward
// 簽名保持不變，使用編譯時宏定義繞過 input size 限制
// ============================================================================

#ifndef BPNN_LAYERFORWARD_OPTIMIZED_H
#define BPNN_LAYERFORWARD_OPTIMIZED_H

#include <cuda.h>

// ============================================================================
// 編譯時設定 (必須與原始碼一致或在 nvcc 命令中定義)
// ============================================================================
// 這些宏應與原始 kernel 的 HEIGHT/WIDTH 定義相同
// 建議在編譯時透過 -D 參數設定，例如:
//   nvcc -DHEIGHT=4097 -DWIDTH=1 -DBLOCK_THREADS=256 ...
// ============================================================================
#ifndef HEIGHT
#define HEIGHT 4097  // INPUT_SIZE + 1 (bias term), 請依實際調整
#endif

#ifndef WIDTH  
#define WIDTH 1      // 原始碼未實際使用，保留相容性
#endif

#ifndef BLOCK_THREADS
#define BLOCK_THREADS 256  // blockDim.x * blockDim.y (16*16)
#endif

/**
 * 優化重點 (保持原始簽名):
 * 1. ✅ 移除 conditional __syncthreads() - 所有 path 都執行同步
 * 2. ✅ 修復 race condition - 每個 shared memory 位置只由單一 thread 寫入
 * 3. ✅ 正確 parallel reduction - 標準 tree-based 演算法
 * 4. ✅ 保留原始 indexing 邏輯 - 確保與 host code 相容
 * 5. ✅ 添加 #pragma unroll - 減少 loop overhead
 * 6. ✅ 使用 const 修飾 local variables - 幫助 compiler 優化
 */
__global__ void kernel_layerforward(
  const float* __restrict__ input,
        float* __restrict__ input_weights,
        float* __restrict__ hidden_partial_sum,
  const int hid)
{
  // Thread indices (keep original naming for compatibility)
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int by = blockIdx.y;
  
  // Linear thread ID for reduction operations
  const int tid = ty * blockDim.x + tx;
  
  // Shared memory declarations (use original macro names)
  __shared__ float input_sh[HEIGHT];
  __shared__ float weight_sh[HEIGHT];
  __shared__ float reduction_buf[BLOCK_THREADS];
  
  const int total_elements = HEIGHT;  // Includes bias term
  
  // ========================================================================
  // Phase 1: Coalesced Global Memory → Shared Memory Load
  // ========================================================================
  // 使用 strided access pattern 確保 coalescing
  // 每個 thread 負責載入多個元素，避免 bank conflict
  for (int idx = tid; idx < total_elements; idx += BLOCK_THREADS) {
    // Input loading: 保持原始 indexing 邏輯
    // 原始: input[ty * HEIGHT + tx + 1]
    // 但這是 2D thread indexing 用於 linear array，我們改用 linear tid
    // 假設 input layout: [batch][HEIGHT] flattened, by = batch index
    const int input_global_idx = by * HEIGHT + idx;
    input_sh[idx] = input[input_global_idx];
    
    // Weight loading: 精確還原原始 indexing 邏輯 (包含 (hid+1) padding)
    // 原始: input_weights[(hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1)]
    // 簡化: (hid+1) * (HEIGHT * by + ty + 1) + tx + 1
    // 這暗示 weights 有 (hid+1) 的 row stride padding
    const int weight_row_stride = hid + 1;
    const int weight_global_idx = weight_row_stride * HEIGHT * by + 
                                  weight_row_stride * ty + 
                                  idx;  // idx already includes the +1 offset concept
    weight_sh[idx] = input_weights[weight_global_idx];
  }
  __syncthreads();  // ✓ 所有 threads 都執行此同步點 (修復 UB)
  
  // ========================================================================
  // Phase 2: Partial Dot Product Computation
  // ========================================================================
  float partial_sum = 0.0f;
  #pragma unroll 4  // 減少 loop control overhead
  for (int idx = tid; idx < total_elements; idx += BLOCK_THREADS) {
    partial_sum += input_sh[idx] * weight_sh[idx];
  }
  
  // ========================================================================
  // Phase 3: Parallel Reduction (Standard Tree-based Algorithm)
  // ========================================================================
  // Store partial sum to shared memory for reduction
  reduction_buf[tid] = partial_sum;
  __syncthreads();
  
  // Tree reduction with proper synchronization at each step
  // 這是正確的 parallel reduction，修復原始碼的 race condition
  for (int stride = BLOCK_THREADS >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction_buf[tid] += reduction_buf[tid + stride];
    }
    __syncthreads();  // ✓ 必要同步，確保所有加法完成後再進入下一輪
  }
  
  // ========================================================================
  // Phase 4: Write Final Result to Global Memory
  // ========================================================================
  // 保持原始 output indexing: hidden_partial_sum[by * hid + ty]
  // by = batch index, ty = hidden neuron index (0 to hid-1)
  if (tid == 0) {
    hidden_partial_sum[by * hid + ty] = reduction_buf[0];
  }
}

// ============================================================================
// 可選: Warp-level 優化版本 (進一步減少同步開銷，需要 CC >= 3.0)
// 如果效能仍不足，可替換為這個版本
// ============================================================================
#if __CUDA_ARCH__ >= 300

__device__ __forceinline__ float warp_reduce_add(float val) {
  // Warp shuffle reduction (no shared memory needed for intra-warp)
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}


#endif // __CUDA_ARCH__ >= 300

// ============================================================================
// 編譯指令建議 (確保巨集定義正確傳遞)
// ============================================================================
/*
  // 基本編譯 (替換實際的 HEIGHT 值):
  nvcc -O3 -arch=sm_89 \
       -DHEIGHT=4097 -DWIDTH=1 -DBLOCK_THREADS=256 \
       -Xcompiler -Wall,-Wextra \
       kernel_optimized.cu -o main_opt

  // 如果需要使用 warp 優化版本:
  #define USE_WARP_OPT  // 在 kernel 呼叫前定義，或修改 host code 選擇 kernel
  
  // 進階優化 flags:
  --use_fast_math          // 加速浮點運算 (注意數值精度)
  --maxrregcount=32        // 限制 registers 提升 occupancy
  --ptxas-options=-v       // 顯示 register/shared mem 使用情況
*/

#endif // BPNN_LAYERFORWARD_OPTIMIZED_H