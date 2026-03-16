#!/usr/bin/env python3
# ============================================================================
# deepseek_r1_think.py
# Function: Direct prompt to DeepSeek-R1 for kernel_layerforward generation
#           No directory reading - just clear task description in prompt
# ============================================================================

import os
import re
import time
from vllm import LLM, SamplingParams

# ============================================================================
# Path Configuration
# ============================================================================
BASE_DIR = "/mnt/data1/yusheng/HeCBench/src/scripts/inference"
OUTPUT_LAYERFORWARD_PATH = os.path.join(BASE_DIR, "bpnn_layerforward_r1.h")
FULL_RESPONSE_PATH = os.path.join(BASE_DIR, "deepseek_r1_full_response.txt")

# ============================================================================
# Parse LLM Response, Extract kernel_layerforward
# ============================================================================
def extract_kernel_layerforward(response_text):
    """Extract kernel_layerforward function from LLM response"""
    kernel_name = 'kernel_layerforward'
    
    # Method 1: Find complete function definition with brace matching
    pattern = rf'__global__\s+void\s+{kernel_name}\s*\([^)]*\)'
    match = re.search(pattern, response_text)
    
    if match:
        start_idx = match.start()
        brace_start = response_text.find('{', start_idx)
        if brace_start != -1:
            brace_count = 1
            end_idx = brace_start + 1
            while end_idx < len(response_text) and brace_count > 0:
                if response_text[end_idx] == '{':
                    brace_count += 1
                elif response_text[end_idx] == '}':
                    brace_count -= 1
                end_idx += 1
            
            if brace_count == 0:
                kernel_code = response_text[start_idx:end_idx]
                print(f"✅ Extracted: {kernel_name}")
                return kernel_code
    
    # Method 2: Try to find code inside ```cuda blocks
    code_block_pattern = rf'```(?:cuda|cpp|c)?\s*([\s\S]*?{kernel_name}[\s\S]*?)```'
    code_match = re.search(code_block_pattern, response_text)
    if code_match:
        kernel_code = code_match.group(1).strip()
        print(f"✅ Extracted from code block: {kernel_name}")
        return kernel_code
    
    print(f"❌ Failed to extract: {kernel_name}")
    return None

# ============================================================================
# Write Kernel to Header File
# ============================================================================
def write_kernel_header(kernel_name, kernel_code, output_path):
    """Write kernel code to header file with include guards"""
    header_guard = os.path.basename(output_path).replace('.', '_').upper()
    
    header_content = f"""// ============================================================================
// Auto-generated CUDA Kernel by DeepSeek-R1
// Kernel: {kernel_name}
// ============================================================================

#ifndef {header_guard}
#define {header_guard}

#include <cuda.h>

{kernel_code}

#endif // {header_guard}
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(header_content)
    
    print(f"📁 Written: {output_path}")

# ============================================================================
# Report Generation Status
# ============================================================================
def report_generation_status(output, max_tokens_limit):
    """Analyze and print generation completion status"""
    print("\n" + "=" * 80)
    print("📊 Generation Status Report")
    print("=" * 80)
    
    generated_text = output.outputs[0].text
    finish_reason = output.outputs[0].finish_reason
    num_generated_tokens = len(output.outputs[0].token_ids) if hasattr(output.outputs[0], 'token_ids') else None
    
    print(f"📝 Finish Reason: {finish_reason}")
    print(f"📏 Generated Characters: {len(generated_text)}")
    if num_generated_tokens:
        print(f"🔢 Generated Tokens: {num_generated_tokens} / {max_tokens_limit}")
    
    if finish_reason == "stop":
        print("✅ Status: Generation completed naturally")
        status = "COMPLETE"
    elif finish_reason == "length":
        print("⚠️  Status: Generation stopped due to max_tokens limit!")
        status = "TRUNCATED"
    elif finish_reason == "abort":
        print("❌ Status: Generation was aborted")
        status = "ABORTED"
    else:
        print(f"❓ Status: Unknown finish reason: {finish_reason}")
        status = "UNKNOWN"
    
    if "kernel_layerforward" in generated_text:
        print("✅ Content Check: kernel_layerforward detected")
    else:
        print("⚠️  Content Check: kernel_layerforward may be missing")
    
    print("=" * 80)
    return status

# ============================================================================
# Main Program
# ============================================================================
def main():
    print("=" * 80)
    print("🚀 DeepSeek-R1 CUDA Kernel Generator (kernel_layerforward)")
    print("=" * 80)
    
    # 1️⃣ Build Direct Prompt (No directory reading - clear task description)
    print("\n📝 Building Prompt...")
    prompt = """# Role
You are an expert CUDA Developer.

# Task
Implement a CUDA kernel for BPNN forward pass.

# Few-Shot Example: Correct CUDA Style
You MUST follow this coding style exactly. Notice the use of `__global__`, `__restrict__`, and `const`.

Example Kernel:
```cuda
__global__ void example_kernel(
  const float* __restrict__ in,
        float* __restrict__ out,
  const int n)
{
  int idx = threadIdx.x;
  __shared__ float shared_data[16];
  // Load data
  shared_data[idx] = in[idx];
  __syncthreads();
  // Compute
  out[idx] = shared_data[idx] * 2.0f;
}
```

# Target Kernel Specification
You must implement `kernel_layerforward` using the EXACT signature below.
DO NOT add `const` to `input_weights`.
DO NOT use `__kernel__`. Use `__global__`.

## Function Signature (COPY EXACTLY)
__global__ void kernel_layerforward(
  const float* __restrict__ input,
        float* __restrict__ input_weights,
        float* __restrict__ hidden_partial_sum,
  const int hid)

## Constants
#define HEIGHT 16
#define WIDTH 16
#define BLOCK_SIZE 16

## Thread Indices (DEFINE EXACTLY)
int tx = threadIdx.x;
int ty = threadIdx.y;
int by = blockIdx.y;

## Shared Memory (DECLARE EXACTLY)
__shared__ float input_node[HEIGHT];
__shared__ float weight_matrix[HEIGHT * WIDTH];

## Implementation Logic
1. **Load Input**: 
   - Index: `HEIGHT * by + ty + 1`
   - Store to: `input_node[ty]`
   - Check bounds.
2. **Load Weights**: 
   - Index: `(hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1)`
   - Store to: `weight_matrix[ty * WIDTH + tx]`
   - Check bounds.
3. **Sync**: `__syncthreads()`
4. **Multiply**: `weight_matrix[ty * WIDTH + tx] *= input_node[ty]`
5. **Reduce**: Sum across `tx` using tree reduction. Use `__syncthreads()` between steps.
6. **Store Result**: If `tx == 0`, write to `hidden_partial_sum[by * hid + ty]`

# Output Requirement
- Return ONLY the CUDA code inside ```cuda ... ``` blocks.
- Ensure the signature matches the Target Kernel Specification exactly.
- Ensure the code compiles with nvcc.
"""
    
    print(f"📊 Prompt length: {len(prompt)} characters")
    
    # 2️⃣ Initialize LLM
    print("\n🤖 Initializing DeepSeek-R1 model...")
    llm = LLM(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        max_model_len=32768,
        gpu_memory_utilization=0.80,
        enforce_eager=True,
        trust_remote_code=True,
        tensor_parallel_size=1
    )
    
    # 3️⃣ Sampling Parameters (reasoning model)
    MAX_TOKENS = 32768
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=MAX_TOKENS,
    )
    
    # 4️⃣ Generate Response
    print("\n⚡ Generating CUDA kernel code...")
    start_time = time.time()
    
    outputs = llm.generate([prompt], sampling_params)
    
    generation_time = time.time() - start_time
    response_text = outputs[0].outputs[0].text
    
    print(f"⏱️  Generation time: {generation_time:.2f} seconds")
    
    # 5️⃣ Save Full Response
    with open(FULL_RESPONSE_PATH, 'w', encoding='utf-8') as f:
        f.write(response_text)
    print(f"💾 Full response saved to: {FULL_RESPONSE_PATH}")
    
    # 6️⃣ Report Generation Status
    gen_status = report_generation_status(outputs[0], MAX_TOKENS)
    
    # 7️⃣ Parse and Extract Kernel
    print("\n🔍 Parsing LLM response...")
    kernel_code = extract_kernel_layerforward(response_text)
    
    # 8️⃣ Write Output File
    print("\n💾 Writing output file...")
    if kernel_code:
        write_kernel_header('kernel_layerforward', kernel_code, OUTPUT_LAYERFORWARD_PATH)
        print("\n" + "=" * 80)
        print("✅ Success!")
        print("=" * 80)
        print(f"📁 Output: {OUTPUT_LAYERFORWARD_PATH}")
    else:
        print("\n" + "=" * 80)
        print("❌ Failed to extract kernel_layerforward")
        print("=" * 80)
    
    # 9️⃣ Final Summary
    print(f"\n🔍 Full response: {FULL_RESPONSE_PATH}")
    print(f"📊 Generation Status: {gen_status}")
    
    if gen_status == "TRUNCATED":
        print("\n⚠️  WARNING: Output may be incomplete.")
        print("💡 Try increasing max_tokens for longer output.")
    
    print("\n💡 Next: Copy header to project and compile for testing")
    print(f"   cp {OUTPUT_LAYERFORWARD_PATH} {BASE_DIR}/testcases/backprop-cuda/bpnn_layerforward.h")

if __name__ == "__main__":
    main()
