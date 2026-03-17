#!/usr/bin/env python3
# ============================================================================
# deepseek_r1_multiround.py
# Function: Multi-round CUDA kernel generation pipeline
#           Round 1: Naive implementation → compile
#           Round 2: Optimization prompt → compile
#           NO tool use during generation (ncu benchmarking is manual)
# ============================================================================

import os
import re
import time
import shutil
import subprocess
from vllm import LLM, SamplingParams

# ============================================================================
# Path Configuration
# ============================================================================
BASE_DIR = "/mnt/data1/yusheng/HeCBench/src/scripts/inference"
TESTCASE_DIR = os.path.join(BASE_DIR, "testcases/backprop-cuda")
HEADER_NAME = "bpnn_layerforward.h"

OUTPUT_PATH_ROUND1 = os.path.join(BASE_DIR, f"bpnn_layerforward_r1_{HEADER_NAME}")
OUTPUT_PATH_ROUND2 = os.path.join(BASE_DIR, f"bpnn_layerforward_r2_{HEADER_NAME}")
FULL_RESPONSE_R1 = os.path.join(BASE_DIR, "deepseek_r1_round1_full_response.txt")
FULL_RESPONSE_R2 = os.path.join(BASE_DIR, "deepseek_r1_round2_full_response.txt")

# ============================================================================
# Parse LLM Response, Extract kernel_layerforward
# ============================================================================
def extract_kernel_layerforward(response_text):
    """Extract kernel_layerforward function from LLM response"""
    kernel_name = 'kernel_layerforward'
    
    # Method 1: Find complete function definition with brace matching
    pattern = r'__global__\s+void\s+' + kernel_name + r'\s*\([^)]*\)'
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
    code_block_pattern = r'```(?:cuda|cpp|c)?\s*([\s\S]*?' + kernel_name + r'[\s\S]*?)```'
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
    basename = os.path.basename(output_path)
    header_guard = basename.replace('.', '_').replace('-', '_').upper()
    
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
# Copy Header to Testcase Directory
# ============================================================================
def copy_header_to_testcase(source_path, testcase_dir, header_name):
    """Copy generated header to testcase directory"""
    dest_path = os.path.join(testcase_dir, header_name)
    shutil.copy2(source_path, dest_path)
    print(f"📋 Copied: {source_path} → {dest_path}")
    return dest_path

# ============================================================================
# Compile Testcase
# ============================================================================
def compile_testcase(testcase_dir):
    """Run make clean && make in testcase directory"""
    print(f"🔨 Compiling in {testcase_dir}...")
    
    result_clean = subprocess.run(
        ["make", "clean"],
        cwd=testcase_dir,
        capture_output=True,
        text=True
    )
    if result_clean.returncode != 0:
        print(f"⚠️  make clean warning: {result_clean.stderr}")
    
    result_make = subprocess.run(
        ["make"],
        cwd=testcase_dir,
        capture_output=True,
        text=True
    )
    
    if result_make.returncode == 0:
        print("✅ Compilation successful!")
        return True
    else:
        print(f"❌ Compilation failed!")
        print(f"   stdout: {result_make.stdout}")
        print(f"   stderr: {result_make.stderr}")
        return False

# ============================================================================
# Report Generation Status
# ============================================================================
def report_generation_status(output, max_tokens_limit, round_num):
    """Analyze and print generation completion status"""
    print(f"\n{'='*80}")
    print(f"📊 Round {round_num} Generation Status Report")
    print(f"{'='*80}")
    
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
    
    print(f"{'='*80}")
    return status

# ============================================================================
# Build Round 1 Prompt (Naive Implementation) - ENGLISH
# ============================================================================
def build_round1_prompt():
    """Build prompt for naive/compiled implementation"""
    return """# Role
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

# ============================================================================
# Build Round 2 Prompt (Optimization Request) - ENGLISH
# ============================================================================
def build_round2_prompt(naive_code_context):
    """Build prompt for optimized implementation based on naive version"""
    return f"""# Role
You are an expert CUDA Developer.

# Task
Optimize belowed CUDA kernel for BPNN forward pass.
You MUST follow this coding style exactly. Notice the use of `__global__`, `__restrict__`, and `const`.

{naive_code_context}

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

## Thread Indices (DEFINE EXACTLY)
int tx = threadIdx.x;
int ty = threadIdx.y;
int by = blockIdx.y;

# Output Requirement
- Return ONLY the CUDA code inside ```cuda ... ``` blocks.
- Ensure the signature matches the Target Kernel Specification exactly.
- Ensure the code compiles with nvcc.
<think>
</think>
"""

# ============================================================================
# Main Pipeline
# ============================================================================
def main():
    print("=" * 80)
    print("🚀 DeepSeek-R1 Multi-Round CUDA Kernel Generation Pipeline")
    print("   Round 1: Naive Implementation → Compile")
    print("   Round 2: Optimization Prompt → Compile")
    print("   ⚠️  ncu benchmarking is MANUAL (run after script completes)")
    print("=" * 80)
    
    # Initialize LLM once (reuse for both rounds)
    print("\n🤖 Initializing DeepSeek-R1 model...")
    llm = LLM(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        max_model_len=32768,
        gpu_memory_utilization=0.80,
        enforce_eager=True,
        trust_remote_code=True,
        tensor_parallel_size=1
    )
    
    MAX_TOKENS = 32768
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=MAX_TOKENS,
    )
    
    # =========================================================================
    # ROUND 1: Generate Naive Implementation
    # =========================================================================
    print(f"\n{'#'*80}")
    print("# ROUND 1: Generate Naive/Initial Implementation")
    print(f"{'#'*80}")
    
    print("\n📝 Building Round 1 prompt (naive implementation)...")
    prompt_r1 = build_round1_prompt()
    print(f"📊 Prompt length: {len(prompt_r1)} characters")
    
    print("\n⚡ Generating naive kernel code...")
    start_time_r1 = time.time()
    outputs_r1 = llm.generate([prompt_r1], sampling_params)
    generation_time_r1 = time.time() - start_time_r1
    response_text_r1 = outputs_r1[0].outputs[0].text
    
    print(f"⏱️  Round 1 generation time: {generation_time_r1:.2f} seconds")
    
    with open(FULL_RESPONSE_R1, 'w', encoding='utf-8') as f:
        f.write(response_text_r1)
    print(f"💾 Round 1 full response saved: {FULL_RESPONSE_R1}")
    
    status_r1 = report_generation_status(outputs_r1[0], MAX_TOKENS, round_num=1)
    
    print("\n🔍 Parsing Round 1 response...")
    kernel_code_r1 = extract_kernel_layerforward(response_text_r1)
    
    if not kernel_code_r1:
        print("❌ Failed to extract kernel from Round 1. Exiting.")
        return
    
    print("\n💾 Writing Round 1 output...")
    write_kernel_header('kernel_layerforward', kernel_code_r1, OUTPUT_PATH_ROUND1)
    
    print("\n📋 Deploying Round 1 implementation...")
    copy_header_to_testcase(OUTPUT_PATH_ROUND1, TESTCASE_DIR, HEADER_NAME)
    
    compile_success_r1 = compile_testcase(TESTCASE_DIR)
    if not compile_success_r1:
        print("⚠️  Round 1 compilation failed. Continuing to Round 2 anyway...")
    
    naive_code_for_prompt = kernel_code_r1
    
    # =========================================================================
    # ROUND 2: Generate Optimized Implementation
    # =========================================================================
    print(f"\n{'#'*80}")
    print("# ROUND 2: Generate Optimized Implementation")
    print(f"{'#'*80}")
    
    print("\n📝 Building Round 2 prompt (optimization request)...")
    prompt_r2 = build_round2_prompt(naive_code_for_prompt)
    print(f"📊 Prompt length: {len(prompt_r2)} characters")
    
    print("\n⚡ Generating optimized kernel code...")
    start_time_r2 = time.time()
    sampling_params_2 = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=MAX_TOKENS,
    )
    outputs_r2 = llm.generate([prompt_r2], sampling_params_2)
    generation_time_r2 = time.time() - start_time_r2
    response_text_r2 = outputs_r2[0].outputs[0].text
    
    print(f"⏱️  Round 2 generation time: {generation_time_r2:.2f} seconds")
    
    with open(FULL_RESPONSE_R2, 'w', encoding='utf-8') as f:
        f.write(response_text_r2)
    print(f"💾 Round 2 full response saved: {FULL_RESPONSE_R2}")
    
    status_r2 = report_generation_status(outputs_r2[0], MAX_TOKENS, round_num=2)
    
    print("\n🔍 Parsing Round 2 response...")
    kernel_code_r2 = extract_kernel_layerforward(response_text_r2)
    
    if not kernel_code_r2:
        print("❌ Failed to extract kernel from Round 2.")
    else:
        print("\n💾 Writing Round 2 output...")
        write_kernel_header('kernel_layerforward', kernel_code_r2, OUTPUT_PATH_ROUND2)
        
        print("\n📋 Deploying Round 2 implementation...")
        copy_header_to_testcase(OUTPUT_PATH_ROUND2, TESTCASE_DIR, HEADER_NAME)
        
        compile_success_r2 = compile_testcase(TESTCASE_DIR)
        if compile_success_r2:
            print("✅ Round 2 compilation successful!")
        else:
            print("⚠️  Round 2 compilation failed.")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print(f"\n{'='*80}")
    print("🎉 Pipeline Complete!")
    print(f"{'='*80}")
    
    print(f"\n📁 Output Files:")
    print(f"   Round 1 (Naive):  {OUTPUT_PATH_ROUND1}")
    print(f"   Round 2 (Optimized): {OUTPUT_PATH_ROUND2}")
    print(f"   Round 1 Response: {FULL_RESPONSE_R1}")
    print(f"   Round 2 Response: {FULL_RESPONSE_R2}")
    
    print(f"\n📊 Generation Status:")
    print(f"   Round 1: {status_r1}")
    print(f"   Round 2: {status_r2}")
    
    print(f"\n🔨 Compilation:")
    print(f"   Round 1: {'✅ Success' if compile_success_r1 else '❌ Failed'}")
    if kernel_code_r2:
        print(f"   Round 2: {'✅ Success' if compile_success_r2 else '❌ Failed'}")
    
    print(f"\n🧪 Manual Benchmarking Instructions:")
    print(f"   1. Navigate to testcase directory:")
    print(f"      cd {TESTCASE_DIR}")
    print(f"   2. Run naive version benchmark:")
    print(f"      ncu ./main 4096 2>&1 | grep duration")
    print(f"   3. Copy optimized header and recompile:")
    print(f"      cp {OUTPUT_PATH_ROUND2} {HEADER_NAME}")
    print(f"      make clean && make")
    print(f"   4. Run optimized version benchmark:")
    print(f"      ncu ./main 4096 2>&1 | grep duration")
    print(f"   5. Compare durations - lower is better! 🎯")
    
    print(f"\n💡 Tips:")
    print(f"   - If Round 2 failed extraction, check {FULL_RESPONSE_R2}")
    print(f"   - If compilation fails, verify kernel signature matches exactly")
    print(f"   - For better optimization, you can add more rounds with feedback")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()
