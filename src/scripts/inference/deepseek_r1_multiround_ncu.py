#!/usr/bin/env python3
# ============================================================================
# deepseek_r1_multiround_ncu.py
# Function: Multi-round CUDA kernel generation pipeline WITH automatic ncu profiling
#           Round 1: Naive implementation → compile → ncu profile → capture report
#           Round 2: Optimization prompt + ncu report → compile → (optional: ncu again)
# ============================================================================

import os
import re
import time
import shutil
import subprocess
import json
from pathlib import Path
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
NCU_REPORT_R1 = os.path.join(BASE_DIR, "ncu_report_round1.txt")
NCU_REPORT_R2 = os.path.join(BASE_DIR, "ncu_report_round2.txt")  # Optional

# ============================================================================
# NCU Profiling Functions
# ============================================================================
def run_ncu_profiler(
    binary_path: str,
    args: list = None,
    timeout: int = 300,
    output_file: str = None
) -> dict:
    """
    Run ncu profiler with sudo and capture output.
    
    Args:
        binary_path: Path to the binary to profile (relative or absolute)
        args: Additional arguments to pass to the binary (e.g., [4096])
        timeout: Timeout in seconds
        output_file: Optional file path to save raw output
    
    Returns:
        dict with profiling results
    """
    if args is None:
        args = []
    
    # Build command: sudo ncu <binary> <args>
    cmd = ["sudo", "ncu", "--set", "basic", binary_path] + [str(a) for a in args]
    
    print(f"🔬 Running ncu: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=BASE_DIR,  # Ensure we're in the right directory
            env={**os.environ, "NCU_INJECT_RES": "1"}  # Help with some ncu issues
        )
        
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        combined = stdout + stderr
        
        # Save raw output if requested
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(combined)
            print(f"💾 NCU output saved: {output_file}")
        
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "combined": combined,
            "duration_extracted": extract_duration_from_ncu(combined)
        }
        
    except subprocess.TimeoutExpired:
        print(f"⚠️  NCU timed out after {timeout}s")
        return {"success": False, "error": f"timeout_{timeout}s", "combined": ""}
    except FileNotFoundError:
        print("❌ ERROR: 'ncu' or 'sudo' not found in PATH")
        return {"success": False, "error": "command_not_found", "combined": ""}
    except PermissionError:
        print("❌ ERROR: Permission denied for sudo/ncu")
        return {"success": False, "error": "permission_denied", "combined": ""}
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {e}")
        return {"success": False, "error": str(e), "combined": ""}


def extract_duration_from_ncu(ncu_output: str) -> dict:
    """
    Extract key duration/metric values from ncu output.
    Returns dict with parsed metrics for prompt injection.
    """
    metrics = {}
    
    # Extract kernel durations (in microseconds)
    duration_pattern = r'Duration\s+us\s+([\d,]+\.?\d*)'
    durations = re.findall(duration_pattern, ncu_output)
    if durations:
        metrics["kernel_durations_us"] = [float(d.replace(',', '')) for d in durations]
    
    # Extract throughput metrics
    throughput_patterns = {
        "memory_throughput_pct": r'Memory Throughput\s+%\s+([\d.]+)',
        "dram_throughput_pct": r'DRAM Throughput\s+%\s+([\d.]+)',
        "sm_throughput_pct": r'Compute \(SM\) Throughput\s+%\s+([\d.]+)',
        "l1_cache_throughput_pct": r'L1/TEX Cache Throughput\s+%\s+([\d.]+)',
        "l2_cache_throughput_pct": r'L2 Cache Throughput\s+%\s+([\d.]+)',
    }
    
    for key, pattern in throughput_patterns.items():
        match = re.search(pattern, ncu_output)
        if match:
            metrics[key] = float(match.group(1))
    
    # Extract occupancy
    occ_match = re.search(r'Achieved Occupancy\s+%\s+([\d.]+)', ncu_output)
    if occ_match:
        metrics["achieved_occupancy_pct"] = float(occ_match.group(1))
    
    # Extract optimization suggestions
    opt_suggestions = re.findall(r'OPT\s+Est\. (?:Local )?Speedup:\s+([\d.]+)%', ncu_output)
    if opt_suggestions:
        metrics["estimated_speedups_pct"] = [float(s) for s in opt_suggestions]
    
    # Extract waves per SM (grid sizing issue indicator)
    waves_match = re.search(r'Waves Per SM\s+([\d.]+)', ncu_output)
    if waves_match:
        metrics["waves_per_sm"] = float(waves_match.group(1))
    
    return metrics


def format_ncu_report_for_prompt(ncu_result: dict, include_raw: bool = True) -> str:
    """
    Format ncu profiling results into a readable prompt section.
    """
    if not ncu_result.get("success") or not ncu_result.get("combined"):
        return "⚠️  NCU profiling failed or produced no output."
    
    metrics = ncu_result.get("duration_extracted", {})
    
    report = []
    report.append("# 📊 NCU Profiling Report (Round 1 Baseline)")
    report.append("")
    
    # Key metrics summary
    if metrics:
        report.append("## Extracted Metrics")
        report.append("```json")
        report.append(json.dumps(metrics, indent=2))
        report.append("```")
        report.append("")
    
    # Duration summary
    if "kernel_durations_us" in metrics:
        durations = metrics["kernel_durations_us"]
        report.append(f"## Kernel Execution Times")
        for i, d in enumerate(durations, 1):
            report.append(f"- Kernel {i}: **{d:.2f} μs**")
        report.append("")
    
    # Optimization hints
    if "estimated_speedups_pct" in metrics:
        speedups = metrics["estimated_speedups_pct"]
        report.append("## 🎯 Optimization Opportunities")
        report.append(f"NCU suggests potential speedups of: {[f'{s:.1f}%' for s in speedups]}")
        report.append("")
    
    # Occupancy warning
    if metrics.get("achieved_occupancy_pct", 100) < 80:
        report.append("⚠️  **Low Occupancy Warning**: Achieved occupancy is below 80%.")
        report.append("   Consider: increasing block size, reducing register usage, or improving warp-level parallelism.")
        report.append("")
    
    # Grid sizing warning
    if metrics.get("waves_per_sm", 1) < 1.0:
        report.append("⚠️  **Grid Sizing Warning**: Less than 1 full wave across SMs.")
        report.append("   Consider: increasing grid size or using dynamic parallelism for better resource utilization.")
        report.append("")
    
    # Raw output (optional, for detailed analysis)
    if include_raw:
        report.append("## 📄 Raw NCU Output (Excerpt - First 2000 chars)")
        report.append("```")
        raw_excerpt = ncu_result["combined"][:2000]
        # Clean up ANSI codes if present
        raw_excerpt = re.sub(r'\x1b\[[0-9;]*m', '', raw_excerpt)
        report.append(raw_excerpt)
        report.append("```")
        report.append("")
        report.append("*(Full report saved to file for detailed analysis)*")
    
    return "\n".join(report)


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
# Build Round 1 Prompt (Naive Implementation)
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
# Build Round 2 Prompt (Optimization + NCU Report)
# ============================================================================
def build_round2_prompt(naive_code_context: str, ncu_report: str):
    """Build prompt for optimized implementation based on naive version + profiling feedback"""
    return f"""# Role
You are an expert CUDA Developer specializing in performance optimization.

# Task
Optimize the CUDA kernel below based on the NCU profiling report.

# 📊 Profiling Feedback (Round 1 Baseline)
{ncu_report}

# 🔍 Key Optimization Guidelines Based on Report:
- If **low occupancy**: reduce register pressure, increase block size, or improve warp utilization
- If **low memory throughput**: improve memory coalescing, use shared memory more effectively
- If **grid too small**: increase grid dimensions or use dynamic parallelism
- If **load imbalance**: ensure work is evenly distributed across warps/threads
- If **cache underutilized**: improve data locality and reuse patterns

# Original Naive Implementation (Reference)
```cuda
{naive_code_context}
```

# Target Kernel Specification (MUST MATCH EXACTLY)
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
- Return ONLY the optimized CUDA code inside ```cuda ... ``` blocks.
- Ensure the signature matches the Target Kernel Specification exactly.
- Ensure the code compiles with nvcc.
- Add brief comments explaining key optimizations made.
<think>
</think>
"""


# ============================================================================
# Main Pipeline
# ============================================================================
def main():
    print("=" * 80)
    print("🚀 DeepSeek-R1 Multi-Round CUDA Kernel Generation Pipeline")
    print("   WITH Automatic NCU Profiling Feedback Loop")
    print("   Round 1: Naive → Compile → NCU Profile → Capture Report")
    print("   Round 2: Optimize + NCU Report → Compile → (Optional: NCU)")
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
        print("❌ Round 1 compilation failed. Cannot profile. Exiting.")
        return
    
    # =========================================================================
    # 🔬 NCU PROFILING: Round 1 Baseline
    # =========================================================================
    print(f"\n{'🔬'*40}")
    print("🔬 Running NCU Profiling on Round 1 Implementation")
    print(f"{'🔬'*40}")
    
    ncu_result_r1 = run_ncu_profiler(
        binary_path="./testcases/backprop-cuda/main",
        args=[4096],
        timeout=300,
        output_file=NCU_REPORT_R1
    )
    
    if not ncu_result_r1["success"]:
        print(f"⚠️  NCU profiling failed: {ncu_result_r1.get('error', 'unknown error')}")
        print("   Continuing to Round 2 without profiling feedback...")
        ncu_report_for_prompt = "⚠️  NCU profiling failed or was skipped."
    else:
        print(f"✅ NCU profiling completed!")
        if ncu_result_r1.get("duration_extracted", {}).get("kernel_durations_us"):
            durations = ncu_result_r1["duration_extracted"]["kernel_durations_us"]
            print(f"📊 Kernel durations: {[f'{d:.2f}μs' for d in durations]}")
        
        # Format report for LLM prompt
        ncu_report_for_prompt = format_ncu_report_for_prompt(ncu_result_r1, include_raw=True)
    
    # =========================================================================
    # ROUND 2: Generate Optimized Implementation (with NCU feedback)
    # =========================================================================
    print(f"\n{'#'*80}")
    print("# ROUND 2: Generate Optimized Implementation + NCU Feedback")
    print(f"{'#'*80}")
    
    print("\n📝 Building Round 2 prompt (optimization + profiling report)...")
    prompt_r2 = build_round2_prompt(kernel_code_r1, ncu_report_for_prompt)
    print(f"📊 Prompt length: {len(prompt_r2)} characters")
    
    print("\n⚡ Generating optimized kernel code...")
    start_time_r2 = time.time()
    sampling_params_2 = SamplingParams(
        temperature=0.7,  # Slightly higher for creative optimization
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
        compile_success_r2 = False
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
    # (Optional) NCU Profiling: Round 2 Optimized Version
    # =========================================================================
    if compile_success_r2:
        print(f"\n{'🔬'*40}")
        print("🔬 [OPTIONAL] Running NCU Profiling on Round 2 Optimized Version")
        print(f"{'🔬'*40}")
        
        run_ncu = input("\n🤔 Run NCU on optimized version? (y/N): ").strip().lower()
        if run_ncu == 'y':
            ncu_result_r2 = run_ncu_profiler(
                binary_path="./testcases/backprop-cuda/main",
                args=[4096],
                timeout=300,
                output_file=NCU_REPORT_R2
            )
            
            if ncu_result_r2["success"]:
                print(f"✅ Round 2 NCU profiling completed!")
                if ncu_result_r2.get("duration_extracted", {}).get("kernel_durations_us"):
                    durations_r2 = ncu_result_r2["duration_extracted"]["kernel_durations_us"]
                    print(f"📊 Optimized kernel durations: {[f'{d:.2f}μs' for d in durations_r2]}")
                    
                    # Compare with Round 1
                    if ncu_result_r1.get("duration_extracted", {}).get("kernel_durations_us"):
                        durations_r1 = ncu_result_r1["duration_extracted"]["kernel_durations_us"]
                        if len(durations_r1) == len(durations_r2):
                            print(f"\n📈 Performance Comparison:")
                            for i, (r1, r2) in enumerate(zip(durations_r1, durations_r2), 1):
                                improvement = ((r1 - r2) / r1) * 100 if r1 > 0 else 0
                                print(f"   Kernel {i}: {r1:.2f}μs → {r2:.2f}μs ({improvement:+.1f}% change)")
            else:
                print(f"⚠️  Round 2 NCU profiling failed: {ncu_result_r2.get('error', 'unknown')}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print(f"\n{'='*80}")
    print("🎉 Pipeline Complete!")
    print(f"{'='*80}")
    
    print(f"\n📁 Output Files:")
    print(f"   Round 1 (Naive):     {OUTPUT_PATH_ROUND1}")
    print(f"   Round 2 (Optimized): {OUTPUT_PATH_ROUND2}")
    print(f"   Round 1 Response:    {FULL_RESPONSE_R1}")
    print(f"   Round 2 Response:    {FULL_RESPONSE_R2}")
    print(f"   Round 1 NCU Report:  {NCU_REPORT_R1}")
    if compile_success_r2:
        print(f"   Round 2 NCU Report:  {NCU_REPORT_R2} *(if profiled)*")
    
    print(f"\n📊 Generation Status:")
    print(f"   Round 1: {status_r1}")
    print(f"   Round 2: {status_r2}")
    
    print(f"\n🔨 Compilation:")
    print(f"   Round 1: {'✅ Success' if compile_success_r1 else '❌ Failed'}")
    if kernel_code_r2:
        print(f"   Round 2: {'✅ Success' if compile_success_r2 else '❌ Failed'}")
    
    print(f"\n🔬 Profiling:")
    print(f"   Round 1 NCU: {'✅ Completed' if ncu_result_r1.get('success') else '❌ Failed/Skipped'}")
    if compile_success_r2:
        print(f"   Round 2 NCU: {'✅ Completed (manual)' if 'ncu_result_r2' in locals() and ncu_result_r2.get('success') else '⏭️  Skipped'}")
    
    print(f"\n💡 Next Steps:")
    print(f"   1. Review generated kernels in output files")
    print(f"   2. Check NCU reports for optimization insights")
    print(f"   3. Iterate with more rounds if needed (add Round 3+ with same pattern)")
    print(f"   4. For production: add error recovery, config files, and logging")
    
    print(f"\n{'='*80}")
    
    return {
        "round1_kernel": kernel_code_r1,
        "round2_kernel": kernel_code_r2,
        "ncu_r1": ncu_result_r1,
        "ncu_r2": locals().get('ncu_result_r2'),
        "compile_r1": compile_success_r1,
        "compile_r2": compile_success_r2
    }


if __name__ == "__main__":
    results = main()
