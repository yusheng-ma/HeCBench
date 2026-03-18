#!/usr/bin/env python3
# ============================================================================
# deepseek_r1_multiround_ncu.py
# Multi-round CUDA Kernel Generation with Stateful Hybrid Retry + Self-Reflection
# ============================================================================
# Features:
#   - Stateful retry memory (compact history, not full conversation)
#   - Hybrid strategy: simple retries → informed retries with error feedback
#   - Self-reflection: LLM critiques its own output before retrying
#   - NCU profiling integration with performance-based retry
#   - Exponential backoff with jitter
# ============================================================================

import os, re, time, shutil, subprocess, json, random
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from vllm import LLM, SamplingParams

# ============================================================================
# 🎛️ CONFIGURATION (Modify these values)
# ============================================================================

# Paths
BASE_DIR = "."
TESTCASE_DIR = os.path.join(BASE_DIR, "testcases/backprop-cuda")
HEADER_NAME = "bpnn_layerforward.h"

# Output files
OUTPUT_PATH_ROUND1 = os.path.join(BASE_DIR, f"bpnn_layerforward_r1_{HEADER_NAME}")
OUTPUT_PATH_ROUND2 = os.path.join(BASE_DIR, f"bpnn_layerforward_r2_{HEADER_NAME}")
FULL_RESPONSE_R1 = os.path.join(BASE_DIR, "deepseek_r1_round1_full_response.txt")
FULL_RESPONSE_R2 = os.path.join(BASE_DIR, "deepseek_r1_round2_full_response.txt")
NCU_REPORT_R1 = os.path.join(BASE_DIR, "ncu_report_round1.txt")
NCU_REPORT_R2 = os.path.join(BASE_DIR, "ncu_report_round2.txt")

# Retry Configuration
MAX_RETRIES = 3                    # 32B 模型更聰明，3 次通常足夠，節省時間 (原為 5)
RETRY_DELAY_BASE = 2               
RETRY_DELAY_JITTER = 0.2           

# Strategy Configuration
CONVERSATIONAL_DEPTH = 2           
SELF_REFLECT_AFTER_ATTEMPT = 1     # 32B 模型第一次失敗後就能給出高質量反思 (原為 2)
PERFORMANCE_IMPROVEMENT_THRESHOLD = 0.05  

# LLM Configuration
# 👇 修改點 1: 模型名稱改為 32B
LLM_MODEL = "Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ" 
LLM_MAX_MODEL_LEN = 4096
LLM_GPU_MEMORY_UTIL = 0.9         # 4090 顯存較大，可稍微調高利用率 (原為 0.80)
LLM_TEMPERATURE_BASE = 0.5         # 32B 模型更穩定，降低溫度以減少隨機錯誤 (原為 0.6)
LLM_TEMPERATURE_MAX = 0.7          # 最高溫度也相應降低 (原為 0.9)
LLM_TEMPERATURE_INCREMENT = 0.05

# ============================================================================
# 📦 Data Classes for State Management
# ============================================================================

@dataclass
class AttemptRecord:
    """Compact record of a single generation attempt"""
    attempt_num: int
    error_type: Optional[str] = None          # "extract", "compile", "performance", "exception"
    error_detail: Optional[str] = None        # Truncated error message
    code_hint: Optional[str] = None           # One-line summary of code approach
    feedback: Optional[str] = None            # Key feedback for next attempt
    duration_us: Optional[float] = None       # Kernel duration if profiled
    
    def to_summary(self) -> str:
        """Convert to compact string for prompt injection"""
        lines = [f"  • Attempt #{self.attempt_num}: {self.error_type or 'success'}"]
        if self.code_hint:
            lines.append(f"    - Approach: {self.code_hint}")
        if self.feedback:
            lines.append(f"    - Feedback: {self.feedback[:150]}")
        if self.duration_us:
            lines.append(f"    - Duration: {self.duration_us:.2f}μs")
        return "\n".join(lines)


@dataclass
class RetryMemory:
    """Stateful memory tracker for informed retries"""
    max_history: int = CONVERSATIONAL_DEPTH
    attempts: List[AttemptRecord] = field(default_factory=list)
    round_num: int = 1
    best_duration_us: Optional[float] = None
    best_kernel_code: Optional[str] = None
    
    def add_attempt(self, code: Optional[str], error_type: Optional[str], 
                    error_detail: Optional[str], feedback: Optional[str] = None,
                    duration_us: Optional[float] = None):
        """Record an attempt with compact metadata"""
        code_hint = self._extract_code_hint(code) if code else None
        
        record = AttemptRecord(
            attempt_num=len(self.attempts) + 1,
            error_type=error_type,
            error_detail=error_detail[:200] if error_detail else None,
            code_hint=code_hint,
            feedback=feedback,
            duration_us=duration_us
        )
        self.attempts.append(record)
        
        # Track best result
        if duration_us and (self.best_duration_us is None or duration_us < self.best_duration_us):
            self.best_duration_us = duration_us
            self.best_kernel_code = code
        
        # Trim history to max_depth
        if len(self.attempts) > self.max_history:
            self.attempts.pop(0)
    
    def _extract_code_hint(self, code: str) -> Optional[str]:
        """Extract one-line summary of optimization approach"""
        code_lower = code.lower()
        hints = []
        if "shared" in code_lower: hints.append("shared memory")
        if "reduction" in code_lower or "tree" in code_lower: hints.append("tree reduction")
        if "coalesc" in code_lower: hints.append("memory coalescing")
        if "occupancy" in code_lower: hints.append("occupancy optimization")
        if "register" in code_lower: hints.append("register tuning")
        return ", ".join(hints) if hints else "standard implementation"
    
    def build_context_suffix(self) -> str:
        """Build compact revision history for prompt injection"""
        if not self.attempts:
            return ""
        
        lines = [
            "\n\n# 🔄 Revision History (Recent Attempts):",
            f"  Round: {self.round_num}, Total attempts so far: {len(self.attempts)}"
        ]
        
        for att in self.attempts:
            lines.append(att.to_summary())
        
        if self.best_duration_us:
            lines.append(f"\n  🏆 Best duration so far: {self.best_duration_us:.2f}μs")
        
        return "\n".join(lines)
    
    def get_failure_pattern(self) -> Optional[str]:
        """Detect recurring failure patterns"""
        if len(self.attempts) < 2:
            return None
        
        error_types = [a.error_type for a in self.attempts if a.error_type]
        if len(error_types) >= 2 and len(set(error_types)) == 1:
            return f"Recurring issue: {error_types[0]} (attempted {len(error_types)} times)"
        return None


# ============================================================================
# 🔬 NCU Profiling Functions
# ============================================================================

def run_ncu_profiler(binary_path: str, args: List[str] = None, 
                     timeout: int = 300, output_file: Optional[str] = None) -> Dict[str, Any]:
    """Run ncu profiler with sudo and capture output"""
    if args is None:
        args = []
    
    cmd = ["ncu", "--set", "basic", binary_path] + [str(a) for a in args]
    print(f"🔬 Running ncu: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=BASE_DIR,
            env={**os.environ, "NCU_INJECT_RES": "1"}
        )
        combined = (result.stdout or "") + (result.stderr or "")
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(combined)
            print(f"💾 NCU output saved: {output_file}")
        
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "combined": combined,
            "duration_extracted": extract_duration_from_ncu(combined)
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"timeout_{timeout}s", "combined": ""}
    except FileNotFoundError:
        return {"success": False, "error": "command_not_found", "combined": ""}
    except PermissionError:
        return {"success": False, "error": "permission_denied", "combined": ""}
    except Exception as e:
        return {"success": False, "error": str(e), "combined": ""}


def extract_duration_from_ncu(ncu_output: str) -> Dict[str, Any]:
    """Extract key metrics from ncu output"""
    metrics = {}
    
    # Kernel durations
    durations = re.findall(r'Duration\s+us\s+([\d,]+\.?\d*)', ncu_output)
    if durations:
        metrics["kernel_durations_us"] = [float(d.replace(',', '')) for d in durations]
    
    # Throughput metrics
    for key, pattern in {
        "memory_throughput_pct": r'Memory Throughput\s+%\s+([\d.]+)',
        "sm_throughput_pct": r'Compute \(SM\) Throughput\s+%\s+([\d.]+)',
        "achieved_occupancy_pct": r'Achieved Occupancy\s+%\s+([\d.]+)',
    }.items():
        match = re.search(pattern, ncu_output)
        if match:
            metrics[key] = float(match.group(1))
    
    # Optimization suggestions
    speedups = re.findall(r'OPT\s+Est\. (?:Local )?Speedup:\s+([\d.]+)%', ncu_output)
    if speedups:
        metrics["estimated_speedups_pct"] = [float(s) for s in speedups]
    
    return metrics


def format_ncu_report_for_prompt(ncu_result: Dict[str, Any], include_raw: bool = True) -> str:
    """Format ncu results into readable prompt section"""
    if not ncu_result.get("success") or not ncu_result.get("combined"):
        return "⚠️  NCU profiling failed or produced no output."
    
    metrics = ncu_result.get("duration_extracted", {})
    lines = ["# 📊 NCU Profiling Report", ""]
    
    if metrics:
        lines.extend(["## Extracted Metrics", "```json", json.dumps(metrics, indent=2), "```", ""])
    
    if "kernel_durations_us" in metrics:
        lines.append("## Kernel Execution Times")
        for i, d in enumerate(metrics["kernel_durations_us"], 1):
            lines.append(f"- Kernel {i}: **{d:.2f} μs**")
        lines.append("")
    
    if metrics.get("achieved_occupancy_pct", 100) < 80:
        lines.extend([
            "⚠️  **Low Occupancy Warning**: <80%",
            "   → Try: increase block size, reduce register usage, improve warp parallelism", ""
        ])
    
    if metrics.get("memory_throughput_pct", 100) < 50:
        lines.extend([
            "⚠️  **Low Memory Throughput**: <50%",
            "   → Try: improve memory coalescing, use shared memory more effectively", ""
        ])
    
    if include_raw:
        lines.extend([
            "## 📄 Raw NCU Output (Excerpt - First 2000 chars)", "```",
            re.sub(r'\x1b\[[0-9;]*m', '', ncu_result["combined"][:2000]), "```"
        ])
    
    return "\n".join(lines)


def analyze_ncu_bottlenecks(ncu_result: Dict[str, Any]) -> List[str]:
    """Extract specific optimization suggestions from NCU output"""
    suggestions = []
    metrics = ncu_result.get("duration_extracted", {})
    
    if metrics.get("achieved_occupancy_pct", 100) < 75:
        suggestions.append("• Low occupancy: reduce register pressure or increase block size")
    
    if metrics.get("memory_throughput_pct", 100) < 50:
        suggestions.append("• Low memory throughput: improve coalescing, use shared memory")
    
    if metrics.get("sm_throughput_pct", 100) < 60:
        suggestions.append("• Low SM throughput: reduce branch divergence, optimize arithmetic intensity")
    
    if not suggestions:
        suggestions.append("• Try alternative reduction strategies or memory access patterns")
    
    return suggestions


# ============================================================================
# 📝 Kernel Extraction & File Operations
# ============================================================================

def extract_kernel_layerforward(response_text: str) -> Optional[str]:
    """Extract kernel_layerforward function from LLM response"""
    kernel_name = 'kernel_layerforward'
    
    # Method 1: Brace matching
    match = re.search(r'__global__\s+void\s+' + kernel_name + r'\s*\([^)]*\)', response_text)
    if match:
        start = match.start()
        brace = response_text.find('{', start)
        if brace != -1:
            count, end = 1, brace + 1
            while end < len(response_text) and count > 0:
                if response_text[end] == '{': count += 1
                elif response_text[end] == '}': count -= 1
                end += 1
            if count == 0:
                print(f"✅ Extracted: {kernel_name}")
                return response_text[start:end]
    
    # Method 2: Code block
    block = re.search(r'```(?:cuda|cpp|c)?\s*([\s\S]*?' + kernel_name + r'[\s\S]*?)```', response_text)
    if block:
        print(f"✅ Extracted from code block: {kernel_name}")
        return block.group(1).strip()
    
    print(f"❌ Failed to extract: {kernel_name}")
    return None


def write_kernel_header(kernel_name: str, kernel_code: str, output_path: str) -> Tuple[bool, str]:
    """Write kernel code to header file"""
    try:
        guard = os.path.basename(output_path).replace('.', '_').replace('-', '_').upper()
        content = f"""// ============================================================================
// Auto-generated CUDA Kernel by DeepSeek-R1
// Kernel: {kernel_name}
// ============================================================================

#ifndef {guard}
#define {guard}

#include <cuda.h>

{kernel_code}

#endif // {guard}
"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"📁 Written: {output_path}")
        return True, ""
    except Exception as e:
        return False, str(e)


def copy_header_to_testcase(src: str, dst_dir: str, name: str) -> Tuple[bool, str]:
    """Copy generated header to testcase directory"""
    try:
        dst = os.path.join(dst_dir, name)
        shutil.copy2(src, dst)
        print(f"📋 Copied: {src} → {dst}")
        return True, ""
    except Exception as e:
        return False, str(e)


def compile_testcase(testcase_dir: str) -> Tuple[bool, str]:
    """Run make clean && make in testcase directory"""
    print(f"🔨 Compiling in {testcase_dir}...")
    
    subprocess.run(["make", "clean"], cwd=testcase_dir, capture_output=True)
    result = subprocess.run(["make"], cwd=testcase_dir, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Compilation successful!")
        return True, ""
    else:
        error_msg = result.stderr[:500] if result.stderr else "Unknown compilation error"
        print(f"❌ Compilation failed! Error: {error_msg[:200]}")
        return False, error_msg


# ============================================================================
# 🧠 Self-Reflection Module
# ============================================================================

def self_reflect_on_error(llm: LLM, code: str, error_type: str, 
                          error_detail: str, sampling_params: SamplingParams) -> str:
    """Ask LLM to critique its own output before retrying"""
    
    critique_prompts = {
        "extract_failed": """You wrote CUDA code but the kernel function couldn't be extracted.
Possible issues:
- Missing ```cuda ... ``` code blocks
- Function name doesn't match 'kernel_layerforward'
- Signature doesn't match specification

Analyze your code and identify the most likely formatting issue.""",
        
        "compile_failed": f"""Your CUDA kernel failed to compile:
```
{error_detail[:400]}
```
Identify the most likely cause:
- Syntax error (missing semicolon, bracket mismatch)
- Type mismatch
- Invalid CUDA builtin usage
- Missing includes or defines""",
        
        "performance_failed": f"""Your kernel compiled but performance is insufficient.
NCU feedback:
{error_detail[:400]}
Suggest specific optimizations to try next.""",
        
        "exception": """An unexpected error occurred during generation or execution.
Suggest how to make the code more robust and simpler."""
    }
    
    base_critique = critique_prompts.get(error_type, "Analyze what went wrong and suggest fixes.")
    
    critique_prompt = f"""You previously generated this CUDA kernel:
```cuda
{code[:1500] if code else '<no code extracted>'}
```

{base_critique}

Output ONLY a brief diagnosis (1-2 sentences) and concrete fix suggestion (1-2 sentences).
No code, no explanations."""

    try:
        critique_params = SamplingParams(temperature=0.3, top_p=0.9, max_tokens=300)
        outputs = llm.generate([critique_prompt], critique_params)
        critique = outputs[0].outputs[0].text.strip()
        print(f"💭 Self-reflection: {critique[:200]}")
        return critique
    except Exception as e:
        print(f"⚠️  Self-reflection failed: {e}")
        return "Unable to self-diagnose. Try a different approach."


# ============================================================================
# 📋 Prompt Builders
# ============================================================================

def build_round1_prompt() -> str:
    """Build prompt for naive/initial implementation"""
    return """<|begin_of_sentence|><|User|># Role
You are an expert CUDA Developer.

# Task
Implement a CUDA kernel for BPNN forward pass.

# Coding Style (FOLLOW EXACTLY)
Use `__global__`, `__restrict__`, and proper CUDA conventions.

Example:
```cuda
__global__ void example_kernel(const float* __restrict__ in, float* __restrict__ out, const int n) {
  int idx = threadIdx.x;
  __shared__ float shared_data[16];
  shared_data[idx] = in[idx];
  __syncthreads();
  out[idx] = shared_data[idx] * 2.0f;
}
```

# Target Kernel Specification
Implement `kernel_layerforward` with EXACT signature below.
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
1. **Load Input**: Index `HEIGHT * by + ty + 1` → `input_node[ty]` (bounds check)
2. **Load Weights**: Index `(hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1)` → `weight_matrix[ty * WIDTH + tx]` (bounds check)
3. **Sync**: `__syncthreads()`
4. **Multiply**: `weight_matrix[ty * WIDTH + tx] *= input_node[ty]`
5. **Reduce**: Sum across `tx` using tree reduction. Use `__syncthreads()` between steps.
6. **Store Result**: If `tx == 0`, write to `hidden_partial_sum[by * hid + ty]`

# Output Requirement
- Return ONLY the CUDA code inside ```cuda ... ``` blocks
- Ensure signature matches Target Kernel Specification exactly
- Ensure code compiles with nvcc<|Assistant|>"""


def build_round2_prompt(naive_code: str, ncu_report: str) -> str:
    """Build prompt for optimized implementation"""
    return f"""<|begin_of_sentence|><|User|># Role
You are an expert CUDA Developer specializing in performance optimization.

# Task
Optimize the CUDA kernel below based on the NCU profiling report.

# 📊 Profiling Feedback (Round 1 Baseline)
{ncu_report}

# 🔍 Key Optimization Guidelines
- **Low occupancy**: reduce register pressure, increase block size, improve warp utilization
- **Low memory throughput**: improve memory coalescing, use shared memory effectively
- **Grid too small**: increase grid dimensions or use dynamic parallelism
- **Load imbalance**: ensure even work distribution across warps/threads
- **Cache underutilized**: improve data locality and reuse patterns

# Original Naive Implementation (Reference)
```cuda
{naive_code}
```

# Target Kernel Specification (MUST MATCH EXACTLY)
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
- Return ONLY the optimized CUDA code inside ```cuda ... ``` blocks
- Ensure signature matches exactly
- Ensure code compiles with nvcc
- Add brief comments explaining key optimizations made<|Assistant|>"""


def enhance_prompt_with_retry_context(base_prompt: str, memory: RetryMemory, 
                                       self_critique: Optional[str] = None) -> str:
    """Add retry context and self-reflection to base prompt"""
    enhanced = base_prompt
    
    # Add revision history
    history_suffix = memory.build_context_suffix()
    if history_suffix:
        enhanced += history_suffix
    
    # Add self-critique if available
    if self_critique:
        enhanced += f"\n\n# 💡 Self-Diagnosis from Previous Attempt:\n{self_critique}"
    
    # Add failure pattern warning if detected
    pattern = memory.get_failure_pattern()
    if pattern:
        enhanced += f"\n\n# ⚠️  Recurring Issue Detected:\n{pattern}\nTry a fundamentally different approach."
    
    return enhanced


# ============================================================================
# 🔄 Retry Helper Functions
# ============================================================================

def exponential_backoff(attempt: int) -> float:
    """Calculate delay with exponential backoff and jitter"""
    base = RETRY_DELAY_BASE
    jitter = random.uniform(-RETRY_DELAY_JITTER * base, RETRY_DELAY_JITTER * base)
    return max(0.5, base + jitter)


def check_performance_improvement(r1_durs: List[float], r2_durs: List[float]) -> Tuple[bool, str]:
    """Check if R2 meets performance improvement threshold"""
    if not r1_durs or not r2_durs or len(r1_durs) != len(r2_durs):
        return False, "Data mismatch or missing"
    
    improvements = [(r1 - r2) / r1 * 100 for r1, r2 in zip(r1_durs, r2_durs) if r1 > 0]
    if not improvements:
        return False, "No valid comparison data"
    
    avg_imp = sum(improvements) / len(improvements)
    threshold_pct = PERFORMANCE_IMPROVEMENT_THRESHOLD * 100
    
    if avg_imp >= threshold_pct:
        return True, f"✅ Avg improvement: {avg_imp:+.1f}% (threshold: {threshold_pct:.0f}%)"
    else:
        return False, f"⚠️  Avg improvement: {avg_imp:+.1f}% < {threshold_pct:.0f}% threshold"


def get_sampling_params(attempt: int, base_temp: float = LLM_TEMPERATURE_BASE) -> SamplingParams:
    """Get sampling params with temperature scaling based on attempt number"""
    temp = min(LLM_TEMPERATURE_MAX, base_temp + attempt * LLM_TEMPERATURE_INCREMENT)
    return SamplingParams(temperature=temp, top_p=0.9, max_tokens=LLM_MAX_MODEL_LEN)


# ============================================================================
# 🎯 Round 1 Generation with Stateful Hybrid Retry
# ============================================================================

def generate_round1_with_retry(llm: LLM, max_retries: int = MAX_RETRIES) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    """Generate Round 1 kernel with stateful hybrid retry logic"""
    print(f"\n{'#'*80}")
    print(f"# ROUND 1: Naive Implementation (Max Retries: {max_retries})")
    print(f"{'#'*80}")
    
    base_prompt = build_round1_prompt()
    memory = RetryMemory(max_history=CONVERSATIONAL_DEPTH, round_num=1)
    self_critique = None
    
    for attempt in range(max_retries + 1):
        print(f"\n🔄 R1 Attempt {attempt+1}/{max_retries+1}")
        
        try:
            # Build prompt with retry context
            if attempt == 0:
                prompt = base_prompt
            else:
                prompt = enhance_prompt_with_retry_context(base_prompt, memory, self_critique)
            
            # Get sampling params with temperature scaling
            params = get_sampling_params(attempt)
            print(f"   Temperature: {params.temperature:.2f}")
            
            # Generate
            start_time = time.time()
            outputs = llm.generate([prompt], params)
            gen_time = time.time() - start_time
            print(f"   ⏱️  Generation time: {gen_time:.2f}s")
            
            text = outputs[0].outputs[0].text
            finish_reason = outputs[0].outputs[0].finish_reason
            
            # Save full response
            with open(FULL_RESPONSE_R1, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Extract kernel
            kernel = extract_kernel_layerforward(text)
            if not kernel:
                error_detail = "Failed to extract kernel_layerforward function"
                print(f"   ⚠️  Extract failed: {error_detail}")
                
                # Self-reflect if past threshold
                if attempt >= SELF_REFLECT_AFTER_ATTEMPT:
                    self_critique = self_reflect_on_error(llm, text, "extract_failed", "", params)
                
                memory.add_attempt(kernel, "extract_failed", error_detail, 
                                  feedback="Check code block formatting and function signature")
                
                if attempt < max_retries:
                    delay = exponential_backoff(attempt)
                    print(f"   ⏳  Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    continue
                else:
                    return None, None, {"status": "extract_failed", "attempts": attempt+1, "memory": memory}
            
            # Write header
            ok, err = write_kernel_header('kernel_layerforward', kernel, OUTPUT_PATH_ROUND1)
            if not ok:
                print(f"   ⚠️  Write failed: {err}")
                memory.add_attempt(kernel, "write_failed", err)
                if attempt < max_retries:
                    time.sleep(exponential_backoff(attempt))
                    continue
                return None, None, {"status": "write_failed", "attempts": attempt+1, "memory": memory}
            
            # Copy to testcase
            ok, err = copy_header_to_testcase(OUTPUT_PATH_ROUND1, TESTCASE_DIR, HEADER_NAME)
            if not ok:
                print(f"   ⚠️  Copy failed: {err}")
                memory.add_attempt(kernel, "copy_failed", err)
                if attempt < max_retries:
                    time.sleep(exponential_backoff(attempt))
                    continue
                return None, None, {"status": "copy_failed", "attempts": attempt+1, "memory": memory}
            
            # Compile
            ok, err = compile_testcase(TESTCASE_DIR)
            if not ok:
                print(f"   ⚠️  Compile failed")
                
                # Self-reflect if past threshold
                if attempt >= SELF_REFLECT_AFTER_ATTEMPT:
                    self_critique = self_reflect_on_error(llm, kernel, "compile_failed", err, params)
                
                memory.add_attempt(kernel, "compile_failed", err, 
                                  feedback="Check syntax, types, and CUDA builtins")
                
                if attempt < max_retries:
                    delay = exponential_backoff(attempt)
                    print(f"   ⏳  Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    continue
                else:
                    return None, None, {"status": "compile_failed", "attempts": attempt+1, "error": err, "memory": memory}
            
            # Success!
            print(f"   ✅ R1 Success on attempt {attempt+1}!")
            memory.add_attempt(kernel, None, None, feedback="Compilation successful")
            return kernel, text, {"status": "success", "attempts": attempt+1, "memory": memory, "gen_time": gen_time}
            
        except Exception as e:
            print(f"   ❌ Attempt {attempt+1} exception: {type(e).__name__}: {e}")
            
            if attempt >= SELF_REFLECT_AFTER_ATTEMPT:
                self_critique = self_reflect_on_error(llm, "", "exception", str(e), params)
            
            memory.add_attempt(None, "exception", str(e))
            
            if attempt < max_retries:
                time.sleep(exponential_backoff(attempt))
                continue
            return None, None, {"status": "exception", "attempts": attempt+1, "error": str(e), "memory": memory}
    
    return None, None, {"status": "max_retries_exceeded", "attempts": max_retries+1, "memory": memory}


# ============================================================================
# 🎯 Round 2 Generation with Performance-Based Retry
# ============================================================================

def generate_round2_with_retry(llm: LLM, naive_code: str, ncu_r1: Dict[str, Any],
                                max_retries: int = MAX_RETRIES) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    """Generate Round 2 optimized kernel with performance-based retry"""
    print(f"\n{'#'*80}")
    print(f"# ROUND 2: Optimized Implementation (Max Retries: {max_retries})")
    print(f"{'#'*80}")
    
    ncu_report = format_ncu_report_for_prompt(ncu_r1) if ncu_r1.get("success") else "⚠️  NCU profiling failed"
    base_prompt = build_round2_prompt(naive_code, ncu_report)
    memory = RetryMemory(max_history=CONVERSATIONAL_DEPTH, round_num=2)
    self_critique = None
    
    r1_durs = ncu_r1.get("duration_extracted", {}).get("kernel_durations_us", [])
    
    for attempt in range(max_retries + 1):
        print(f"\n🔄 R2 Attempt {attempt+1}/{max_retries+1}")
        
        try:
            # Build prompt with retry context
            if attempt == 0:
                prompt = base_prompt
            else:
                prompt = enhance_prompt_with_retry_context(base_prompt, memory, self_critique)
            
            # Get sampling params (slightly higher base temp for R2)
            params = get_sampling_params(attempt, base_temp=0.7)
            print(f"   Temperature: {params.temperature:.2f}")
            
            # Generate
            start_time = time.time()
            outputs = llm.generate([prompt], params)
            gen_time = time.time() - start_time
            print(f"   ⏱️  Generation time: {gen_time:.2f}s")
            
            text = outputs[0].outputs[0].text
            
            # Save full response
            with open(FULL_RESPONSE_R2, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Extract kernel
            kernel = extract_kernel_layerforward(text)
            if not kernel:
                error_detail = "Failed to extract kernel_layerforward function"
                print(f"   ⚠️  Extract failed: {error_detail}")
                
                if attempt >= SELF_REFLECT_AFTER_ATTEMPT:
                    self_critique = self_reflect_on_error(llm, text, "extract_failed", "", params)
                
                memory.add_attempt(kernel, "extract_failed", error_detail)
                
                if attempt < max_retries:
                    time.sleep(exponential_backoff(attempt))
                    continue
                return None, None, {"status": "extract_failed", "attempts": attempt+1, "memory": memory}
            
            # Write header
            ok, err = write_kernel_header('kernel_layerforward', kernel, OUTPUT_PATH_ROUND2)
            if not ok:
                memory.add_attempt(kernel, "write_failed", err)
                if attempt < max_retries:
                    time.sleep(exponential_backoff(attempt))
                    continue
                return None, None, {"status": "write_failed", "attempts": attempt+1, "memory": memory}
            
            # Copy to testcase
            ok, err = copy_header_to_testcase(OUTPUT_PATH_ROUND2, TESTCASE_DIR, HEADER_NAME)
            if not ok:
                memory.add_attempt(kernel, "copy_failed", err)
                if attempt < max_retries:
                    time.sleep(exponential_backoff(attempt))
                    continue
                return None, None, {"status": "copy_failed", "attempts": attempt+1, "memory": memory}
            
            # Compile
            ok, err = compile_testcase(TESTCASE_DIR)
            if not ok:
                print(f"   ⚠️  Compile failed")
                
                if attempt >= SELF_REFLECT_AFTER_ATTEMPT:
                    self_critique = self_reflect_on_error(llm, kernel, "compile_failed", err, params)
                
                memory.add_attempt(kernel, "compile_failed", err)
                
                if attempt < max_retries:
                    time.sleep(exponential_backoff(attempt))
                    continue
                return None, None, {"status": "compile_failed", "attempts": attempt+1, "memory": memory}
            
            # Profile for performance check (if R1 has durations)
            duration_us = None
            if r1_durs and attempt < max_retries:  # Don't profile on last attempt
                print(f"   🔬 Profiling for performance comparison...")
                ncu_r2 = run_ncu_profiler("./testcases/backprop-cuda/main", [4096], 300, NCU_REPORT_R2)
                
                if ncu_r2.get("success"):
                    r2_durs = ncu_r2.get("duration_extracted", {}).get("kernel_durations_us", [])
                    if r2_durs:
                        duration_us = r2_durs[0] if r2_durs else None
                        improved, msg = check_performance_improvement(r1_durs, r2_durs)
                        print(f"   {msg}")
                        
                        if not improved:
                            # Get NCU-based suggestions
                            bottlenecks = analyze_ncu_bottlenecks(ncu_r2)
                            feedback = "Performance insufficient. " + " ".join(bottlenecks)
                            
                            if attempt >= SELF_REFLECT_AFTER_ATTEMPT:
                                self_critique = self_reflect_on_error(llm, kernel, "performance_failed", 
                                                                     "\n".join(bottlenecks), params)
                            
                            memory.add_attempt(kernel, "performance_failed", msg, 
                                              feedback=feedback, duration_us=duration_us)
                            
                            delay = exponential_backoff(attempt)
                            print(f"   ⏳  Retrying in {delay:.1f}s...")
                            time.sleep(delay)
                            continue
            
            # Success!
            print(f"   ✅ R2 Success on attempt {attempt+1}!")
            memory.add_attempt(kernel, None, None, duration_us=duration_us, 
                              feedback="Performance acceptable" if duration_us else "Compiled successfully")
            return kernel, text, {"status": "success", "attempts": attempt+1, "memory": memory, 
                                 "gen_time": gen_time, "duration_us": duration_us}
            
        except Exception as e:
            print(f"   ❌ Attempt {attempt+1} exception: {type(e).__name__}: {e}")
            memory.add_attempt(None, "exception", str(e))
            
            if attempt < max_retries:
                time.sleep(exponential_backoff(attempt))
                continue
            return None, None, {"status": "exception", "attempts": attempt+1, "error": str(e), "memory": memory}
    
    return None, None, {"status": "max_retries_exceeded", "attempts": max_retries+1, "memory": memory}


# ============================================================================
# 🚀 Main Pipeline
# ============================================================================

def main():
    print("="*80)
    print("🚀 DeepSeek-R1 CUDA Kernel Generation Pipeline")
    print("   Strategy: Stateful Hybrid Retry + Self-Reflection")
    print(f"   Config: MAX_RETRIES={MAX_RETRIES}, CONV_DEPTH={CONVERSATIONAL_DEPTH},")
    print(f"           SELF_REFLECT_AFTER={SELF_REFLECT_AFTER_ATTEMPT}, PERF_THRESHOLD={PERFORMANCE_IMPROVEMENT_THRESHOLD*100:.0f}%")
    print("="*80)
    
    # Initialize LLM
    print("\n🤖 Initializing LLM...")
    llm = LLM(
        model=LLM_MODEL,
        max_model_len=LLM_MAX_MODEL_LEN,
        gpu_memory_utilization=LLM_GPU_MEMORY_UTIL,
        enforce_eager=False,
        trust_remote_code=True,
        tensor_parallel_size=1
    )
    print("✅ LLM ready")
    
    # =========================================================================
    # ROUND 1
    # =========================================================================
    kernel_r1, text_r1, meta_r1 = generate_round1_with_retry(llm, MAX_RETRIES)
    
    if not kernel_r1:
        print(f"\n❌ Round 1 failed: {meta_r1.get('status', 'unknown')}")
        print(f"   Attempts: {meta_r1.get('attempts', 'N/A')}")
        return
    
    print(f"\n✅ Round 1 complete in {meta_r1['attempts']} attempt(s)")
    
    # NCU Profiling R1
    print(f"\n{'🔬'*40}")
    print("🔬 NCU Profiling Round 1")
    print(f"{'🔬'*40}")
    ncu_r1 = run_ncu_profiler("./testcases/backprop-cuda/main", [4096], 300, NCU_REPORT_R1)
    
    if ncu_r1.get("success"):
        durs = ncu_r1.get("duration_extracted", {}).get("kernel_durations_us", [])
        if durs:
            print(f"📊 R1 Kernel Duration: {[f'{d:.2f}μs' for d in durs]}")
    else:
        print(f"⚠️  NCU R1 failed: {ncu_r1.get('error', 'unknown')}")
    
    # =========================================================================
    # ROUND 2
    # =========================================================================
    kernel_r2, text_r2, meta_r2 = generate_round2_with_retry(llm, kernel_r1, ncu_r1, MAX_RETRIES)
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print(f"\n{'='*80}")
    print("🎉 Pipeline Complete!")
    print(f"{'='*80}")
    
    print(f"\n📁 Output Files:")
    print(f"   Round 1: {OUTPUT_PATH_ROUND1}")
    if kernel_r2:
        print(f"   Round 2: {OUTPUT_PATH_ROUND2}")
    print(f"   R1 Response: {FULL_RESPONSE_R1}")
    if text_r2:
        print(f"   R2 Response: {FULL_RESPONSE_R2}")
    print(f"   R1 NCU Report: {NCU_REPORT_R1}")
    if kernel_r2:
        print(f"   R2 NCU Report: {NCU_REPORT_R2} *(if profiled)*")
    
    print(f"\n📊 Generation Summary:")
    print(f"   Round 1: {meta_r1['status']} ({meta_r1['attempts']} attempts)")
    if kernel_r2:
        print(f"   Round 2: {meta_r2['status']} ({meta_r2['attempts']} attempts)")
        if meta_r1.get('memory') and meta_r2.get('memory'):
            print(f"   R1 Memory entries: {len(meta_r1['memory'].attempts)}")
            print(f"   R2 Memory entries: {len(meta_r2['memory'].attempts)}")
    
    print(f"\n🔬 Performance:")
    if ncu_r1.get("success") and kernel_r2:
        r1_durs = ncu_r1.get("duration_extracted", {}).get("kernel_durations_us", [])
        if meta_r2.get('duration_us') and r1_durs:
            improvement = ((r1_durs[0] - meta_r2['duration_us']) / r1_durs[0]) * 100 if r1_durs[0] > 0 else 0
            print(f"   R1 Duration: {r1_durs[0]:.2f}μs → R2 Duration: {meta_r2['duration_us']:.2f}μs ({improvement:+.1f}%)")
    
    print(f"\n💡 Configuration Tips:")
    print(f"   • Adjust MAX_RETRIES at top of script for more/fewer attempts")
    print(f"   • Lower SELF_REFLECT_AFTER_ATTEMPT for earlier self-diagnosis")
    print(f"   • Increase PERFORMANCE_IMPROVEMENT_THRESHOLD for stricter optimization")
    print(f"   • Reduce CONVERSATIONAL_DEPTH to save tokens")
    
    print(f"\n{'='*80}")
    
    return {
        "r1_kernel": kernel_r1,
        "r2_kernel": kernel_r2,
        "r1_text": text_r1,
        "r2_text": text_r2,
        "ncu_r1": ncu_r1,
        "meta_r1": meta_r1,
        "meta_r2": meta_r2
    }


if __name__ == "__main__":
    results = main()
