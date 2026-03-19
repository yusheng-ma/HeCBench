#!/usr/bin/env python3
# ============================================================================
# deepseek_r1_benchmark_suite.py
# Multi-Benchmark CUDA Kernel Generation with Stateful Hybrid Retry
# ============================================================================
# Features:
#   - Supports multiple benchmarks defined in ./benchmarks/subset.json
#   - Dynamic Prompt Engineering based on Kernel Specs
#   - Generate Once, Profile Many (Efficient LLM usage)
#   - Dual Verification: NCU for optimization, Regex for final reporting
#   - Cleaned up intermediate file outputs
#   - 🔥 Phase 3: Unified NCU Duration as Primary Metric
# ============================================================================
import os, re, time, shutil, subprocess, json, random, argparse, statistics
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from vllm import LLM, SamplingParams

# ============================================================================
# 🎛️ CONFIGURATION
# ============================================================================
BASE_DIR = "."
TESTCASES_ROOT = os.path.join(BASE_DIR, "testcases")
SUBSET_JSON_PATH = os.path.join(BASE_DIR, "benchmarks/subset.json")

# LLM Configuration
LLM_MODEL = "Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ"
LLM_MAX_MODEL_LEN = 4096
LLM_GPU_MEMORY_UTIL = 0.9
LLM_TEMPERATURE_BASE = 0.5
LLM_TEMPERATURE_MAX = 0.7
LLM_TEMPERATURE_INCREMENT = 0.05

# Retry Configuration
MAX_RETRIES = 5
RETRY_DELAY_BASE = 2
RETRY_DELAY_JITTER = 0.2
CONVERSATIONAL_DEPTH = 2
SELF_REFLECT_AFTER_ATTEMPT = 2
PERFORMANCE_IMPROVEMENT_THRESHOLD = 0.01

# NCU Profiling Configuration
NCU_LAUNCH_COUNT = 5
NCU_TIMEOUT_BASE = 120  # seconds per input set
NCU_METRIC_AGGREGATION = "mean"  # "mean", "median", or "first"

# ============================================================================
# 📦 KERNEL SPECIFICATIONS (Must match main.cu in each testcase)
# ============================================================================
KERNEL_SPECS = {
    "backprop": {
        "dir_name": "backprop-cuda",
        "header_name": "bpnn_layerforward.h",
        "kernel_name": "kernel_layerforward",
        "signature": """__global__ void kernel_layerforward(
    const float* __restrict__ input,
    float* __restrict__ input_weights,
    float* __restrict__ hidden_partial_sum,
    const int hid)""",
        "algorithm_hint": "BPNN forward pass. Load input and weights into shared memory. Perform matrix multiplication via tree reduction. Ensure bounds checking.",
        "representative_input": ["4096"]
    },
    "floydwarshall": {
        "dir_name": "floydwarshall-cuda",
        "header_name": "fw.h",
        "kernel_name": "fw_kernel",
        "signature": """__global__ void fw_kernel(
    float* __restrict__ dist,
    const int n)""",
        "algorithm_hint": "Floyd-Warshall all-pairs shortest path. Use tiled shared memory to reduce global memory access. Handle diagonal updates carefully.",
        "representative_input": ["256", "50", "16"]
    },
    "convolution3D": {
        "dir_name": "convolution3D-cuda",
        "header_name": "conv3d.h",
        "kernel_name": "conv3d_kernel",
        "signature": """__global__ void conv3d_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    int dimx, int dimy, int dimz, int kdimx, int kdimy, int kdimz)""",
        "algorithm_hint": "3D Convolution. Use 3D blocking and shared memory caching for input and kernel tiles. Optimize for memory coalescing.",
        "representative_input": ["8", "3", "8", "7", "7", "3", "100"]
    },
    "crossEntropy": {
        "dir_name": "crossEntropy-cuda",
        "header_name": "cross_entropy.h",
        "kernel_name": "cross_entropy_kernel",
        "signature": """__global__ void cross_entropy_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int n)""",
        "algorithm_hint": "Cross Entropy Loss. Parallel reduction for summing log probabilities. Handle log(0) stability.",
        "representative_input": ["10"]
    },
    "softmax": {
        "dir_name": "softmax-cuda",
        "header_name": "softmax.h",
        "kernel_name": "softmax_kernel",
        "signature": """__global__ void softmax_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int n, const int dim)""",
        "algorithm_hint": "Softmax function. Subtract max for numerical stability. Use shared memory for sum reduction.",
        "representative_input": ["10000", "256", "0", "100"]
    }
}

# ============================================================================
# 📦 Data Classes
# ============================================================================
@dataclass
class AttemptRecord:
    attempt_num: int
    error_type: Optional[str] = None
    error_detail: Optional[str] = None
    code_hint: Optional[str] = None
    feedback: Optional[str] = None
    duration_us: Optional[float] = None

    def to_summary(self) -> str:
        lines = [f"  • Attempt #{self.attempt_num}: {self.error_type or 'success'}"]
        if self.code_hint: lines.append(f"    - Approach: {self.code_hint}")
        if self.feedback: lines.append(f"    - Feedback: {self.feedback[:150] if self.feedback else ''}")
        if self.duration_us: lines.append(f"    - Duration: {self.duration_us:.2f}μs")
        return "\n".join(lines)

@dataclass
class RetryMemory:
    max_history: int = CONVERSATIONAL_DEPTH
    attempts: List[AttemptRecord] = field(default_factory=list)
    round_num: int = 1
    best_duration_us: Optional[float] = None
    best_kernel_code: Optional[str] = None

    def add_attempt(self, code: Optional[str], error_type: Optional[str],
                    error_detail: Optional[str], feedback: Optional[str] = None,
                    duration_us: Optional[float] = None):
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
        if duration_us and (self.best_duration_us is None or duration_us < self.best_duration_us):
            self.best_duration_us = duration_us
            self.best_kernel_code = code
        if len(self.attempts) > self.max_history:
            self.attempts.pop(0)

    def _extract_code_hint(self, code: str) -> Optional[str]:
        code_lower = code.lower()
        hints = []
        if "shared" in code_lower: hints.append("shared memory")
        if "reduction" in code_lower: hints.append("tree reduction")
        if "coalesc" in code_lower: hints.append("memory coalescing")
        return ", ".join(hints) if hints else "standard implementation"

    def build_context_suffix(self) -> str:
        if not self.attempts: return ""
        lines = ["\n# 🔄 Revision History (Recent Attempts):", f"  Round: {self.round_num}"]
        for att in self.attempts: lines.append(att.to_summary())
        if self.best_duration_us: lines.append(f"\n🏆 Best duration so far: {self.best_duration_us:.2f}μs")
        return "\n".join(lines)

    def get_failure_pattern(self) -> Optional[str]:
        if len(self.attempts) < 2: return None
        error_types = [a.error_type for a in self.attempts if a.error_type]
        if len(error_types) >= 2 and len(set(error_types)) == 1:
            return f"Recurring issue: {error_types[0]}"
        return None

# ============================================================================
# 🔬 Profiling & Extraction
# ============================================================================
def run_ncu_profiler(binary_path: str, args: List[str], timeout: int = NCU_TIMEOUT_BASE) -> Dict[str, Any]:
    """Run Nsight Compute profiler and return structured results"""
    cmd = ["ncu", "--set", "basic", "--launch-count", str(NCU_LAUNCH_COUNT), binary_path] + [str(a) for a in args]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=BASE_DIR)
        combined = (result.stdout or "") + (result.stderr or "")
        return {
            "success": result.returncode == 0,
            "combined": combined,
            "duration_extracted": extract_duration_from_ncu(combined),
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired as e:
        return {"success": False, "error": f"Timeout after {timeout}s", "combined": "", "timeout": True}
    except FileNotFoundError:
        return {"success": False, "error": "ncu command not found. Ensure Nsight Compute is in PATH", "combined": ""}
    except Exception as e:
        return {"success": False, "error": str(e), "combined": ""}

def extract_duration_from_ncu(ncu_output: str) -> Dict[str, Any]:
    """Extract kernel duration metrics from NCU output with multiple pattern matching"""
    metrics = {}
    durations = []
    
    # Pattern 1: Standard "Duration (us) XXX.XX" format
    pattern1 = re.findall(r'Duration\s+(?:\(us\)|us)?\s*([\d,]+\.?\d*)', ncu_output, re.I)
    durations.extend([float(d.replace(',', '')) for d in pattern1 if d])
    
    # Pattern 2: Kernel-specific duration with kernel name context
    kernel_dur = re.findall(r'Kernel:\s*\S+.*?Duration:\s*([\d.]+)\s*us', ncu_output, re.I | re.DOTALL)
    durations.extend([float(d) for d in kernel_dur if d])
    
    # Pattern 3: CSV-style output parsing (if --csv flag used)
    if 'Kernel Name,Duration' in ncu_output or 'duration' in ncu_output.lower():
        csv_durs = re.findall(r'[\d.]+\s*us', ncu_output)
        durations.extend([float(d.replace('us','').strip()) for d in csv_durs])
    
    # Aggregate based on config
    if durations:
        if NCU_METRIC_AGGREGATION == "mean":
            metrics["kernel_duration_us"] = statistics.mean(durations)
        elif NCU_METRIC_AGGREGATION == "median":
            metrics["kernel_duration_us"] = statistics.median(durations)
        else:  # "first"
            metrics["kernel_duration_us"] = durations[0]
        metrics["all_durations_us"] = durations  # Keep raw for analysis
    
    # Extract additional useful metrics
    sm_eff = re.search(r'SM\s+Efficiency[:\s]+([\d.]+)\s*%?', ncu_output, re.I)
    if sm_eff:
        metrics["sm_efficiency_pct"] = float(sm_eff.group(1))
    
    mem_bw = re.search(r'Memory\s+Throughput[:\s]+([\d.]+)\s*(?:GB/s|GBps)', ncu_output, re.I)
    if mem_bw:
        metrics["memory_bandwidth_gbps"] = float(mem_bw.group(1))
    
    return metrics

def parse_binary_output(stdout: str, regex_pattern: str) -> Optional[float]:
    """Legacy: Parse benchmark specific metric using subset.json regex (fallback only)"""
    try:
        match = re.search(regex_pattern, stdout)
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return None

def extract_kernel_function(response_text: str, kernel_name: str) -> Optional[str]:
    """Extract CUDA kernel function from LLM response using brace-matching or code blocks"""
    # Method 1: Brace matching from signature
    match = re.search(r'__global__\s+void\s+' + re.escape(kernel_name) + r'\s*\([^)]*\)', response_text)
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
                return response_text[start:end]
    
    # Method 2: Code block extraction
    block = re.search(r'```(?:cuda|cpp|c)?\s*([\s\S]*?' + re.escape(kernel_name) + r'[\s\S]*?)```', response_text)
    if block:
        return block.group(1).strip()
    
    return None

def write_kernel_header(kernel_name: str, kernel_code: str, output_path: str) -> Tuple[bool, str]:
    """Write kernel code to header file with proper include guards"""
    try:
        guard = os.path.basename(output_path).replace('.', '_').replace('-', '_').upper()
        content = f"""// Auto-generated CUDA Kernel by DeepSeek-R1
#ifndef {guard}
#define {guard}
#include <cuda.h>
{kernel_code}
#endif
"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, ""
    except Exception as e:
        return False, str(e)

def compile_testcase(testcase_dir: str) -> Tuple[bool, str]:
    """Compile testcase using Makefile"""
    try:
        subprocess.run(["make", "clean"], cwd=testcase_dir, capture_output=True, check=False)
        result = subprocess.run(["make"], cwd=testcase_dir, capture_output=True, text=True)
        if result.returncode == 0:
            return True, ""
        return False, result.stderr[:500]
    except Exception as e:
        return False, str(e)

# ============================================================================
# 🧠 Self-Reflection
# ============================================================================
def self_reflect_on_error(llm: LLM, code: str, error_type: str, error_detail: str, params: SamplingParams) -> str:
    """Generate self-critique prompt for error analysis"""
    critique_prompts = {
        "extract_failed": "Kernel function couldn't be extracted. Check signature matching and code block formatting.",
        "compile_failed": f"Compilation failed:\n{error_detail[:400]}\nIdentify syntax errors, type mismatches, or missing includes.",
        "performance_failed": f"Performance insufficient per NCU report:\n{error_detail[:400]}\nSuggest CUDA optimizations.",
        "exception": "Unexpected runtime error. Suggest robustness improvements or error handling."
    }
    base = critique_prompts.get(error_type, "Analyze what went wrong and suggest concrete fixes.")
    prompt = f"""Previous Code:
```cuda
{code[:1500] if code else '<no code>'}
```
{base}
Output ONLY a brief diagnosis (1-2 sentences) and specific fix suggestion."""
    try:
        outputs = llm.generate([prompt], SamplingParams(temperature=0.3, max_tokens=300))
        return outputs[0].outputs[0].text.strip()
    except Exception:
        return "Unable to diagnose. Consider manual review."

# ============================================================================
# 📋 Dynamic Prompt Builders
# ============================================================================
def build_round1_prompt(spec: Dict[str, Any], input_example: List[str]) -> str:
    return f"""<|begin_of_sentence|><|User|># Role
You are an expert CUDA Developer.
# Task
Implement a CUDA kernel for {spec.get('algorithm_hint', 'the specified algorithm')}.
# Target Kernel Specification (MUST MATCH EXACTLY)
```cpp
{spec['signature']}
```
# Implementation Guidelines
- Use `__global__`, `__restrict__`, and proper CUDA conventions.
- {spec['algorithm_hint']}
- Input sizes example for context: {input_example}
# Output Requirement
- Return ONLY the CUDA code inside ```cuda ... ``` blocks
- Ensure signature matches Target Kernel Specification exactly<|Assistant|>"""

def build_round2_prompt(spec: Dict[str, Any], naive_code: str, ncu_report: str) -> str:
    return f"""<|begin_of_sentence|><|User|># Role
Expert CUDA Developer specializing in performance optimization.
# Task
Optimize the kernel below based on Nsight Compute profiling report.
# 📊 Profiling Feedback (Nsight Compute)
{ncu_report}
# Original Implementation
```cuda
{naive_code}
```
# Target Kernel Specification (MUST MATCH EXACTLY)
```cpp
{spec['signature']}
```
# Output Requirement
- Return ONLY optimized CUDA code inside ```cuda ... ``` blocks
- Keep signature unchanged
- Focus on: memory coalescing, shared memory usage, occupancy, instruction-level parallelism<|Assistant|>"""

def enhance_prompt_with_retry_context(base_prompt: str, memory: RetryMemory, self_critique: Optional[str] = None) -> str:
    """Add revision history and failure patterns to prompt for context-aware retry"""
    enhanced = base_prompt
    history = memory.build_context_suffix()
    if history: enhanced += history
    if self_critique: enhanced += f"\n# 💡 Self-Diagnosis:\n{self_critique}"
    pattern = memory.get_failure_pattern()
    if pattern: enhanced += f"\n# ⚠️ Recurring Issue Detected:\n{pattern}"
    return enhanced

# ============================================================================
# 🔄 Generation Logic (Generic)
# ============================================================================
def generate_kernel_round(llm: LLM, spec: Dict[str, Any], round_num: int, 
                          base_prompt: str, previous_code: Optional[str] = None,
                          ncu_feedback: Optional[Dict] = None,
                          use_ncu: bool = False,
                          max_retries: int = MAX_RETRIES) -> Tuple[Optional[str], Dict]:
    """
    Generate/optimize kernel with hybrid retry logic.
    
    Args:
        use_ncu: If True, use NCU duration for performance-based retry decisions in Round 2.
                 If False, still profile for logging but only retry on compile/extract errors.
    """
    memory = RetryMemory(max_history=CONVERSATIONAL_DEPTH, round_num=round_num)
    self_critique = None
    
    # Prepare NCU report string for prompt
    ncu_report_str = "⚠️ No profiling data available."
    if ncu_feedback and ncu_feedback.get("success"):
        metrics = ncu_feedback.get("duration_extracted", {})
        ncu_report_str = f"## Key Metrics\n{json.dumps(metrics, indent=2)}\n"
        if metrics.get("kernel_duration_us"):
            ncu_report_str += f"## Kernel Duration\n{metrics['kernel_duration_us']:.2f} μs\n"
    
    if round_num == 2 and previous_code:
        base_prompt = build_round2_prompt(spec, previous_code, ncu_report_str)
    
    # Reference duration from Round 1 for improvement calculation
    r1_duration = ncu_feedback.get("duration_extracted", {}).get("kernel_duration_us") if ncu_feedback else None

    for attempt in range(max_retries + 1):
        print(f"   🔄 Round {round_num} Attempt {attempt+1}/{max_retries+1}")
        start_time = time.time()
        try:
            # Build prompt with context for retries
            prompt = base_prompt if attempt == 0 else enhance_prompt_with_retry_context(base_prompt, memory, self_critique)
            params = SamplingParams(
                temperature=min(LLM_TEMPERATURE_MAX, LLM_TEMPERATURE_BASE + attempt * LLM_TEMPERATURE_INCREMENT), 
                top_p=0.9, 
                max_tokens=LLM_MAX_MODEL_LEN
            )
            
            # Generate response
            outputs = llm.generate([prompt], params)
            text = outputs[0].outputs[0].text
            gen_duration = (time.time() - start_time) * 1e6  # μs
            
            # Extract kernel function
            kernel = extract_kernel_function(text, spec['kernel_name'])
            if not kernel:
                raise Exception("Extract failed: Could not find kernel function in response")
            
            # Write to temp header
            temp_header = os.path.join(BASE_DIR, f"temp_{spec['dir_name']}_{round_num}.h")
            ok, err = write_kernel_header(spec['kernel_name'], kernel, temp_header)
            if not ok: raise Exception(f"Write failed: {err}")
            
            # Copy to testcase directory
            dst_dir = os.path.join(TESTCASES_ROOT, spec['dir_name'])
            dst_path = os.path.join(dst_dir, spec['header_name'])
            shutil.copy2(temp_header, dst_path)
            
            # Compile
            ok, err = compile_testcase(dst_dir)
            if not ok: raise Exception(f"Compile failed: {err}")
            
            # Profile with NCU for retry decision & logging
            duration_us = None
            binary_path = os.path.join(dst_dir, "main")
            
            if os.path.exists(binary_path):
                ncu_res = run_ncu_profiler(str(binary_path), spec.get('representative_input', []))
                if ncu_res.get("success"):
                    extracted = ncu_res.get("duration_extracted", {})
                    if extracted.get("kernel_duration_us"):
                        duration_us = float(extracted["kernel_duration_us"])
                        
                        # 🔑 NCU-based retry decision (Round 2 only, if enabled)
                        if use_ncu and round_num == 2 and r1_duration and r1_duration > 0:
                            improvement = (r1_duration - duration_us) / r1_duration
                            if improvement < PERFORMANCE_IMPROVEMENT_THRESHOLD:
                                raise Exception(f"Performance insufficient: improvement={improvement*100:.2f}% < threshold={PERFORMANCE_IMPROVEMENT_THRESHOLD*100:.2f}%")
            
            print(f"   ✅ Round {round_num} Success | Duration: {duration_us:.2f}μs" if duration_us else f"   ✅ Round {round_num} Success")
            memory.add_attempt(kernel, None, None, duration_us=duration_us)
            
            # Cleanup temp file
            if os.path.exists(temp_header): os.remove(temp_header)
            
            return kernel, {
                "status": "success", 
                "attempts": attempt+1, 
                "memory": memory, 
                "duration_us": duration_us,
                "generation_time_us": gen_duration
            }
            
        except Exception as e:
            err_str = str(e)
            print(f"   ⚠️ Failed: {err_str[:100]}")
            error_type = "exception"
            if "Extract" in err_str: error_type = "extract_failed"
            elif "Compile" in err_str or "write" in err_str.lower(): error_type = "compile_failed"
            elif "Performance" in err_str: error_type = "performance_failed"
            
            # Self-reflection after threshold attempts
            if attempt >= SELF_REFLECT_AFTER_ATTEMPT:
                self_critique = self_reflect_on_error(
                    llm, 
                    kernel if 'kernel' in locals() else "", 
                    error_type, 
                    err_str, 
                    params
                )
            
            memory.add_attempt(
                kernel if 'kernel' in locals() else None, 
                error_type, 
                err_str
            )
            
            if attempt < max_retries:
                jitter = random.uniform(-RETRY_DELAY_JITTER, RETRY_DELAY_JITTER)
                time.sleep(max(0.1, RETRY_DELAY_BASE + jitter))
            else:
                if os.path.exists(temp_header): os.remove(temp_header)
                return None, {"status": "failed", "error": err_str, "memory": memory, "attempts": attempt+1}
    
    return None, {"status": "failed", "memory": memory}

# ============================================================================
# 🚀 Main Pipeline
# ============================================================================
def main():
    args = parse_args()
    
    # Load Subset Config
    if not os.path.exists(SUBSET_JSON_PATH):
        print(f"❌ Error: {SUBSET_JSON_PATH} not found.")
        return
    with open(SUBSET_JSON_PATH, 'r') as f:
        subset_config = json.load(f)
    
    print(f"🚀 Starting Benchmark Suite Run: {args.run_name}")
    print(f"📂 Testcases Root: {TESTCASES_ROOT}")
    print(f"🔧 NCU Feedback Mode: {'ENABLED (performance retry)' if args.use_ncu else 'DISABLED (baseline only)'}")
    print(f"📊 Phase 3 Metric: NCU kernel duration ({NCU_METRIC_AGGREGATION} of {NCU_LAUNCH_COUNT} launches)")

    # Initialize LLM
    llm = LLM(
        model=LLM_MODEL, 
        max_model_len=LLM_MAX_MODEL_LEN, 
        gpu_memory_utilization=LLM_GPU_MEMORY_UTIL, 
        enforce_eager=False, 
        trust_remote_code=True, 
        tensor_parallel_size=1
    )
    
    # Setup logging
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = log_dir / f"{args.run_name}.jsonl"
    
    for bmark_name, bmark_info in subset_config.items():
        print(f"\n{'='*80}")
        print(f"🎯 Processing Benchmark: {bmark_name}")
        print(f"{'='*80}")
        
        if bmark_name not in KERNEL_SPECS:
            print(f"⚠️ Skipping {bmark_name}: No kernel spec defined.")
            continue
            
        spec = KERNEL_SPECS[bmark_name]
        regex_pattern = bmark_info[0]  # Legacy regex (fallback only)
        input_sets = bmark_info[1]
        testcase_dir = os.path.join(TESTCASES_ROOT, spec['dir_name'])
        
        if not os.path.exists(testcase_dir):
            print(f"❌ Directory not found: {testcase_dir}")
            continue
        
        # ─────────────────────────────────────────────────────────────
        # Phase 1: Initial Code Generation (Once per benchmark)
        # ─────────────────────────────────────────────────────────────
        print(f"\n🛠️ Phase 1: Generating Initial Code for {bmark_name}...")
        rep_input = spec.get('representative_input', [])
        
        prompt_r1 = build_round1_prompt(spec, rep_input)
        kernel_r1, meta_r1 = generate_kernel_round(llm, spec, 1, prompt_r1, use_ncu=False)
        if not kernel_r1:
            print(f"❌ Round 1 Failed for {bmark_name}. Skipping to next benchmark.")
            continue
        
        # Round 1 profiling (baseline)
        binary = os.path.join(testcase_dir, "main")
        ncu_r1 = run_ncu_profiler(binary, rep_input)
        dur_r1 = ncu_r1.get("duration_extracted", {}).get("kernel_duration_us") if ncu_r1.get("success") else None
        
        # ─────────────────────────────────────────────────────────────
        # Phase 2: Optimization Round (with optional NCU-guided retry)
        # ─────────────────────────────────────────────────────────────
        print(f"\n🚀 Phase 2: Optimizing Code...")
        kernel_r2, meta_r2 = None, {}
        dur_r2 = None
        
        kernel_r2, meta_r2 = generate_kernel_round(
            llm, spec, 2, "", kernel_r1, ncu_r1, 
            use_ncu=args.use_ncu  # 🔑 Key flag for NCU-based retry logic
        )
        
        if kernel_r2:
            dur_r2 = meta_r2.get('duration_us')
        else:
            print(f"⚠️ Round 2 Failed, falling back to Round 1 kernel for profiling.")
            kernel_r2 = kernel_r1
            meta_r2 = meta_r1
            dur_r2 = dur_r1

        # ─────────────────────────────────────────────────────────────
        # 🔥 Phase 3: Full Profiling with NCU Duration as Primary Metric
        # ─────────────────────────────────────────────────────────────
        print(f"\n📊 Phase 3: Profiling All Input Sizes via Nsight Compute...")
        profiling_results = []
        
        for idx, inputs in enumerate(input_sets):
            try:
                # Run NCU profiler for this input configuration
                ncu_res = run_ncu_profiler(binary, inputs, timeout=NCU_TIMEOUT_BASE)
                
                if ncu_res.get("success"):
                    extracted = ncu_res.get("duration_extracted", {})
                    duration_val = extracted.get("kernel_duration_us")
                    
                    if duration_val is not None:
                        profiling_results.append({
                            "inputs": inputs,
                            "metric_value": duration_val,
                            "metric_unit": "microseconds",
                            "metric_source": "ncu_kernel_duration",
                            "aggregation": NCU_METRIC_AGGREGATION,
                            "launch_count": NCU_LAUNCH_COUNT,
                            "all_durations": extracted.get("all_durations_us", []),
                            "additional_metrics": {k:v for k,v in extracted.items() if k not in ["kernel_duration_us", "all_durations_us"]},
                            "success": True
                        })
                        print(f"   ✅ [{idx+1}/{len(input_sets)}] Inputs {inputs}: {duration_val:.2f} μs")
                    else:
                        profiling_results.append({
                            "inputs": inputs,
                            "error": "Duration not found in NCU output",
                            "ncu_output_snippet": ncu_res.get("combined", "")[:300],
                            "success": False
                        })
                        print(f"   ❌ [{idx+1}/{len(input_sets)}] Inputs {inputs}: No duration extracted")
                else:
                    # 🔁 Optional fallback to legacy regex parsing
                    if regex_pattern:
                        cmd = [binary] + [str(x) for x in inputs]
                        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=BASE_DIR)
                        fallback_val = parse_binary_output(res.stdout, regex_pattern)
                        if fallback_val:
                            profiling_results.append({
                                "inputs": inputs,
                                "metric_value": fallback_val,
                                "metric_unit": "unknown",
                                "metric_source": "regex_fallback",
                                "success": True,
                                "warning": "Used legacy regex fallback (NCU failed)"
                            })
                            print(f"   ⚠️ [{idx+1}/{len(input_sets)}] Inputs {inputs}: {fallback_val} (regex fallback)")
                            continue
                    
                    profiling_results.append({
                        "inputs": inputs,
                        "error": ncu_res.get("error", "NCU profiling failed"),
                        "timeout": ncu_res.get("timeout", False),
                        "success": False
                    })
                    print(f"   ❌ [{idx+1}/{len(input_sets)}] Inputs {inputs}: NCU failed - {ncu_res.get('error', 'unknown')}")
                    
            except subprocess.TimeoutExpired:
                profiling_results.append({"inputs": inputs, "error": f"Timeout > {NCU_TIMEOUT_BASE}s", "success": False})
                print(f"   ⏱️ [{idx+1}/{len(input_sets)}] Inputs {inputs}: Timeout")
            except Exception as e:
                profiling_results.append({"inputs": inputs, "error": str(e), "success": False})
                print(f"   ❌ [{idx+1}/{len(input_sets)}] Inputs {inputs}: Exception - {e}")

        # ─────────────────────────────────────────────────────────────
        # Log Results to JSONL
        # ─────────────────────────────────────────────────────────────
        entry = {
            "timestamp": datetime.now().isoformat(),
            "benchmark": bmark_name,
            "run_name": args.run_name,
            "config": {
                "use_ncu_retry": args.use_ncu,
                "ncu_launch_count": NCU_LAUNCH_COUNT,
                "ncu_aggregation": NCU_METRIC_AGGREGATION,
                "ncu_timeout_sec": NCU_TIMEOUT_BASE
            },
            "generation": {
                "round1": {
                    "attempts": meta_r1.get("attempts"),
                    "duration_us": dur_r1,
                    "generation_time_us": meta_r1.get("generation_time_us")
                },
                "round2": {
                    "attempts": meta_r2.get("attempts") if meta_r2 else None,
                    "duration_us": dur_r2,
                    "generation_time_us": meta_r2.get("generation_time_us") if meta_r2 else None,
                    "improvement_pct": ((dur_r1 - dur_r2) / dur_r1 * 100) if (dur_r1 and dur_r2) else None
                }
            },
            "profiling_results": profiling_results,
            "summary": {
                "total_inputs_tested": len(input_sets),
                "successful_profilings": sum(1 for r in profiling_results if r.get("success")),
                "avg_duration_us": statistics.mean([r["metric_value"] for r in profiling_results if r.get("success") and r.get("metric_value")]) if any(r.get("success") and r.get("metric_value") for r in profiling_results) else None
            }
        }

        with open(jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\n🎉 Suite Complete. Logs saved to: {jsonl_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="DeepSeek-R1 CUDA Benchmark Suite")
    parser.add_argument("--run_name", type=str, required=True, help="Unique identifier for this experiment run")
    parser.add_argument("--log_dir", type=str, default="experiment_logs", help="Directory for JSONL logs")
    parser.add_argument("--use_ncu", action="store_true", default=False, 
                       help="Enable NCU feedback for Round 2 retry decisions (performance-guided optimization)")
    return parser.parse_args()

if __name__ == "__main__":
    main()
