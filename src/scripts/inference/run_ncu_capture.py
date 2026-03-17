#!/usr/bin/env python3
"""
Script to launch ncu profiling and capture output.
"""

import subprocess
import sys
from pathlib import Path


def run_ncu_profiler(
    binary_path: str = "./testcases/backprop-cuda/main",
    args: list = None,
    capture_stderr: bool = False,
    timeout: int = None
) -> dict:
    """
    Run ncu profiler with sudo and capture output.
    
    Args:
        binary_path: Path to the binary to profile
        args: Additional arguments to pass to the binary (e.g., [4096])
        capture_stderr: Whether to capture stderr (ncu output often goes here)
        timeout: Optional timeout in seconds
    
    Returns:
        dict with keys: 'success', 'stdout', 'stderr', 'returncode'
    """
    if args is None:
        args = []
    
    # Build the command: sudo ncu <binary> <args...>
    cmd = ["sudo", "ncu", binary_path] + [str(a) for a in args]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        # Run the command and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            # stdin=subprocess.DEVNULL,  # Uncomment if you don't want interactive sudo prompt
        )
        
        # Combine stdout and stderr if needed (ncu sometimes mixes them)
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        
        return {
            "success": result.returncode == 0,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": result.returncode,
            "combined_output": stdout + stderr if capture_stderr else stdout
        }
        
    except subprocess.TimeoutExpired:
        print(f"ERROR: Command timed out after {timeout} seconds")
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Timeout after {timeout}s",
            "returncode": -1,
            "combined_output": ""
        }
    except FileNotFoundError:
        print("ERROR: 'ncu' or 'sudo' not found in PATH")
        return {
            "success": False,
            "stdout": "",
            "stderr": "Command not found",
            "returncode": -1,
            "combined_output": ""
        }
    except PermissionError:
        print("ERROR: Permission denied. Make sure you have sudo access.")
        return {
            "success": False,
            "stdout": "",
            "stderr": "Permission denied",
            "returncode": -1,
            "combined_output": ""
        }
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
            "combined_output": ""
        }


def main():
    # Configuration
    binary = "./testcases/backprop-cuda/main"
    input_size = 4096
    
    # Run the profiler
    result = run_ncu_profiler(
        binary_path=binary,
        args=[input_size],
        timeout=300  # 5 minute timeout, adjust as needed
    )
    
    # Access the captured output
    output_string = result["combined_output"]
    
    # Print summary
    print("\n" + "="*60)
    print("PROFILING COMPLETE")
    print("="*60)
    print(f"Success: {result['success']}")
    print(f"Return code: {result['returncode']}")
    print(f"Output length: {len(output_string)} characters")
    print("="*60 + "\n")
    
    # Print the full captured output
    print("=== CAPTURED NCU OUTPUT ===\n")
    print(output_string)
    
    # Optional: Save to file
    # with open("ncu_output.txt", "w") as f:
    #     f.write(output_string)
    # print("\nOutput saved to ncu_output.txt")
    
    return result


if __name__ == "__main__":
    main()