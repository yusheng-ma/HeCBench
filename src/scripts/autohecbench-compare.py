#!/usr/bin/env python3
#
# Script to compare benchmark results from different backends
# Simplified: single numeric speedup column
# speedup = time_b / time_a  (>1 means backend_a is faster)

import argparse
import csv
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import math

def parse_benchmark_name(name: str) -> Tuple[str, str, Optional[str]]:
    """
    Parse benchmark name into (base_name, backend, params)
    
    Examples:
        floydwarshall-cuda-1024_100_16 → ("floydwarshall", "cuda", "1024_100_16")
        backprop-sycl-65536 → ("backprop", "sycl", "65536")
    """
    match = re.match(r'^(.+?)-(cuda|sycl|hip|omp)(?:-(.+))?$', name)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return name, None, None


def load_csv(filepath: str) -> Dict[str, float]:
    """Load CSV file and return dict of {benchmark_name: time}"""
    results = {}
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2 and row[0].strip() and row[1].strip():
                name = row[0].strip()
                try:
                    time_val = float(row[1].strip())
                    results[name] = time_val
                except ValueError:
                    print(f"Warning: Could not parse '{row[1]}' for '{name}'", file=sys.stderr)
    return results


def match_benchmarks(results_a: Dict[str, float], results_b: Dict[str, float], 
                     backend_a: str, backend_b: str) -> List[Dict]:
    """Match benchmarks by base_name + params, calculate speedup"""
    comparisons = []
    index: Dict[Tuple[str, Optional[str]], Dict[str, float]] = defaultdict(dict)
    
    for name, time_val in {**results_a, **results_b}.items():
        base, backend, params = parse_benchmark_name(name)
        if backend and base:
            index[(base, params)][backend] = time_val
    
    for (base_name, params), backends in index.items():
        if backend_a in backends and backend_b in backends:
            time_a = backends[backend_a]
            time_b = backends[backend_b]
            
            # 🔥 speedup = time_b / time_a
            # >1: backend_a is faster, <1: backend_a is slower, =1: equal
            speedup = time_b / time_a if time_a > 0 else float('inf')
            
            param_str = f"-{params}" if params else ""
            comparisons.append({
                'benchmark': f"{base_name}{param_str}",
                'base_name': base_name,
                'params': params or '',
                f'{backend_a}_time': time_a,
                f'{backend_b}_time': time_b,
                'speedup': speedup,  # 🔥 Single numeric column
            })
    
    comparisons.sort(key=lambda x: (x['base_name'], x['params']))
    return comparisons


def main():
    parser = argparse.ArgumentParser(description='Compare benchmark results (simplified)')
    parser.add_argument('csv_a', help='First CSV file (baseline, e.g., cuda.csv)')
    parser.add_argument('csv_b', help='Second CSV file (compare against, e.g., sycl.csv)')
    parser.add_argument('--backend-a', '-a', default='cuda', help='Backend name for csv_a')
    parser.add_argument('--backend-b', '-b', default='sycl', help='Backend name for csv_b')
    parser.add_argument('--output', '-o', help='Output CSV file')
    parser.add_argument('--summary-only', action='store_true', help='Only print summary')
    parser.add_argument('--sort-by', choices=['name', 'speedup', 'time-a', 'time-b'], 
                        default='name', help='Sort order')
    parser.add_argument('--filter', '-f', help='Filter by benchmark name (regex)')
    
    args = parser.parse_args()
    
    # Load
    print(f"Loading {args.csv_a}...")
    results_a = load_csv(args.csv_a)
    print(f"  Loaded {len(results_a)} benchmarks")
    
    print(f"Loading {args.csv_b}...")
    results_b = load_csv(args.csv_b)
    print(f"  Loaded {len(results_b)} benchmarks")
    
    # Match
    comparisons = match_benchmarks(results_a, results_b, args.backend_a, args.backend_b)
    print(f"\nMatched {len(comparisons)} benchmark pairs")
    
    # Filter
    if args.filter:
        pattern = re.compile(args.filter)
        comparisons = [c for c in comparisons if pattern.search(c['benchmark'])]
        print(f"After filter: {len(comparisons)} benchmarks")
    
    if not comparisons:
        print("No matching benchmarks!", file=sys.stderr)
        sys.exit(1)
    
    # Sort
    if args.sort_by == 'speedup':
        comparisons.sort(key=lambda x: x['speedup'], reverse=True)
    elif args.sort_by == 'time-a':
        comparisons.sort(key=lambda x: x[f'{args.backend_a}_time'])
    elif args.sort_by == 'time-b':
        comparisons.sort(key=lambda x: x[f'{args.backend_b}_time'])
    
    # Output CSV
    if args.output:
        out_f = open(args.output, 'w', newline='')
    else:
        out_f = sys.stdout
    
    if not args.summary_only:
        writer = csv.writer(out_f)
        writer.writerow(['benchmark', f'{args.backend_a}_time', f'{args.backend_b}_time', 'speedup'])
        for comp in comparisons:
            writer.writerow([
                comp['benchmark'],
                f"{comp[f'{args.backend_a}_time']:.6f}",
                f"{comp[f'{args.backend_b}_time']:.6f}",
                f"{comp['speedup']:.4f}"  # 🔥 Pure numeric
            ])
    
    if args.output:
        out_f.close()
        print(f"\nWrote to {args.output}")
    
    # Summary
    print("\n" + "="*70)
    print(f"SUMMARY: {args.backend_a.upper()} vs {args.backend_b.upper()}")
    print(f"speedup = {args.backend_b}_time / {args.backend_a}_time")
    print(f"  >1: {args.backend_a} is faster  |  <1: {args.backend_a} is slower")
    print("="*70)
    
    speedups = [c['speedup'] for c in comparisons if 0 < c['speedup'] < float('inf')]
    
    if speedups:
        geo_mean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        print(f"\n📈 Speedup Statistics:")
        print(f"   Count: {len(comparisons)}")
        print(f"   Geometric mean: {geo_mean:.3f}")
        print(f"   Arithmetic mean: {sum(speedups)/len(speedups):.3f}")
        print(f"   Min: {min(speedups):.3f}  |  Max: {max(speedups):.3f}")
        
        # Categories
        faster = sum(1 for s in speedups if s > 1.05)
        slower = sum(1 for s in speedups if s < 0.95)
        similar = len(speedups) - faster - slower
        print(f"\n📊 Categories (>5% threshold):")
        print(f"   {args.backend_a} faster: {faster}/{len(speedups)} ({100*faster/len(speedups):.1f}%)")
        print(f"   Similar: {similar}/{len(speedups)} ({100*similar/len(speedups):.1f}%)")
        print(f"   {args.backend_b} faster: {slower}/{len(speedups)} ({100*slower/len(speedups):.1f}%)")
    
    # Compact table
    print(f"\n📋 Per-Benchmark (first 20):")
    print(f"{'Benchmark':<35} {args.backend_a:>9} {args.backend_b:>9} speedup")
    print("-" * 65)
    
    for comp in comparisons[:20]:
        name = comp['benchmark']
        if len(name) > 34:
            name = name[:31] + "..."
        print(f"{name:<35} {comp[f'{args.backend_a}_time']:9.4f} {comp[f'{args.backend_b}_time']:9.4f} {comp['speedup']:7.4f}")
    
    if len(comparisons) > 20:
        print(f"... and {len(comparisons) - 20} more")
    
    print("="*70)


if __name__ == "__main__":
    main()