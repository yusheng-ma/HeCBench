#!/usr/bin/env python3
"""
Runner to profile a benchmark family defined in benchmarks/subset.json using ncu.

This updated script reads benchmarks the same way as the HeCBench runner:
- Accepts positional --bench arguments which can be backend groups ('cuda','hip','sycl')
  or specific benchmark names (optionally including backend and parameter suffixes).
- Supports benchmarks with multiple parameter sets in subset.json.
- Creates per-testcase output under --ncu-out/<set>/<bench>/<benchname>/<benchname>.csv
- Default: do NOT pass --metrics to ncu (use --use-metrics to enable).
- If requested metric name not present in CSV, script will try to match a CSV metric
  (heuristics for sm_efficiency/occupancy).
- Uses CSV Duration (ns) as ncu_time_s (seconds). Also keeps wallclock ncu_time_s_wallclock.
- Cleans CSV preamble, removes empty logs, retries on missing metrics when --use-metrics used.

Usage example (no --metrics passed to ncu):
TMPDIR=/mnt/disk3/yusheng/tmp_ncu XDG_RUNTIME_DIR=/mnt/disk3/yusheng/tmp_ncu \
./run_ncu_bench.py --bench floydwarshall --bench-dir /mnt/disk3/yusheng/HeCBench/src \
  --ncu-binary ncu --ncu-set basic --launch-skip 0 --launch-count 1 \
  --ncu-out reports/ncu --metrics "sm_efficiency,achieved_occupancy" --keep-logs
"""
from __future__ import annotations
import os
import sys
import json
import time
import shlex
import csv
import re
import argparse
import subprocess
from collections import defaultdict
from typing import List, Tuple

# -----------------------
# utility helpers
# -----------------------
def safe_mkdir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def detect_csv_delimiter_and_header(path: str):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        sample = ''
        for _ in range(50):
            line = f.readline()
            if not line:
                break
            if not line.startswith('#') and line.strip():
                sample += line
                break
        if not sample:
            return ',', None
        try:
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample)
            delim = dialect.delimiter
        except Exception:
            delim = ','
    return delim, sample

def clean_csv_if_needed(csv_path: str) -> bool:
    """Trim any profiler preamble and duplicate CSV headers; return True if header found."""
    header_re = re.compile(r'^\s*"ID"\s*,\s*"Process ID"', re.IGNORECASE)
    try:
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception:
        return False

    header_idx = None
    for i, L in enumerate(lines):
        if header_re.search(L):
            header_idx = i
            break
    if header_idx is None:
        return False

    cleaned = []
    seen_header = False
    for L in lines[header_idx:]:
        if header_re.search(L):
            if not seen_header:
                cleaned.append(L)
                seen_header = True
            else:
                # skip duplicate header
                continue
        else:
            cleaned.append(L)
    try:
        with open(csv_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.writelines(cleaned)
    except Exception:
        return False
    return True

def parse_csv_rows(csv_path: str):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)
    delim, _ = detect_csv_delimiter_and_header(csv_path)
    rows = []
    header = None
    with open(csv_path, newline='', encoding='utf-8', errors='ignore') as csvfile:
        reader = csv.reader((line for line in csvfile if not line.startswith('#') and line.strip()), delimiter=delim)
        for i, row in enumerate(reader):
            if i == 0:
                header = [h.strip() for h in row]
                continue
            if len(row) < len(header):
                row += [''] * (len(header) - len(row))
            rec = {header[j]: row[j].strip() for j in range(len(header))}
            rows.append(rec)
    return header, rows

def extract_metric_values(rows):
    if not rows:
        return {}
    sample_keys = list(rows[0].keys())
    metric_col = None
    value_col = None
    for k in sample_keys:
        kl = k.lower()
        if 'metric name' == kl or kl == 'metricname' or 'metric name' in kl:
            metric_col = k
        if 'metric value' == kl or kl == 'metricvalue' or 'metric value' in kl:
            value_col = k
    if metric_col is None or value_col is None:
        if 'Metric Name' in rows[0]:
            metric_col = 'Metric Name'
        if 'Metric Value' in rows[0]:
            value_col = 'Metric Value'
    if metric_col is None or value_col is None:
        return {}

    def to_float(s):
        if s is None:
            return None
        s2 = s.strip()
        if s2 == '':
            return None
        s2 = s2.replace(',', '')
        if s2.endswith('%'):
            try:
                return float(s2[:-1])
            except:
                return None
        try:
            return float(s2)
        except:
            return None

    metric_map = defaultdict(list)
    for r in rows:
        m = r.get(metric_col, '').strip()
        v = r.get(value_col, '').strip()
        fv = to_float(v)
        if fv is not None:
            metric_map[m].append(fv)
    return metric_map

def aggregate_metric_map(metric_map, agg='mean'):
    agg_map = {}
    for k, vals in metric_map.items():
        if not vals:
            agg_map[k] = ""
            continue
        if agg == 'mean':
            agg_map[k] = sum(vals) / len(vals)
        elif agg == 'sum':
            agg_map[k] = sum(vals)
        elif agg == 'max':
            agg_map[k] = max(vals)
        else:
            agg_map[k] = sum(vals) / len(vals)
    return agg_map

def remove_empty_file(path: str) -> bool:
    try:
        if os.path.isfile(path) and os.path.getsize(path) == 0:
            os.remove(path)
            return True
    except Exception:
        pass
    return False

def extract_offending_metric_from_text(text: str):
    if not text:
        return None
    m = re.search(r'Failed to find metric regex:\^([A-Za-z0-9_]+)\.', text)
    if m:
        return m.group(1)
    m2 = re.search(r'Failed to find metric regex:.*?([A-Za-z0-9_]+)\\?\.', text)
    if m2:
        return m2.group(1)
    m3 = re.search(r'Failed to find metric.*?([A-Za-z0-9_]+)', text)
    if m3:
        return m3.group(1)
    return None

def find_best_match_for_requested(requested_normalized: str, metric_keys: List[str]):
    for k in metric_keys:
        kn = re.sub(r'[^a-z0-9]', '', k.lower())
        if kn == requested_normalized:
            return k
    for k in metric_keys:
        kn = re.sub(r'[^a-z0-9]', '', k.lower())
        if requested_normalized in kn:
            return k
    if 'sm' in requested_normalized or 'smeff' in requested_normalized or 'smefficiency' in requested_normalized:
        for k in metric_keys:
            kl = k.lower()
            if 'compute' in kl and 'sm' in kl:
                return k
            if 'compute' in kl and 'throughput' in kl:
                return k
            if 'sm efficiency' in kl or 'sm_efficiency' in kl:
                return k
    if 'occup' in requested_normalized:
        for k in metric_keys:
            if 'occup' in k.lower():
                return k
    return None

def choose_csv_from_dir(bench_out_dir: str, bench_name: str):
    candidate = os.path.join(bench_out_dir, bench_name + ".csv")
    if os.path.isfile(candidate):
        return candidate
    alt = os.path.join(bench_out_dir, bench_name + "-0.csv")
    if os.path.isfile(alt):
        return alt
    for fn in os.listdir(bench_out_dir):
        if fn.startswith(bench_name) and fn.endswith('.csv'):
            return os.path.join(bench_out_dir, fn)
    return None

# -----------------------
# ncu invocation helper
# -----------------------
def run_ncu_to_files(cmd: List[str], cwd: str, csv_path: str, log_path: str, timeout: int, env: dict):
    safe_mkdir(os.path.dirname(csv_path))
    safe_mkdir(os.path.dirname(log_path))
    start = time.time()
    with open(csv_path, 'w', encoding='utf-8', errors='ignore') as fout, \
         open(log_path, 'w', encoding='utf-8', errors='ignore') as ferr:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=fout, stderr=ferr)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            elapsed = time.time() - start
            return proc.returncode or -9, elapsed
    elapsed = time.time() - start
    return proc.returncode, elapsed

# -----------------------
# subset.json -> bench list construction
# -----------------------
def load_subset_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_fails(path: str) -> List[str]:
    if not os.path.isfile(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]

def build_bench_list(requested: List[str], benchmarks: dict, fails: List[str]) -> List[Tuple[str, str, List[str]]]:
    """
    Returns list of tuples:
      (bench_name, bench_dir_name, param_set)
    bench_name is unique output name (may include param suffix).
    bench_dir_name is the directory name under src (e.g., 'floydwarshall-cuda').
    param_set is list of runtime args (may be []).
    """
    benches = []
    for r in requested:
        if r in ['sycl', 'cuda', 'hip']:
            backend = r
            for k, v in benchmarks.items():
                bench_dir_name = f"{k}-{backend}"
                if bench_dir_name in fails:
                    continue
                res_regex = v[0]
                run_args_list = v[1] if len(v) > 1 else []
                if run_args_list and isinstance(run_args_list[0], list):
                    for param_set in run_args_list:
                        key = '_'.join(str(x) for x in param_set[:3])
                        bench_name = f"{bench_dir_name}-{key}"
                        benches.append((bench_name, bench_dir_name, param_set, res_regex))
                else:
                    bench_name = bench_dir_name
                    benches.append((bench_name, bench_dir_name, run_args_list, res_regex))
        else:
            # specific benchmark requested; may include backend and params.
            found = False
            for k, v in benchmarks.items():
                if r == k:
                    # user provided just base name - ambiguous: pick default backend? skip
                    continue
                if r.startswith(k + '-'):
                    suffix = r[len(k)+1:]
                    parts = suffix.split('-')
                    backend = parts[0]
                    bench_dir_name = f"{k}-{backend}"
                    if bench_dir_name in fails:
                        found = True
                        break
                    res_regex = v[0]
                    run_args_list = v[1] if len(v) > 1 else []
                    if run_args_list and isinstance(run_args_list[0], list):
                        # suffix may include param suffix after backend
                        if len(parts) > 1:
                            params = '-'.join(parts[1:])
                            for param_set in run_args_list:
                                expected_params = '_'.join(str(p) for p in param_set[:3])
                                if expected_params == params:
                                    bench_name = f"{bench_dir_name}-{params}"
                                    benches.append((bench_name, bench_dir_name, param_set, res_regex))
                                    found = True
                                    break
                        else:
                            # no param suffix -> take first param set
                            param_set = run_args_list[0]
                            key = '_'.join(str(x) for x in param_set[:3])
                            bench_name = f"{bench_dir_name}-{key}"
                            benches.append((bench_name, bench_dir_name, param_set, res_regex))
                            found = True
                    else:
                        # simple single-arg run
                        benches.append((r, bench_dir_name, run_args_list, res_regex))
                        found = True
                    break
            if not found:
                raise ValueError(f"Unknown benchmark or malformed name: {r}")
    return benches

# -----------------------
# main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description='run_ncu_bench (HeCBench-style subset.json runner + ncu)')
    parser.add_argument('--bench', '-B', nargs='+', required=True,
                        help='Either backend group (sycl/cuda/hip) or specific benchmark name(s) (can include backend and param suffix)')
    parser.add_argument('--subset-json', default=os.path.join('benchmarks','subset.json'),
                        help='Path to subset.json')
    parser.add_argument('--bench-dir', default=None, help='Root dir containing bench folders (e.g., /path/to/HeCBench/src)')
    parser.add_argument('--bench-fails', default=os.path.join('benchmarks','subset-fails.txt'),
                        help='File listing bench names to skip (one per line)')
    parser.add_argument('--ncu-binary', default='ncu')
    parser.add_argument('--ncu-set', default='basic')
    parser.add_argument('--metrics', default='sm_efficiency,achieved_occupancy,inst_executed,dram_read_throughput,flop_count_sp')
    parser.add_argument('--use-metrics', action='store_true', help='Pass --metrics to ncu (disabled by default)')
    parser.add_argument('--ncu-args', default='', help='Extra args to pass to ncu')
    parser.add_argument('--ncu-out', default='reports/ncu')
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--tmpdir', default='/mnt/disk3/yusheng/tmp_ncu')
    parser.add_argument('--launch-skip', type=int, default=0)
    parser.add_argument('--launch-count', type=int, default=1)
    parser.add_argument('--keep-logs', action='store_true', help='Keep .ncu.log files even if empty')
    parser.add_argument('--summary', default='ncu_summary.csv', help='Path to summary CSV to write')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    subset_json = args.subset_json
    if not os.path.isabs(subset_json):
        subset_json = os.path.join(script_dir, subset_json)
    if not os.path.isfile(subset_json):
        print("subset.json not found at", subset_json, file=sys.stderr)
        sys.exit(1)

    benchmarks = load_subset_json(subset_json)
    fails = load_fails(args.bench_fails if os.path.isabs(args.bench_fails) else os.path.join(script_dir, args.bench_fails))

    # build bench list
    try:
        benches = build_bench_list(args.bench, benchmarks, fails)
    except Exception as e:
        print("Failed to build benchmark list:", e, file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.join(args.ncu_out, args.ncu_set)
    safe_mkdir(out_dir)
    default_tmp = os.path.expanduser(args.tmpdir)
    safe_mkdir(default_tmp)
    try:
        os.chmod(default_tmp, 0o700)
    except Exception:
        pass

    extra_args = shlex.split(args.ncu_args) if args.ncu_args else []
    requested_metrics = [m.strip() for m in args.metrics.split(',') if m.strip()]

    summary_rows = []

    for bench_name, bench_dir_name, param_set, res_regex in benches:
        print("="*80)
        print("Profiling:", bench_name, "params:", param_set)
        # locate bench path
        if args.bench_dir:
            bench_path = os.path.realpath(os.path.join(args.bench_dir, bench_dir_name))
        else:
            bench_path = os.path.realpath(os.path.join(script_dir, '..', bench_dir_name))
        if not os.path.isdir(bench_path):
            print("Benchmark path not found:", bench_path, " skipping.", file=sys.stderr)
            continue
        print("Using benchmark path:", bench_path)

        bench_out_dir = os.path.join(out_dir, bench_dir_name, bench_name)
        safe_mkdir(bench_out_dir)

        exec_path = os.path.join(bench_path, 'main')
        if not os.path.isfile(exec_path):
            print("Executable not found:", exec_path, " skipping.", file=sys.stderr)
            continue
        if not os.access(exec_path, os.X_OK):
            try:
                os.chmod(exec_path, os.stat(exec_path).st_mode | 0o111)
            except Exception as e:
                print("Failed to set executable bit:", e, file=sys.stderr)
                continue

        per_tmp = os.path.join(default_tmp, bench_name)
        safe_mkdir(per_tmp)
        try:
            os.chmod(per_tmp, 0o700)
        except Exception:
            pass
        env = os.environ.copy()
        env['TMPDIR'] = per_tmp
        env['XDG_RUNTIME_DIR'] = per_tmp

        # prepare command
        current_metrics = requested_metrics.copy() if args.use_metrics else []
        max_metric_retries = max(1, len(current_metrics) + 1)
        attempt_num = 0
        succeeded = False
        run_elapsed = None

        csv_path = os.path.join(bench_out_dir, bench_name + ".csv")
        log_path = os.path.join(bench_out_dir, bench_name + ".ncu.log")

        while attempt_num < max_metric_retries and not succeeded:
            attempt_num += 1
            cmd = [args.ncu_binary, '--set', args.ncu_set, '--csv']
            if current_metrics:
                cmd += ['--metrics', ','.join(current_metrics)]
            if args.launch_skip:
                cmd += ['--launch-skip', str(args.launch_skip)]
            if args.launch_count:
                cmd += ['--launch-count', str(args.launch_count)]
            if extra_args:
                cmd += extra_args
            cmd += ['--', exec_path] + [str(x) for x in param_set]

            print(f"[attempt {attempt_num}/{max_metric_retries}] {' '.join(shlex.quote(c) for c in cmd)}")
            rc, elapsed = run_ncu_to_files(cmd, cwd=bench_out_dir, csv_path=csv_path, log_path=log_path, timeout=args.timeout, env=env)
            run_elapsed = elapsed
            print(f"ncu rc={rc} elapsed={elapsed:.1f}s")

            # read both log and csv text
            log_text = ''
            csv_text = ''
            try:
                if os.path.isfile(log_path):
                    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                        log_text = f.read()
            except Exception:
                log_text = ''
            try:
                if os.path.isfile(csv_path):
                    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                        csv_text = f.read()
            except Exception:
                csv_text = ''

            offending = extract_offending_metric_from_text(log_text) or extract_offending_metric_from_text(csv_text)
            if offending and current_metrics:
                print("ncu reported missing metric regex for:", offending)
                before = list(current_metrics)
                current_metrics = [m for m in current_metrics if not m.startswith(offending)]
                removed = [m for m in before if m not in current_metrics]
                print("Removed offending metric(s):", removed)
                try:
                    if os.path.isfile(csv_path):
                        os.remove(csv_path)
                except Exception:
                    pass
                continue

            combined_text = log_text + "\n" + csv_text
            if rc != 0:
                if 'Failed to profile' in combined_text or 'application returned an error code' in combined_text or 'The application returned an error code' in combined_text:
                    print("ncu/log indicates profiling failure or application error; not retrying further.")
                    break

            if os.path.isfile(csv_path):
                ok = clean_csv_if_needed(csv_path)
                if ok:
                    print("CSV captured and cleaned at", csv_path)
                    succeeded = True
                    break
                else:
                    print("CSV present but no CSV header found (likely only profiler preamble/errors).")
                    found_csv = choose_csv_from_dir(bench_out_dir, bench_name)
                    if found_csv and found_csv != csv_path:
                        print("Found alternate CSV:", found_csv)
                        if clean_csv_if_needed(found_csv):
                            csv_path = found_csv
                            succeeded = True
                            break
                    try:
                        os.remove(csv_path)
                    except Exception:
                        pass
                    continue
            else:
                print("No CSV file produced by this run. See log:", log_path if os.path.isfile(log_path) else "(no log)")
                continue

        if not succeeded:
            print("Cannot find/clean CSV for", bench_name, " - see", bench_out_dir)
            if not args.keep_logs:
                remove_empty_file(log_path)
            continue

        # parse CSV and aggregate
        try:
            header, rows = parse_csv_rows(csv_path)
            metric_map = extract_metric_values(rows)
            metric_agg = aggregate_metric_map(metric_map, agg='mean')
        except Exception as e:
            print("Failed to parse CSV", csv_path, ":", e, file=sys.stderr)
            continue

        # duration from CSV (ns -> s) preference
        duration_seconds = None
        for k, v in metric_agg.items():
            if 'duration' in k.lower():
                try:
                    duration_seconds = float(v) / 1e9
                    break
                except Exception:
                    duration_seconds = None

        out_rec = {
            'benchmark': bench_name,
            'params': "_".join(str(x) for x in param_set),
            'ncu_csv': csv_path,
            'ncu_set': args.ncu_set,
            'ncu_time_s': round(duration_seconds, 6) if duration_seconds is not None else '',
            'ncu_time_s_wallclock': round(run_elapsed or 0.0, 3)
        }

        # match requested metrics -> CSV canonical metric names (use canonical name in summary)
        requested = [m.strip() for m in args.metrics.split(',') if m.strip()]
        metric_keys = list(metric_agg.keys())
        for req in requested:
            rn = re.sub(r'[^a-z0-9]', '', req.lower())
            matched = find_best_match_for_requested(rn, metric_keys)
            if matched:
                out_rec[matched] = metric_agg.get(matched, "")
            else:
                # keep requested label if no match
                out_rec[req] = ""

        summary_rows.append(out_rec)
        if not args.keep_logs:
            remove_empty_file(log_path)

    # write summary (union of keys across rows)
    if summary_rows:
        all_keys = set()
        for r in summary_rows:
            all_keys.update(r.keys())
        preferred = ['benchmark','params','ncu_csv','ncu_set','ncu_time_s','ncu_time_s_wallclock']
        other_keys = sorted([k for k in all_keys if k not in preferred])
        fieldnames = [k for k in preferred if k in all_keys] + other_keys
        with open(args.summary, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in summary_rows:
                row = {k: r.get(k, "") for k in fieldnames}
                writer.writerow(row)
        print("Wrote summary to", args.summary)
    else:
        print("No successful runs to summarize.")

if __name__ == '__main__':
    main()