#!/usr/bin/env python3
"""
Runner to profile a benchmark family defined in benchmarks/subset.json using ncu.

Updates requested:
- By default do NOT pass --metrics to ncu (some ncu versions/devices don't support the metric names).
  Use --use-metrics to explicitly enable passing a metrics list.
- If a requested metric (like "sm_efficiency") is not present, try to derive it from CSV metrics
  (e.g. map "sm_efficiency" -> "Compute (SM) Throughput" when available).
- Use the CSV "Duration" metric (ns) as ncu_time_s (converted to seconds). Keep the wallclock
  elapsed as ncu_time_s_wallclock for reference.
- When writing summary, prefer the original CSV metric name for the column header when we matched
  a requested metric to a CSV metric (so the summary uses the CSV's canonical metric labels).
- Build summary CSV fieldnames as the union of keys across all rows (so different runs with
  different available metrics are handled).
- Keep previous behavior: per-testcase CSV under --ncu-out/<set>/<bench>/<benchname>/<benchname>.csv,
  optional --keep-logs, automatic cleaning of preamble before the CSV header, retries on missing metrics.

Usage example (no metrics passed to ncu):
TMPDIR=/mnt/disk3/yusheng/tmp_ncu XDG_RUNTIME_DIR=/mnt/disk3/yusheng/tmp_ncu \
./run_ncu_bench.py --bench floydwarshall --bench-dir /mnt/disk3/yusheng/HeCBench/src \
  --ncu-binary ncu --ncu-set basic --launch-skip 0 --launch-count 1 \
  --ncu-out reports/ncu --metrics "sm_efficiency,achieved_occupancy" --keep-logs

To force ncu to be invoked with --metrics (if you want and your version supports them):
add the flag --use-metrics.
"""
import os
import json
import subprocess
import argparse
import csv
import time
import shlex
import re
from collections import defaultdict

def safe_mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def load_entry(subset_json, bench_base):
    with open(subset_json) as f:
        d = json.load(f)
    if bench_base not in d:
        raise KeyError(f"{bench_base} not found in {subset_json}")
    entry = d[bench_base]
    res_regex = entry[0]
    param_list = entry[1] if len(entry) > 1 else []
    return res_regex, param_list

def detect_csv_delimiter_and_header(path):
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
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(sample)
            delim = dialect.delimiter
        except Exception:
            delim = ','
        header_line = sample
    return delim, header_line

def clean_csv_if_needed(csv_path):
    header_re = re.compile(r'^\s*"ID"\s*,\s*"Process ID"', re.IGNORECASE)
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

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
                continue
        else:
            cleaned.append(L)

    with open(csv_path, 'w', encoding='utf-8', errors='ignore') as f:
        f.writelines(cleaned)
    return True

def parse_csv_rows(csv_path):
    """
    Read CSV rows (skipping comment lines) and return header list and row dicts.
    """
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
    """
    From parsed CSV rows, build a mapping metric_name -> list of numeric values.
    Handles 'Metric Name' and 'Metric Value' columns (case-insensitive search).
    Cleans numeric strings (removes commas and percent signs).
    """
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
        # fallback to common names
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

def find_bench_dir(script_dir, bench_base, backend=None, bench_dir_arg=None):
    candidates = []
    if bench_dir_arg:
        root = os.path.realpath(bench_dir_arg)
        if backend:
            candidates.append(os.path.join(root, f"{bench_base}-{backend}"))
        candidates.append(os.path.join(root, bench_base))
        try:
            for name in os.listdir(root):
                if name.startswith(bench_base):
                    candidates.append(os.path.join(root, name))
        except FileNotFoundError:
            pass
    parent = os.path.realpath(os.path.join(script_dir, '..'))
    if backend:
        candidates.append(os.path.join(parent, f"{bench_base}-{backend}"))
    candidates.append(os.path.join(parent, bench_base))
    try:
        for name in os.listdir(parent):
            if name.startswith(bench_base):
                candidates.append(os.path.join(parent, name))
    except FileNotFoundError:
        pass
    for c in candidates:
        if os.path.isdir(c):
            return c
    cwd = os.getcwd()
    for name in os.listdir(cwd):
        if name.startswith(bench_base) and os.path.isdir(os.path.join(cwd, name)):
            return os.path.join(cwd, name)
    raise FileNotFoundError("Benchmark path not found for base '{}'. Tried: {}".format(bench_base, ", ".join(candidates)))

def run_ncu_to_files(cmd, cwd, csv_path, log_path, timeout, env):
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

def choose_csv_from_dir(bench_out_dir, bench_name):
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

def remove_empty_file(path):
    try:
        if os.path.isfile(path) and os.path.getsize(path) == 0:
            os.remove(path)
            return True
    except Exception:
        pass
    return False

def extract_offending_metric_from_text(text):
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

def find_best_match_for_requested(requested_normalized, metric_keys):
    """
    Find the best metric key from metric_keys that matches the normalized requested token.
    Returns the matched metric key or None.
    """
    # exact normalized match
    for k in metric_keys:
        kn = re.sub(r'[^a-z0-9]', '', k.lower())
        if kn == requested_normalized:
            return k
    # substring match
    for k in metric_keys:
        kn = re.sub(r'[^a-z0-9]', '', k.lower())
        if requested_normalized in kn:
            return k
    # Some heuristic mappings for sm_efficiency -> look for compute/sm throughput metrics
    if 'sm' in requested_normalized or 'smeff' in requested_normalized or 'smefficiency' in requested_normalized:
        for k in metric_keys:
            kl = k.lower()
            if 'compute' in kl and 'sm' in kl:
                return k
            if 'compute' in kl and 'throughput' in kl:
                return k
            if 'sm efficiency' in kl or 'sm_efficiency' in kl or 'sm efficiency' in kl:
                return k
    # occupancy -> Achieved Occupancy
    if 'occup' in requested_normalized:
        for k in metric_keys:
            if 'occup' in k.lower():
                return k
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', default='convolution3D')
    parser.add_argument('--backend', choices=['cuda','hip','sycl','omp'], default='cuda')
    parser.add_argument('--subset-json', default=os.path.join('benchmarks','subset.json'))
    parser.add_argument('--bench-dir', default=None)
    parser.add_argument('--ncu-set', default='basic')
    parser.add_argument('--metrics', default='sm_efficiency,achieved_occupancy,inst_executed,dram_read_throughput,flop_count_sp')
    parser.add_argument('--use-metrics', action='store_true', help='Pass --metrics to ncu (disabled by default)')
    parser.add_argument('--ncu-out', default='reports/ncu')
    parser.add_argument('--summary', default='ncu_summary.csv')
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--ncu-binary', default='ncu')
    parser.add_argument('--ncu-args', default='')
    parser.add_argument('--tmpdir', default='/mnt/disk3/yusheng/tmp_ncu')
    parser.add_argument('--launch-skip', type=int, default=0)
    parser.add_argument('--launch-count', type=int, default=1)
    parser.add_argument('--keep-logs', action='store_true', help='Keep .ncu.log files even if empty (useful for debugging)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    subset_json = args.subset_json
    if not os.path.isabs(subset_json):
        subset_json = os.path.join(script_dir, subset_json)

    res_regex, param_list = load_entry(subset_json, args.bench)

    out_dir = os.path.join(args.ncu_out, args.ncu_set, args.bench)
    safe_mkdir(out_dir)

    default_tmp = os.path.expanduser(args.tmpdir)
    safe_mkdir(default_tmp)
    try:
        os.chmod(default_tmp, 0o700)
    except Exception:
        pass

    benches = []
    for p in param_list:
        key = "_".join(str(x) for x in p[:3])
        bench_name = f"{args.bench}-{key}"
        benches.append( (bench_name, p) )

    summary_rows = []
    for bench_name, param_set in benches:
        print("="*80)
        print("Profiling:", bench_name, "params:", param_set)
        bench_path = None
        if args.bench_dir:
            candidate = os.path.join(os.path.realpath(args.bench_dir), f"{args.bench}-{args.backend}")
            if os.path.isdir(candidate):
                bench_path = candidate
        if not bench_path:
            try:
                bench_path = find_bench_dir(script_dir, args.bench, backend=args.backend, bench_dir_arg=args.bench_dir)
            except FileNotFoundError as e:
                print("Benchmark path not found:", e)
                continue
        print("Using benchmark path:", bench_path)

        bench_out_dir = os.path.join(out_dir, bench_name)
        safe_mkdir(bench_out_dir)

        exec_path = os.path.join(bench_path, 'main')
        if not os.path.isfile(exec_path):
            print("Executable not found:", exec_path)
            continue
        if not os.access(exec_path, os.X_OK):
            try:
                os.chmod(exec_path, os.stat(exec_path).st_mode | 0o111)
            except Exception as e:
                print("Failed to set executable bit:", e)
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

        # Use metrics only if explicitly requested
        current_metrics = [m.strip() for m in args.metrics.split(',') if m.strip()] if args.use_metrics else []
        extra_args = shlex.split(args.ncu_args) if args.ncu_args else []

        csv_path = os.path.join(bench_out_dir, bench_name + ".csv")
        log_path = os.path.join(bench_out_dir, bench_name + ".ncu.log")

        # If metrics are requested (use-metrics) we still handle offending-metric removal logic
        max_metric_retries = max(1, len(current_metrics) + 1)
        attempt_num = 0
        succeeded = False
        run_elapsed = None

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

            print(f"[attempt {attempt_num}/{max_metric_retries}] running: {' '.join(shlex.quote(c) for c in cmd)}")
            rc, elapsed = run_ncu_to_files(cmd, cwd=bench_out_dir, csv_path=csv_path, log_path=log_path, timeout=args.timeout, env=env)
            run_elapsed = elapsed
            print(f"ncu rc={rc} elapsed={elapsed:.1f}s")

            # read both log and csv text (if exist)
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

            # If csv exists, attempt to clean (strip preamble) and accept if successful
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

        # parse CSV rows and extract per-metric aggregates
        try:
            header, rows = parse_csv_rows(csv_path)
            metric_map = extract_metric_values(rows)
            metric_agg = aggregate_metric_map(metric_map, agg='mean')
        except Exception as e:
            print("Failed to parse CSV", csv_path, ":", e)
            continue

        # determine kernel duration from CSV (Duration metadata typically in ns)
        duration_seconds = None
        for k, v in metric_agg.items():
            if 'duration' in k.lower():
                try:
                    # metric_agg stores mean values; ncu Duration is in ns -> convert to seconds
                    duration_seconds = float(v) / 1e9
                    break
                except Exception:
                    duration_seconds = None
        # build output record: set ncu_time_s to kernel duration (per request) and keep wallclock
        out_rec = {
            'benchmark': bench_name,
            'params': "_".join(str(x) for x in param_set),
            'ncu_csv': csv_path,
            'ncu_set': args.ncu_set,
            'ncu_time_s': round(duration_seconds, 6) if duration_seconds is not None else '',
            'ncu_time_s_wallclock': round(run_elapsed or 0.0, 3)
        }

        # Match requested metrics against metric_agg keys, preferring to use CSV's canonical metric name
        requested = [m.strip() for m in args.metrics.split(',') if m.strip()]
        metric_keys = list(metric_agg.keys())

        # For each requested token, find best match in CSV metrics. If matched, use the CSV metric name as
        # the summary column header (so summary keeps CSV's canonical labels).
        for req in requested:
            rn = re.sub(r'[^a-z0-9]', '', req.lower())
            matched = find_best_match_for_requested(rn, metric_keys)
            if matched:
                # put aggregated value under the CSV metric name
                out_rec[matched] = metric_agg.get(matched, "")
            else:
                # fallback: put empty column with requested label (user may want it)
                out_rec[req] = ""

        summary_rows.append(out_rec)
        # remove empty log after success unless keep-logs requested
        if not args.keep_logs:
            remove_empty_file(log_path)

    # Write summary: union of all keys across all rows (ensures columns exist even if not present in first row)
    if summary_rows:
        all_keys = set()
        for r in summary_rows:
            all_keys.update(r.keys())
        # prefer a sane column order for common fields
        preferred = ['benchmark','params','ncu_csv','ncu_set','ncu_time_s','ncu_time_s_wallclock']
        other_keys = sorted([k for k in all_keys if k not in preferred])
        fieldnames = [k for k in preferred if k in all_keys] + other_keys
        with open(args.summary, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in summary_rows:
                # ensure all keys exist in row
                row = {k: r.get(k, "") for k in fieldnames}
                writer.writerow(row)
        print("Wrote summary to", args.summary)
    else:
        print("No successful runs to summarize.")

if __name__ == '__main__':
    main()