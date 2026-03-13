#!/usr/bin/env python3
"""
Simple extractor for ncu CSV reports.

Produces a tidy summary CSV with exactly these columns (in this order):
  benchmark,params,ncu_time_ns,Achieved Occupancy,Compute (SM) Throughput,DRAM Throughput,Memory Throughput

Behavior:
- Scans <reports>/<set>/ for .csv files (e.g. reports/ncu/basic).
- For each CSV:
  - Cleans preamble if present and parses rows.
  - Aggregates metrics per (kernel, metric) and computes total Duration per kernel to pick the "hottest" kernel.
  - For each requested metric token (default the four above) finds the best matching kernel||metric and uses its aggregated value.
  - Sets ncu_time_ns from the hottest kernel's mean Duration (ns) when available.
- Writes a single summary CSV with one row per input CSV and exactly the requested columns.
- Numeric outputs are rounded to 2 decimal places.

Usage:
  python3 extract_reports_simple.py --reports reports/ncu --set basic --out ncu_summary_simple.csv

You can change which short metrics to extract with --metrics (comma-separated).
"""
from __future__ import annotations
import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from typing import List, Dict, Tuple

# -----------------------
# small helpers
# -----------------------
def clean_csv_if_needed(csv_path: str) -> bool:
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
                continue
        else:
            cleaned.append(L)
    try:
        with open(csv_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.writelines(cleaned)
    except Exception:
        return False
    return True

def detect_delim_and_header(csv_path: str) -> Tuple[str, List[str]]:
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        sample = ''
        for _ in range(50):
            line = f.readline()
            if not line:
                break
            if not line.startswith('#') and line.strip():
                sample += line
                break
        if not sample:
            return ',', []
        try:
            dialect = csv.Sniffer().sniff(sample)
            delim = dialect.delimiter
        except Exception:
            delim = ','
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                header = [h.strip().strip('"') for h in next(csv.reader([line], delimiter=delim))]
                return delim, header
    return ',', []

def parse_csv_rows(csv_path: str) -> Tuple[List[str], List[Dict[str,str]]]:
    delim, header = detect_delim_and_header(csv_path)
    rows = []
    with open(csv_path, newline='', encoding='utf-8', errors='ignore') as csvfile:
        reader = csv.reader((line for line in csvfile if line.strip() and not line.startswith('#')), delimiter=delim)
        for i, row in enumerate(reader):
            if i == 0:
                header = [h.strip().strip('"') for h in row]
                continue
            if len(row) < len(header):
                row += [''] * (len(header) - len(row))
            rec = {header[j]: row[j].strip().strip('"') for j in range(len(header))}
            rows.append(rec)
    return header, rows

def extract_kernel_metric_values(rows: List[Dict[str,str]]):
    if not rows:
        return {}, {}
    keys = list(rows[0].keys())
    kcol = next((k for k in keys if 'kernel' in k.lower() and 'name' in k.lower()), None)
    mcol = next((k for k in keys if 'metric' in k.lower() and 'name' in k.lower()), None)
    vcol = next((k for k in keys if 'metric' in k.lower() and 'value' in k.lower()), None)
    if kcol is None and 'Kernel Name' in rows[0]:
        kcol = 'Kernel Name'
    if mcol is None and 'Metric Name' in rows[0]:
        mcol = 'Metric Name'
    if vcol is None and 'Metric Value' in rows[0]:
        vcol = 'Metric Value'
    if not (kcol and mcol and vcol):
        return {}, {}
    def to_float(s: str):
        if s is None or s == '':
            return None
        s2 = s.strip().replace(',','')
        if s2.endswith('%'):
            try:
                return float(s2[:-1])
            except:
                return None
        try:
            return float(s2)
        except:
            return None
    kernel_metric_map = defaultdict(list)
    for r in rows:
        kn = r.get(kcol,'').strip()
        mn = r.get(mcol,'').strip()
        mv = r.get(vcol,'').strip()
        val = to_float(mv)
        if val is None:
            continue
        key = f"{kn} || {mn}"
        kernel_metric_map[key].append(val)
    meta = {'kernel_col': kcol, 'metric_col': mcol, 'value_col': vcol}
    return kernel_metric_map, meta

def aggregate_kernel_metric_map(kernel_metric_map: Dict[str,List[float]]):
    agg_map = {}
    total_map = {}
    for k, vals in kernel_metric_map.items():
        if not vals:
            agg_map[k] = ""
            total_map[k] = 0.0
            continue
        agg_map[k] = sum(vals) / len(vals)
        total_map[k] = sum(vals)
    return agg_map, total_map

def normalize_token(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', s.lower() if s else '')

def find_matches_for_token(token: str, metric_keys: List[str]) -> List[str]:
    tok = normalize_token(token)
    matches = []
    for km in metric_keys:
        if ' || ' in km:
            _, mn = km.split(' || ', 1)
        else:
            mn = km
        mnorm = normalize_token(mn)
        if tok == mnorm or tok in mnorm or mnorm in tok:
            matches.append(km)
    # heuristics for common tokens
    if not matches:
        if 'sm' in tok or 'smeff' in tok:
            for km in metric_keys:
                if 'compute' in km.lower() and 'sm' in km.lower():
                    matches.append(km)
                elif 'compute' in km.lower() and 'throughput' in km.lower():
                    matches.append(km)
        if 'occup' in tok:
            for km in metric_keys:
                if 'occup' in km.lower():
                    matches.append(km)
        if 'dram' in tok and 'throughput' in tok:
            for km in metric_keys:
                if 'dram' in km.lower() and 'throughput' in km.lower():
                    matches.append(km)
        if 'memory' in tok and 'throughput' in tok:
            for km in metric_keys:
                if 'memory' in km.lower() and 'throughput' in km.lower():
                    matches.append(km)
    # deduplicate
    seen = set(); uniq = []
    for m in matches:
        if m not in seen:
            seen.add(m); uniq.append(m)
    return uniq

def find_csv_files(root: str, setname: str) -> List[str]:
    base = os.path.join(root, setname)
    if not os.path.isdir(base):
        return []
    out = []
    for dirpath, _, files in os.walk(base):
        for fn in files:
            if fn.endswith('.csv'):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)

def derive_benchmark_from_path(csv_path: str, setname: str) -> Tuple[str,str]:
    parts = os.path.normpath(csv_path).split(os.sep)
    if setname in parts:
        i = parts.index(setname)
        if len(parts) > i+3:
            bench_name = parts[i+2]
            if '-' in bench_name:
                suffix = bench_name.split('-')[-1]
                if any(c.isdigit() for c in suffix):
                    return bench_name, suffix
            return bench_name, ''
    fname = os.path.splitext(os.path.basename(csv_path))[0]
    return fname, ''

def format_num_for_output(value) -> str:
    """
    Format numeric value to 2 decimal places.
    Accepts float or numeric string. Returns empty string for '' or None.
    """
    if value is None or value == "":
        return ""
    try:
        # if already a float
        if isinstance(value, float) or isinstance(value, int):
            return f"{value:.2f}"
        # try parse string
        s = str(value).strip()
        if s == "":
            return ""
        s2 = s.replace(',', '')
        if s2.endswith('%'):
            s2 = s2[:-1]
        v = float(s2)
        return f"{v:.2f}"
    except Exception:
        # non-numeric, return original trimmed
        return str(value).strip()

# -----------------------
# main extraction
# -----------------------
def extract_simple(reports_root: str, setname: str, out_csv: str, requested_metrics: List[str]):
    csv_files = find_csv_files(reports_root, setname)
    if not csv_files:
        print("No CSV files found under", os.path.join(reports_root, setname), file=sys.stderr)
        return

    rows_out = []
    for csvf in csv_files:
        clean_csv_if_needed(csvf)
        try:
            header, rows = parse_csv_rows(csvf)
        except Exception as e:
            print("Failed to parse", csvf, ":", e, file=sys.stderr)
            continue
        if not rows:
            continue

        km_map, meta = extract_kernel_metric_values(rows)
        if not km_map:
            # nothing to pull
            continue
        km_agg, km_total = aggregate_kernel_metric_map(km_map)

        bench_label, params = derive_benchmark_from_path(csvf, setname)
        out_rec = {
            'benchmark': bench_label,
            'params': params,
            'ncu_time_ns': '',
        }

        # compute kernel durations per kernel to pick hottest kernel
        kernel_durations = {}
        for km_key, total in km_total.items():
            if ' || ' in km_key:
                kn, mn = km_key.split(' || ', 1)
            else:
                kn, mn = km_key, ''
            if 'duration' in mn.lower():
                kernel_durations[kn] = kernel_durations.get(kn, 0.0) + total
        hottest_kernel = None
        if kernel_durations:
            hottest_kernel = max(kernel_durations.items(), key=lambda x: x[1])[0]
        # set ncu_time_ns from hottest kernel Duration if present (mean value is stored in km_agg in ns)
        if hottest_kernel:
            dur_key = None
            for km in km_agg.keys():
                if km.startswith(hottest_kernel + ' || ') and 'duration' in km.lower():
                    dur_key = km; break
            if dur_key and km_agg.get(dur_key) != "":
                try:
                    out_rec['ncu_time_ns'] = km_agg[dur_key]
                except Exception:
                    out_rec['ncu_time_ns'] = ''

        metric_keys = list(km_agg.keys())
        # For each requested short metric, find best match and pick the value from the hottest-matching kernel
        for req in requested_metrics:
            matches = find_matches_for_token(req, metric_keys)
            chosen_val = ""
            if matches:
                # prefer matches on the hottest kernel if any
                if hottest_kernel:
                    hot_matches = [m for m in matches if m.startswith(hottest_kernel + ' || ')]
                    if hot_matches:
                        chosen_key = hot_matches[0]
                        chosen_val = km_agg.get(chosen_key, "")
                    else:
                        # otherwise pick the match whose kernel has largest total duration
                        def kernel_total_for_key(km_key):
                            if ' || ' in km_key:
                                kn, _ = km_key.split(' || ', 1)
                            else:
                                kn = ''
                            return kernel_durations.get(kn, 0.0)
                        matches.sort(key=lambda k: -kernel_total_for_key(k))
                        chosen_key = matches[0]
                        chosen_val = km_agg.get(chosen_key, "")
                else:
                    # no duration info: pick first match
                    chosen_key = matches[0]
                    chosen_val = km_agg.get(chosen_key, "")
            out_rec[req] = chosen_val
        rows_out.append(out_rec)

    # write summary with exact desired header order and round numeric fields to 2 decimals
    header = ['benchmark', 'params', 'ncu_time_ns'] + requested_metrics
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows_out:
            row = {}
            for k in header:
                val = r.get(k, "")
                # format numeric fields (ncu_time_ns and requested metrics)
                if k == 'ncu_time_ns' or k in requested_metrics:
                    row[k] = format_num_for_output(val)
                else:
                    row[k] = val
            writer.writerow(row)
    print("Wrote simple summary to", out_csv)

def main_cli():
    parser = argparse.ArgumentParser(description='Create a simple ncu summary CSV from reports')
    parser.add_argument('--reports', default='reports/ncu', help='reports base dir')
    parser.add_argument('--set', default='basic', help='ncu set subdir (e.g. basic)')
    parser.add_argument('--out', default='ncu_summary_simple.csv', help='output summary path')
    parser.add_argument('--metrics', default='Achieved Occupancy,Compute (SM) Throughput,DRAM Throughput,Memory Throughput',
                        help='Comma-separated short metric tokens (these will become column headers in this order)')
    args = parser.parse_args()

    requested_metrics = [m.strip() for m in args.metrics.split(',') if m.strip()]
    extract_simple(args.reports, args.set, args.out, requested_metrics)

if __name__ == '__main__':
    main_cli()