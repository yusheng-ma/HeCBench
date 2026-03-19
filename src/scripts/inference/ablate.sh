# 1. Run BASELINE (no NCU feedback in Round 2)
python deepseek_r1_multiround_ncu_retry.py --run_name baseline_v1

# 2. Run WITH NCU feedback
python deepseek_r1_multiround_ncu_retry.py --run_name ncu_v1 --use_ncu

# 3. Compare results with jq (JSONL is line-delimited, easy to filter)
echo "=== BASELINE ==="
jq -r 'select(.performance.improvement_pct != null) | "\(.timestamp)\tR1:\(.round1.duration_us)μs → R2:\(.round2.duration_us)μs (\(.performance.improvement_pct)%)"' experiment_logs/baseline_v1.jsonl

echo "=== WITH NCU ==="
jq -r 'select(.performance.improvement_pct != null) | "\(.timestamp)\tR1:\(.round1.duration_us)μs → R2:\(.round2.duration_us)μs (\(.performance.improvement_pct)%)"' experiment_logs/ncu_v1.jsonl

# 4. Aggregate stats across all runs of same type
echo "Baseline avg improvement:"
jq -s '[.[] | select(.performance.improvement_pct != null) | .performance.improvement_pct] | add/length' experiment_logs/baseline_v1.jsonl

echo "NCU avg improvement:"
jq -s '[.[] | select(.performance.improvement_pct != null) | .performance.improvement_pct] | add/length' experiment_logs/ncu_v1.jsonl