# 1. Baseline: minimal logging (default)
python deepseek_r1_multiround_ncu_retry.py --run_name baseline_v1

# 2. With NCU feedback + minimal logging
python deepseek_r1_multiround_ncu_retry.py --run_name ncu_v1 --use_ncu

# 3. Baseline + verbose logging for debugging extraction issues
python deepseek_r1_multiround_ncu_retry.py --run_name baseline_debug \
  --log_verbose --log_code_extract

# 4. Full debug: log everything (prompts, outputs, extraction, NCU)
python deepseek_r1_multiround_ncu_retry.py --run_name full_debug \
  --use_ncu --log_prompts --log_outputs --log_code_extract

# 5. Analyze metrics (JSONL is clean regardless of verbose flags)
jq -r 'select(.round2.duration_us != null) | "\(.config.run_name)\t\(.round2.duration_us)μs\t\(.performance.improvement_pct)%"' experiment_logs/*.jsonl