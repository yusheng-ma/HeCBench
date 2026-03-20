#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark Profiling Results Visualizer
解析並視覺化 FloydWarshall, Convolution3D, CrossEntropy, Softmax, Backprop 等benchmark的NCU profiling數據
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import re
from pathlib import Path

# 設置中文字體（可根據系統調整）
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

def parse_log_file(filepath):
    """解析JSON lines格式的log文件"""
    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    results.append(data)
                except json.JSONDecodeError:
                    continue
    return results

def extract_profiling_data(results):
    """提取profiling結果為DataFrame"""
    records = []
    
    for entry in results:
        benchmark = entry.get('benchmark', 'unknown')
        run_name = entry.get('run_name', 'unknown')
        config = entry.get('config', {})
        use_ncu_retry = config.get('use_ncu_retry', False)
        summary = entry.get('summary', {})
        generation = entry.get('generation', {})
        
        # 提取generation資訊
        for round_name, round_data in generation.items():
            records.append({
                'benchmark': benchmark,
                'run_name': run_name,
                'use_ncu_retry': use_ncu_retry,
                'timestamp': entry.get('timestamp', ''),
                'metric_type': f'generation_{round_name}',
                'inputs': 'N/A',
                'duration_us': round_data.get('duration_us', 0),
                'generation_time_us': round_data.get('generation_time_us', 0),
                'improvement_pct': round_data.get('improvement_pct', 0),
                'attempts': round_data.get('attempts', 0),
                'success': True,
                'metric_source': 'generation'
            })
        
        # 提取profiling_results
        for prof in entry.get('profiling_results', []):
            inputs = prof.get('inputs', [])
            records.append({
                'benchmark': benchmark,
                'run_name': run_name,
                'use_ncu_retry': use_ncu_retry,
                'timestamp': entry.get('timestamp', ''),
                'metric_type': 'kernel_duration',
                'inputs': '|'.join(map(str, inputs)) if inputs else 'N/A',
                'input_1': float(inputs[0]) if len(inputs) > 0 and inputs[0].isdigit() else None,
                'input_2': float(inputs[1]) if len(inputs) > 1 and inputs[1].isdigit() else None,
                'input_3': float(inputs[2]) if len(inputs) > 2 and inputs[2].isdigit() else None,
                'duration_us': prof.get('metric_value', 0),
                'metric_unit': prof.get('metric_unit', 'unknown'),
                'all_durations': prof.get('all_durations', []),
                'success': prof.get('success', False),
                'metric_source': prof.get('metric_source', 'unknown'),
                'error': prof.get('error', None)
            })
        
        # 提取summary
        records.append({
            'benchmark': benchmark,
            'run_name': run_name,
            'use_ncu_retry': use_ncu_retry,
            'timestamp': entry.get('timestamp', ''),
            'metric_type': 'summary_avg',
            'inputs': 'N/A',
            'duration_us': summary.get('avg_duration_us', 0),
            'total_tested': summary.get('total_inputs_tested', 0),
            'successful': summary.get('successful_profilings', 0),
            'success_rate': summary.get('successful_profilings', 0) / max(summary.get('total_inputs_tested', 1), 1),
            'success': True,
            'metric_source': 'summary'
        })
    
    return pd.DataFrame(records)

def plot_benchmark_comparison(df, output_dir='output'):
    """比較不同benchmark的平均kernel duration"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 過濾kernel duration數據
    kernel_df = df[(df['metric_type'] == 'kernel_duration') & (df['success'] == True)].copy()
    
    if kernel_df.empty:
        print("⚠️  沒有kernel duration數據可繪圖")
        return
    
    plt.figure(figsize=(12, 6))
    
    # 按benchmark和retry config分組
    grouped = kernel_df.groupby(['benchmark', 'use_ncu_retry'])['duration_us'].mean().unstack()
    
    grouped.plot(kind='bar', figsize=(12, 6), color=['#3498db', '#e74c3c'], edgecolor='black')
    plt.title('Average Kernel Duration by Benchmark', fontsize=14, fontweight='bold')
    plt.xlabel('Benchmark', fontsize=12)
    plt.ylabel('Duration (microseconds)', fontsize=12)
    plt.legend(title='use_ncu_retry', labels=['False', 'True'])
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/benchmark_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已儲存: {output_dir}/benchmark_comparison.png")

def plot_scaling_analysis(df, output_dir='output'):
    """分析input size對performance的影響"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    kernel_df = df[(df['metric_type'] == 'kernel_duration') & (df['success'] == True)].copy()
    
    benchmarks = kernel_df['benchmark'].unique()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, benchmark in enumerate(benchmarks[:5]):  # 最多顯示5個benchmark
        bench_df = kernel_df[kernel_df['benchmark'] == benchmark].copy()
        
        if bench_df.empty or bench_df['input_1'].isna().all():
            continue
            
        # 按input_1分組取平均
        scaling_df = bench_df.groupby('input_1')['duration_us'].agg(['mean', 'std']).reset_index()
        
        ax = axes[idx]
        ax.errorbar(scaling_df['input_1'], scaling_df['mean'], 
                   yerr=scaling_df['std'], 
                   marker='o', capsize=4, linewidth=2, markersize=6)
        ax.set_xlabel('Input Size (param 1)', fontsize=10)
        ax.set_ylabel('Duration (μs)', fontsize=10)
        ax.set_title(f'{benchmark}\nScaling Analysis', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    # 隱藏多餘的subplots
    for idx in range(len(benchmarks), 5):
        axes[idx].set_visible(False)
    
    plt.suptitle('Performance Scaling by Input Size (Log-Log Scale)', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scaling_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已儲存: {output_dir}/scaling_analysis.png")

def plot_retry_comparison(df, output_dir='output'):
    """比較use_ncu_retry=true/false的差異"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    summary_df = df[df['metric_type'] == 'summary_avg'].copy()
    
    if summary_df.empty:
        return
    
    # 準備數據
    comparison_data = []
    for benchmark in summary_df['benchmark'].unique():
        bench_df = summary_df[summary_df['benchmark'] == benchmark]
        for retry_val in [False, True]:
            subset = bench_df[bench_df['use_ncu_retry'] == retry_val]
            if not subset.empty:
                comparison_data.append({
                    'benchmark': benchmark,
                    'use_ncu_retry': retry_val,
                    'avg_duration': subset['duration_us'].mean(),
                    'success_rate': subset['success_rate'].mean() if 'success_rate' in subset.columns else 1.0
                })
    
    comp_df = pd.DataFrame(comparison_data)
    if comp_df.empty:
        return
    
    # 繪製雙軸圖
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Duration軸
    bar_width = 0.35
    x = np.arange(len(comp_df['benchmark'].unique()))
    
    retry_false = comp_df[comp_df['use_ncu_retry'] == False].set_index('benchmark')['avg_duration']
    retry_true = comp_df[comp_df['use_ncu_retry'] == True].set_index('benchmark')['avg_duration']
    
    benchmarks = list(retry_false.index.union(retry_true.index))
    
    ax1.bar(x - bar_width/2, [retry_false.get(b, 0) for b in benchmarks], 
           bar_width, label='Retry=False', color='#3498db', edgecolor='black', alpha=0.8)
    ax1.bar(x + bar_width/2, [retry_true.get(b, 0) for b in benchmarks], 
           bar_width, label='Retry=True', color='#2ecc71', edgecolor='black', alpha=0.8)
    
    ax1.set_xlabel('Benchmark', fontsize=12)
    ax1.set_ylabel('Average Duration (μs)', fontsize=12, color='#2c3e50')
    ax1.set_xticks(x)
    ax1.set_xticklabels(benchmarks, rotation=45, ha='right')
    ax1.tick_params(axis='y', labelcolor='#2c3e50')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Success rate軸
    ax2 = ax1.twinx()
    sr_false = comp_df[comp_df['use_ncu_retry'] == False].set_index('benchmark')['success_rate']
    sr_true = comp_df[comp_df['use_ncu_retry'] == True].set_index('benchmark')['success_rate']
    
    ax2.plot(x, [sr_false.get(b, 1) for b in benchmarks], 
            marker='s', color='#e74c3c', linewidth=2, label='Success Rate (False)', markersize=8)
    ax2.plot(x, [sr_true.get(b, 1) for b in benchmarks], 
            marker='^', color='#f39c12', linewidth=2, label='Success Rate (True)', markersize=8)
    
    ax2.set_ylabel('Success Rate', fontsize=12, color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    ax2.set_ylim(0, 1.1)
    
    # 合併圖例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.title('NCU Retry Configuration Comparison\nDuration & Success Rate', 
             fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/retry_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已儲存: {output_dir}/retry_comparison.png")

def plot_generation_stats(df, output_dir='output'):
    """視覺化generation階段的統計數據"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    gen_df = df[df['metric_type'].str.startswith('generation')].copy()
    
    if gen_df.empty:
        return
    
    # 提取round
    gen_df['round'] = gen_df['metric_type'].str.extract(r'generation_(round\d+)')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 圖1: Generation time by benchmark
    gen_time_pivot = gen_df.pivot_table(
        index='benchmark', 
        columns='round', 
        values='generation_time_us', 
        aggfunc='mean'
    ) / 1e6  # 轉換為秒
    
    gen_time_pivot.plot(kind='barh', ax=axes[0], color=['#9b59b6', '#3498db'], edgecolor='black')
    axes[0].set_xlabel('Generation Time (seconds)', fontsize=11)
    axes[0].set_title('Code Generation Time by Benchmark', fontsize=12, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3, linestyle='--')
    
    # 圖2: Improvement percentage
    improvement_df = gen_df[gen_df['round'] == 'round2'][['benchmark', 'use_ncu_retry', 'improvement_pct']].copy()
    
    for retry_val in [False, True]:
        subset = improvement_df[improvement_df['use_ncu_retry'] == retry_val]
        if not subset.empty:
            axes[1].plot(subset['benchmark'], subset['improvement_pct'], 
                        marker='o', label=f'Retry={retry_val}', 
                        linewidth=2, markersize=8)
    
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Benchmark', fontsize=11)
    axes[1].set_ylabel('Improvement (%)', fontsize=11)
    axes[1].set_title('Round2 vs Round1 Improvement', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3, linestyle='--')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.suptitle('Generation Stage Statistics', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/generation_stats.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已儲存: {output_dir}/generation_stats.png")

def plot_success_rate_heatmap(df, output_dir='output'):
    """成功率熱力圖"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    kernel_df = df[(df['metric_type'] == 'kernel_duration')].copy()
    
    if kernel_df.empty:
        return
    
    # 計算每個benchmark+retry的成功率
    success_data = kernel_df.groupby(['benchmark', 'use_ncu_retry'])['success'].mean().unstack()
    
    plt.figure(figsize=(10, 6))
    im = plt.imshow(success_data.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # 添加標籤
    plt.xticks(range(len(success_data.columns)), success_data.columns, fontsize=10)
    plt.yticks(range(len(success_data.index)), success_data.index, fontsize=10)
    
    # 添加數值標籤
    for i in range(len(success_data.index)):
        for j in range(len(success_data.columns)):
            val = success_data.values[i, j]
            if not np.isnan(val):
                plt.text(j, i, f'{val*100:.1f}%', ha='center', va='center', 
                        fontsize=9, color='black' if val > 0.5 else 'white', fontweight='bold')
    
    plt.colorbar(im, label='Success Rate')
    plt.xlabel('use_ncu_retry', fontsize=12, fontweight='bold')
    plt.ylabel('Benchmark', fontsize=12, fontweight='bold')
    plt.title('Profiling Success Rate Heatmap', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/success_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已儲存: {output_dir}/success_heatmap.png")

def generate_summary_report(df, output_dir='output'):
    """生成文字摘要報告"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    report = []
    report.append("=" * 70)
    report.append("BENCHMARK PROFILING SUMMARY REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)
    report.append("")
    
    # 基本統計
    report.append(f"📊 Total Records Analyzed: {len(df)}")
    report.append(f"🔬 Unique Benchmarks: {df['benchmark'].nunique()}")
    report.append(f"⏰ Time Range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    report.append("")
    
    # 各benchmark統計
    report.append("📈 BENCHMARK PERFORMANCE SUMMARY")
    report.append("-" * 50)
    
    summary_df = df[df['metric_type'] == 'summary_avg'].copy()
    kernel_df = df[(df['metric_type'] == 'kernel_duration') & (df['success'] == True)].copy()
    
    for benchmark in df['benchmark'].unique():
        bench_summary = summary_df[summary_df['benchmark'] == benchmark]
        bench_kernel = kernel_df[kernel_df['benchmark'] == benchmark]
        
        if not bench_summary.empty:
            avg_dur = bench_summary['duration_us'].mean()
            success_rate = bench_summary['success_rate'].mean() if 'success_rate' in bench_summary.columns else 1.0
            
            report.append(f"\n🔹 {benchmark.upper()}")
            report.append(f"   ├─ Avg Duration: {avg_dur:.3f} μs")
            report.append(f"   ├─ Success Rate: {success_rate*100:.1f}%")
            
            if not bench_kernel.empty:
                min_dur = bench_kernel['duration_us'].min()
                max_dur = bench_kernel['duration_us'].max()
                report.append(f"   ├─ Duration Range: {min_dur:.3f} ~ {max_dur:.3f} μs")
                report.append(f"   └─ Test Cases: {bench_kernel['inputs'].nunique()}")
    
    report.append("")
    report.append("⚙️  CONFIGURATION COMPARISON")
    report.append("-" * 50)
    
    for retry_val in [False, True]:
        subset = summary_df[summary_df['use_ncu_retry'] == retry_val]
        if not subset.empty:
            avg_overall = subset['duration_us'].mean()
            avg_success = subset['success_rate'].mean() if 'success_rate' in subset.columns else 1.0
            report.append(f"\n📌 use_ncu_retry={retry_val}")
            report.append(f"   ├─ Overall Avg Duration: {avg_overall:.3f} μs")
            report.append(f"   └─ Overall Success Rate: {avg_success*100:.1f}%")
    
    report.append("")
    report.append("=" * 70)
    
    # 寫入文件
    with open(f'{output_dir}/summary_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"✓ 已儲存: {output_dir}/summary_report.txt")
    print('\n'.join(report))  # 同時輸出到console

def main():
    """主函數"""
    import sys
    
    # 預設輸入輸出路徑
    input_file = 'profiling_log.jsonl' if len(sys.argv) < 2 else sys.argv[1]
    output_dir = 'visualization_output'
    
    print(f"🔍 讀取日誌文件: {input_file}")
    
    # 解析數據
    results = parse_log_file(input_file)
    if not results:
        print("❌ 未找到有效數據，請檢查輸入文件")
        return
    
    print(f"✓ 成功解析 {len(results)} 條記錄")
    
    # 提取為DataFrame
    df = extract_profiling_data(results)
    print(f"✓ 提取 {len(df)} 條可視覺化記錄")
    
    # 生成視覺化
    print("\n🎨 生成視覺化圖表...")
    plot_benchmark_comparison(df, output_dir)
    plot_scaling_analysis(df, output_dir)
    plot_retry_comparison(df, output_dir)
    plot_generation_stats(df, output_dir)
    plot_success_rate_heatmap(df, output_dir)
    
    # 生成摘要報告
    print("\n📝 生成摘要報告...")
    generate_summary_report(df, output_dir)
    
    print(f"\n✨ 完成！所有輸出已儲存至 '{output_dir}/' 目錄")
    print(f"📁 生成的文件:")
    for f in Path(output_dir).iterdir():
        print(f"   • {f.name}")

if __name__ == '__main__':
    main()