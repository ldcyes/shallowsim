#!/usr/bin/env python3
"""Regenerate disaggregated Groq-Rubin72 per-user results, plot breakdowns, and export xlsx."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shallowsim as sb

OUTPUT_DIR = './results'
PREFIX = 'groq_rubin72_400k_eval_nvfp4'
MODEL_CSV = './model.csv'
GPU_CSV = './device/gpu_info.csv'
SEQ_LEN = 400000

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Regenerate disaggregated results ──────────────────────
print("=" * 60)
print("Regenerating disaggregated per-user results ...")
print("=" * 60)

candidates_df, best_rows_df, feasible_summary_df = sb.generate_groq_rubin72_per_user_results(
    model_csv=MODEL_CSV,
    gpu_info_csv=GPU_CSV,
    output_dir=OUTPUT_DIR,
    output_prefix=PREFIX,
)

print(f"\nCandidates: {len(candidates_df)} rows")
print(f"Best rows:  {len(best_rows_df)} rows")
print(f"Feasible:   {len(feasible_summary_df)} rows")

if not feasible_summary_df.empty:
    print("\n--- Feasible Summary ---")
    print(feasible_summary_df.to_markdown(floatfmt='.3f'))

# ── 2. Plot per-model stage breakdown (existing) ────────────
if not best_rows_df.empty:
    print("\n" + "=" * 60)
    print("Plotting per-model stage stacked time ...")
    print("=" * 60)

    files = sb.plot_disagg_stage_stacked_time_by_model(
        best_rows_df,
        output_dir=OUTPUT_DIR,
        file_prefix=f'{PREFIX}_stage',
        model_csv=MODEL_CSV,
        width=12, height=7,
    )
    for f in files:
        print(f"  Saved: {f}")

    all_chart = os.path.join(OUTPUT_DIR, f'{PREFIX}_stage_all.png')
    sb.plot_disagg_stage_stacked_time(
        best_rows_df, all_chart,
        model_csv=MODEL_CSV, width=18, height=9,
    )
    print(f"  Saved: {all_chart}")

# ── 3. Per-layer latency breakdown analysis ──────────────────
print("\n" + "=" * 60)
print("Per-layer attention latency breakdown ...")
print("=" * 60)

args_dict = sb.get_model_args_dict(filename=MODEL_CSV)
gpu_dict = sb.get_gpu_info(filename=GPU_CSV, decoding_mode=True)
rubin = gpu_dict['Rubin-NVL72']

breakdown_rows = []
for model_name, args in args_dict.items():
    dense_layers = args.n_dense_layers
    sparse_layers = args.n_layers - dense_layers
    for tp in [1, 2, 4, 8]:
        for bs in [1, 4, 8, 16]:
            comp = sb.decode_attention_component_times(
                args, rubin, SEQ_LEN, bs, tp=tp, cp=1)
            row = {
                'Model': model_name,
                'TP': tp,
                'BS': bs,
                'DenseLayers': dense_layers,
                'SparseLayers': sparse_layers,
                'KVLoad_ms': comp['KVLoad_ms'],
                'WeightLoad_ms': comp['WeightLoad_ms'],
                'ComputeLowp_ms': comp['ComputeLowp_ms'],
                'ComputeQK_ms': comp['ComputeQK_ms'],
                'Compute_ms': comp['Compute_ms'],
                'KernelStatic_ms': comp['KernelStatic_ms'],
                'AllReduce_ms': comp['AllReduce_ms'],
                'DenseWeightLoad_ms': comp['DenseWeightLoad_ms'],
                'DenseCompute_ms': comp['DenseCompute_ms'],
                'DenseOverlap_ms': comp['DenseOverlap_ms'],
                'DenseAttnTotal_ms': comp['DenseOverlap_ms'] * dense_layers,
                'SparseWeightLoad_ms': comp['SparseWeightLoad_ms'],
                'SparseCompute_ms': comp['SparseCompute_ms'],
                'SparseOverlap_ms': comp['SparseOverlap_ms'],
                'SparseAttnTotal_ms': comp['SparseOverlap_ms'] * sparse_layers,
                'SparseMemTotal_ms': (comp['KVLoad_ms'] + comp['SparseWeightLoad_ms']),
            }
            # Bottleneck classification
            sparse_mem = comp['KVLoad_ms'] + comp['SparseWeightLoad_ms']
            sparse_comp = comp['SparseCompute_ms']
            row['SparseBound'] = 'MEM' if sparse_mem >= sparse_comp else 'COMPUTE'
            row['SparseMemPct'] = sparse_mem / comp['SparseOverlap_ms'] * 100 if comp['SparseOverlap_ms'] > 0 else 0
            row['SparseComputePct'] = sparse_comp / comp['SparseOverlap_ms'] * 100 if comp['SparseOverlap_ms'] > 0 else 0
            row['KernelStaticPct'] = comp['KernelStatic_ms'] / comp['SparseOverlap_ms'] * 100 if comp['SparseOverlap_ms'] > 0 else 0
            breakdown_rows.append(row)

breakdown_df = pd.DataFrame(breakdown_rows)
print(f"Generated {len(breakdown_df)} breakdown rows across {len(args_dict)} models")

# ── 4. Plot: per-layer latency waterfall (BS=1, TP=1) ───────
print("\nPlotting per-layer latency breakdown ...")

plot_df = breakdown_df[(breakdown_df['BS'] == 1) & (breakdown_df['TP'] == 1)].copy()
plot_df = plot_df.sort_values('SparseOverlap_ms', ascending=True).reset_index(drop=True)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# (a) Stacked bar: per-layer sparse attention components
ax = axes[0]
models = plot_df['Model'].tolist()
x = np.arange(len(models))
bar_w = 0.6

# Components that go into max(mem, compute) + kernel_static
kv_load = plot_df['KVLoad_ms'].values
weight_load = plot_df['SparseWeightLoad_ms'].values
compute_qk = plot_df['ComputeQK_ms'].values
compute_lowp = plot_df['ComputeLowp_ms'].values
kernel_static = plot_df['KernelStatic_ms'].values

# Show memory and compute side-by-side within each bar
# Memory side
ax.bar(x - 0.15, kv_load, bar_w * 0.4, label='KV Cache Load', color='#4E79A7')
ax.bar(x - 0.15, weight_load, bar_w * 0.4, bottom=kv_load, label='Weight Load', color='#A0CBE8')
# Compute side
ax.bar(x + 0.15, compute_qk, bar_w * 0.4, label='Compute QK+SV (highp)', color='#F28E2B')
ax.bar(x + 0.15, compute_lowp, bar_w * 0.4, bottom=compute_qk, label='Compute Proj (lowp)', color='#FFBE7D')
ax.bar(x + 0.15, kernel_static, bar_w * 0.4, bottom=compute_qk + compute_lowp,
       label='Kernel Static', color='#E15759')

# Overlap line
overlap = plot_df['SparseOverlap_ms'].values
ax.plot(x, overlap, 'ko-', ms=6, lw=2, label='Per-layer Total (overlap)')

ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Time (ms)')
ax.set_title(f'Sparse Attention Per-Layer Breakdown (BS=1, TP=1, seq={SEQ_LEN//1000}K)')
ax.legend(fontsize=8, ncol=2, loc='upper left')
ax.grid(axis='y', linestyle='--', alpha=0.3)

# (b) Total SparseAttn time stacked by model
ax2 = axes[1]
dense_total = plot_df['DenseAttnTotal_ms'].values
sparse_total = plot_df['SparseAttnTotal_ms'].values

ax2.barh(x, dense_total, 0.6, label='Dense Attn Total', color='#4E79A7')
ax2.barh(x, sparse_total, 0.6, left=dense_total, label='Sparse Attn Total', color='#59A14F')

for i in range(len(models)):
    total = dense_total[i] + sparse_total[i]
    ax2.text(total + 0.05, x[i], f'{total:.2f} ms', va='center', fontsize=9)

ax2.set_yticks(x)
ax2.set_yticklabels(models, fontsize=9)
ax2.set_xlabel('Total Time (ms)')
ax2.set_title(f'Attention Machine Total Time (BS=1, TP=1, seq={SEQ_LEN//1000}K)')
ax2.legend(fontsize=9)
ax2.grid(axis='x', linestyle='--', alpha=0.3)

plt.tight_layout()
latency_chart = os.path.join(OUTPUT_DIR, f'{PREFIX}_latency_breakdown.png')
plt.savefig(latency_chart, bbox_inches='tight', dpi=150)
plt.close(fig)
print(f"  Saved: {latency_chart}")

# ── 5. Plot: BS sweep effect on sparse attention ─────────────
fig2, ax3 = plt.subplots(figsize=(14, 7))
tp1_df = breakdown_df[breakdown_df['TP'] == 1].copy()
for model_name in args_dict:
    mdf = tp1_df[tp1_df['Model'] == model_name]
    ax3.plot(mdf['BS'], mdf['SparseAttnTotal_ms'], 'o-', label=model_name, lw=2)

ax3.set_xlabel('Batch Size')
ax3.set_ylabel('Sparse Attn Total (ms)')
ax3.set_title(f'Sparse Attention Total vs Batch Size (TP=1, seq={SEQ_LEN//1000}K)')
ax3.legend(fontsize=9)
ax3.grid(True, linestyle='--', alpha=0.3)
bs_chart = os.path.join(OUTPUT_DIR, f'{PREFIX}_sparse_attn_vs_bs.png')
plt.tight_layout()
plt.savefig(bs_chart, bbox_inches='tight', dpi=150)
plt.close(fig2)
print(f"  Saved: {bs_chart}")

# ── 6. Export xlsx with multiple sheets ──────────────────────
print("\n" + "=" * 60)
print("Exporting xlsx ...")
print("=" * 60)

xlsx_path = os.path.join(OUTPUT_DIR, f'{PREFIX}_analysis.xlsx')
with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
    # Sheet 1: Simulation parameters
    param_data = {
        'Parameter': [
            'ATTN_KERNEL_DISCOUNT', 'ATTN_KERNEL_STATIC_MS', 'ALLREDUCE_LATENCY_MS',
            'RING_HOP_LATENCY_MS', 'MOE_A2A_LATENCY_MS', 'MEM_UTIL_RATE',
            'WEIGHT_UTIL_RATE', 'CX9_BW_GBS', 'CX9_LATENCY_MS',
            'seq_len', 'decode_len',
        ],
        'Value': [
            sb.ATTN_KERNEL_DISCOUNT, sb.ATTN_KERNEL_STATIC_MS, sb.ALLREDUCE_LATENCY_MS,
            sb.RING_HOP_LATENCY_MS, sb.MOE_A2A_LATENCY_MS, sb.MEM_UTIL_RATE,
            sb.WEIGHT_UTIL_RATE, sb.CX9_BW_GBS, sb.CX9_LATENCY_MS,
            SEQ_LEN, 1,
        ],
    }
    pd.DataFrame(param_data).to_excel(writer, sheet_name='Parameters', index=False)

    # Sheet 2: Per-layer latency breakdown
    breakdown_df.to_excel(writer, sheet_name='PerLayerBreakdown', index=False)

    # Sheet 3: Best rows (main result)
    if not best_rows_df.empty:
        best_rows_df.to_excel(writer, sheet_name='BestRows', index=False)

    # Sheet 4: Candidates
    if not candidates_df.empty:
        candidates_df.to_excel(writer, sheet_name='Candidates', index=False)

    # Sheet 5: Feasible summary
    if not feasible_summary_df.empty:
        feasible_summary_df.to_excel(writer, sheet_name='FeasibleSummary', index=False)

    # Sheet 6: Summary table (compact view)
    if not best_rows_df.empty:
        summary_cols = ['Model', 'Config', 'TP', 'DP', 'BatchSize', 'DeviceNum',
                        'DenseAttnTotal_ms', 'SparseAttnTotal_ms',
                        'DenseFFNTotal_ms', 'SharedExpertTotal_ms', 'RoutedExpertTotal_ms',
                        'InterStageTotal_ms', 'TPOT_Disagg_ms',
                        'TPS_USER_Disagg', 'TPS_GPU_Disagg', 'TPS_Cluster_GPU_Disagg']
        avail = [c for c in summary_cols if c in best_rows_df.columns]
        summary = best_rows_df[avail].drop_duplicates(
            subset=['Model', 'Config'], keep='first'
        ).reset_index(drop=True)
        summary.to_excel(writer, sheet_name='Summary', index=False)

print(f"  Saved: {xlsx_path}")
print("\nDone.")
