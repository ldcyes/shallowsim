# Shallow Sim

ShallowSim is a simulator used for analyzing the inference performance of the DeepSeek-V3/R1 model.

This project primarily focuses on the optimal design of model architectures for different GPU architectures, as well as the analysis of interconnect bus requirements for Scale-Up and Scale-Out scenarios.

## New Features

Recent updates added the following capabilities:

1. Separate hardware configuration for prefill and decode stages.
2. CSV-driven model configuration with support for both MLA and GQA attention.
3. Disaggregated decode profiling for mixed deployments such as `GROQ + Rubin-NVL72`.
4. Precision is now configurable instead of hard-coded.
    - `NVFP4`
    - `MXFP8`
    - `BF16`
5. KV cache storage, parameter storage, and non-QK GEMM precision can be configured independently.
6. GROQ decode expert path has a dedicated discount table based on the `m=4` ISA alignment assumption.
7. Result plotting now includes a stacked stage-time chart for:
    - `Atten`
    - `Combine`
    - `Shared`
    - `Dispatch`
    - `Route`
8. `TargetTPS` in disaggregated result tables is evaluated against `TPS_Disagg`.
    - This is per-user TPS.
    - It is not compared against aggregate throughput `Total_Disagg`.

```python
import shallowsim as sb 

args = sb.ModelArgs()
c = sb.Config()

# Option 1: keep same hardware list for decode (backward compatible)
gpu_list = sb.get_gpu_info('./device/gpu_info.csv', 
                            decoding_mode=True, print_console=True)

dfs = sb.decode_time_with_ep_list(args,gpu_list,c,print_console=False,fp8_combine=True)
```

## Precision Configuration

You can switch precision globally for GEMM, parameter storage, and KV cache storage.

```python
import shallowsim as sb

args = sb.ModelArgs()

# One-line switch for GEMM + parameter + KV precision
sb.set_precision_mode(args, 'NVFP4')
# sb.set_precision_mode(args, 'MXFP8')
# sb.set_precision_mode(args, 'BF16')

# You can also control them independently
args.gemm_dtype = 'MXFP8'
args.param_dtype = 'BF16'
args.kv_cache_dtype = 'NVFP4'
```

## Use Different Hardware For Prefill And Decode

```python
import shallowsim as sb

args = sb.ModelArgs()
c = sb.Config()

# Configure hardware independently by stage
c.prefill_gpu_info_file = './device/gpu_info.csv'
c.decode_gpu_info_file = './device/gpu_info.csv'
c.prefill_device_list = ['GB200-NVL72']
c.decode_device_list = ['GB300-NVL72', 'B200-SXM']

gpu_prefill, gpu_decode = sb.get_stage_gpu_info(c, print_console=False)

# Prefill uses prefill hardware set
prefill_detail, prefill_summary = sb.prefill_time(
    args, gpu_prefill, c.seq_len, c.kv_cache_rate, tp=4, dp=8)

# Decode uses decode hardware set
decode_df = sb.decode_time_with_ep_list(
    args, gpu_decode, c, print_console=False, fp8_combine=True)
```

## Disaggregated Decode Stage Plot

Use the best-row result table to generate one stacked bar chart that shows stage breakdown in the same figure.

```python
import pandas as pd
import shallowsim as sb

df = pd.read_csv('results/groq_rubin72_400k_eval_nvfp4_best_rows.csv')
sb.plot_disagg_stage_stacked_time(
    df,
    'results/groq_rubin72_400k_stage_stacked_time_nvfp4.png',
    rotation=45,
)
```

Generated figure:

![Disaggregated stage stacked time](results/groq_rubin72_400k_stage_stacked_time_nvfp4.png)
## Summary Report 

```python
dfs_o = dfs.groupby(['GPU','BatchSize'],as_index=False).apply(lambda t: t[t.Total==t.Total.max()]).sort_values(['Total'],ascending=False).reset_index(drop=True)
dfs_o.style.bar(subset=['TPS','Total'],color='#6495ED').applymap(sb.color_positive_red, subset=['Delta']).background_gradient(subset=['Comm_Impact'],cmap=sb.cm).format(precision=3) 
```
![Performance result](figures/performance.png)

## Sort by GPU

```python
gpu = 'GB300-NVL72'
tps_limit = 20

tdf = sb.df_filter(dfs,gpu,tps_limit=tps_limit)
sb.df_sort(tdf,value='Total',ascending=False).style.bar(subset=['TPS','Total'],color='#6495ED').applymap(sb.color_positive_red, subset=['Delta']).background_gradient(subset=['Comm_Impact'],cmap=sb.cm).format(precision=3) 
```
![GB300 NVL72](figures/gb300_nvl72.png)

## Sequence Length analysis

1. Generate data

```python
dfs = []
for seq_len in trange(1024,16384,32):
    c.seq_len = seq_len
    df = sb.decode_time_with_ep_list(args,gpu_all_decode,c,fp8_combine=True)
    df['index_value'] = seq_len
    df_o = df.groupby(['GPU','BatchSize','EP'],as_index=False).apply(lambda t: t[t.Total==t.Total.max()]).sort_values(['Total'],ascending=False).reset_index(drop=True)
    df_o.drop_duplicates(subset=['GPU','BatchSize','EP'], keep='first', inplace=True)
    dfs.append(df_o)
df = pd.concat(dfs)    
df.reset_index(inplace=True,drop=True)
df.to_csv('perf_vs_seq_len.csv')
```

2. Load data and plot
```python
df = pd.read_csv('perf_vs_seq_len.csv')

# filert on EP
df1 = df[df['BatchSize'] == 128].reset_index(drop=True)

# plot
sb.draw(df1, gpu_all_decode, 
        comp_name='EP',comp_val_list=[36,72,144,320],
        val_list=['Total','TPS'],val_unit_name='Token per second',
        title='seq_len under EP strategies',savefig=True,filename='seq_len.png')
```
![Thoughput vs Seq_len](figures/seq.png)

## Smoke Tests

The following smoke tests are useful after changing model precision, KV assumptions, or stage plotting logic.

1. Precision mode switch

```python
import shallowsim as sb

args = sb.ModelArgs()
for precision in ['NVFP4', 'MXFP8', 'BF16']:
    sb.set_precision_mode(args, precision)
    print(args.param_dtype, args.kv_cache_dtype, args.gemm_dtype)
```

Expected behavior:

1. All three fields should switch together.
2. Output should be:
   - `NVFP4 NVFP4 NVFP4`
   - `MXFP8 MXFP8 MXFP8`
   - `BF16 BF16 BF16`

2. Precision-dependent memory and latency check

```python
import shallowsim as sb

args = sb.ModelArgs()
gpu = next(iter(sb.get_gpu_info('./device/gpu_info.csv', decoding_mode=True).values()))

for precision in ['NVFP4', 'MXFP8', 'BF16']:
    sb.set_precision_mode(args, precision)
    total, _ = sb.mla_elapse_time(
        args,
        gpu,
        seq_len=4096,
        kv_cache_rate=1,
        tp=[1],
        decoding_mode=True,
        batchsize=4,
        enable_gemm_fp4=True,
    )
    kv_gb = sb.decode_kv_cache_bytes(args, 4096, 1024, 1) / 1024 / 1024 / 1024
    print(precision, round(sb.decode_other_parameter_gb(args), 4), round(kv_gb, 4), round(total, 4))
```

Expected behavior:

1. `NVFP4` should have the smallest parameter/KV footprint.
2. `MXFP8` should be larger than `NVFP4`.
3. `BF16` should be the largest.

3. Stage stacked plot generation

```python
import pandas as pd
import shallowsim as sb

df = pd.read_csv('results/groq_rubin72_400k_eval_nvfp4_best_rows.csv')
sb.plot_disagg_stage_stacked_time(
    df,
    'results/groq_rubin72_400k_stage_stacked_time_nvfp4.png',
    rotation=45,
)
```

Expected behavior:

1. A new PNG is generated at `results/groq_rubin72_400k_stage_stacked_time_nvfp4.png`.
2. Each bar is stacked with `Atten`, `Combine`, `Shared`, `Dispatch`, `Route`.
3. X-axis labels are rotated for readability.
