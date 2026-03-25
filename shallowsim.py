import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import numpy as np
import copy as copy
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, Literal

cm = sns.light_palette("red", as_cmap=True)
NVL_GPU_LIST = [72, 144, 576]

class ModelArgs:
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    vocab_size: int = 129280
    dim: int = 7168
    inter_dim: int = 18432
    moe_inter_dim: int = 2048
    n_layers: int = 61
    n_dense_layers: int = 3
    n_heads: int = 128
    n_kv_heads: int = 1
    bpe: int = 1
    # moe
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 8
    n_expert_groups: int = 8
    n_limited_groups: int = 4
    route_scale: float = 2.5
    # mla
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.
    attn_type: str = 'MLA'


MODEL_ARG_FIELDS = [
    'max_batch_size', 'max_seq_len', 'vocab_size', 'dim', 'inter_dim',
    'moe_inter_dim', 'n_layers', 'n_dense_layers', 'n_heads', 'n_kv_heads', 'bpe',
    'n_routed_experts', 'n_shared_experts', 'n_activated_experts',
    'n_expert_groups', 'n_limited_groups', 'route_scale', 'q_lora_rank',
    'kv_lora_rank', 'qk_nope_head_dim', 'qk_rope_head_dim', 'v_head_dim',
    'original_seq_len', 'rope_theta', 'rope_factor', 'beta_fast',
    'beta_slow', 'mscale', 'attn_type'
]


MODEL_FIELD_ALIASES = {
    'model_name': ['model_name', 'model'],
    'n_layers': ['n_layers', 'layer'],
    'n_heads': ['n_heads', 'head'],
    'n_kv_heads': ['n_kv_heads', 'kv_head', 'kv head'],
    'dim': ['dim', 'hidden'],
    'q_lora_rank': ['q_lora_rank', 'q_lora'],
    'kv_lora_rank': ['kv_lora_rank', 'kv_lora'],
    'bpe': ['bpe'],
    'moe_inter_dim': ['moe_inter_dim', 'moe_intermediate', 'moe intermediate'],
    'n_activated_experts': ['n_activated_experts', 'top_k', 'top-k'],
    'n_routed_experts': ['n_routed_experts', 'expert'],
    'vocab_size': ['vocab_size', 'vocabulary', 'vacabulary'],
    'attn_type': ['attn_type', 'attention', 'attention_type'],
}


def _get_row_value(row, field):
    aliases = MODEL_FIELD_ALIASES.get(field, [field])
    for alias in aliases:
        if alias in row.index and not pd.isna(row[alias]):
            return row[alias]
    return None


def _build_model_args_from_row(row):
    args = ModelArgs()
    for field in MODEL_ARG_FIELDS:
        default_value = getattr(args, field)
        value = _get_row_value(row, field)
        if value is None:
            continue
        if isinstance(default_value, int):
            setattr(args, field, int(value))
        elif isinstance(default_value, float):
            setattr(args, field, float(value))
        else:
            setattr(args, field, value)
    return args


def get_model_args(filename='./model.csv',
                   model_name='DeepSeek-V3',
                   print_console=False):
    """Load one model config from CSV into ModelArgs."""
    df = pd.read_csv(filename)
    if print_console:
        name_col = 'model_name' if 'model_name' in df.columns else 'model'
        print(df.set_index(name_col).to_markdown())
    name_col = 'model_name' if 'model_name' in df.columns else 'model'
    model_df = df[df[name_col] == model_name]
    if model_df.empty:
        raise KeyError(f'Model {model_name} not found in {filename}')
    return _build_model_args_from_row(model_df.iloc[0])


def get_model_args_dict(filename='./model.csv',
                        model_list=None,
                        print_console=False):
    """Load multiple model configs from CSV."""
    if model_list is None:
        model_list = []
    df = pd.read_csv(filename)
    if print_console:
        name_col = 'model_name' if 'model_name' in df.columns else 'model'
        print(df.set_index(name_col).to_markdown())
    model_dict = {}
    name_col = 'model_name' if 'model_name' in df.columns else 'model'
    for _, row in df.iterrows():
        model_name = row[name_col]
        if model_list and model_name not in model_list:
            continue
        model_dict[model_name] = _build_model_args_from_row(row)
    return model_dict


class Config:
    seq_len = 4383
    decode_len = 1210
    kv_cache_rate = 0.563
    decode_len = 1210
    bs_list = [16, 32, 64, 128, 256, 512]
    eplist = [8, 16, 36, 72, 144, 320]
    # Hardware profiles can now be configured independently for prefill/decode.
    prefill_gpu_info_file = './device/gpu_info.csv'
    decode_gpu_info_file = './device/gpu_info.csv'
    prefill_device_list = []
    decode_device_list = []
    prefill_discount_rate = 0.85
    decode_discount_rate = 0.85


class GPU_perf:
    def __init__(self, gpu_type, sm, comm_sm, gpu_per_node,
                 fp16_flops, fp8_flops, fp4_flops,
                 mem, mem_bw, nvlink_bw, pcie_bw, discount_rate):
        self.gpu_type = gpu_type
        self.sm = sm
        self.gpu_per_node = gpu_per_node
        self.comm_sm = comm_sm
        self.fp16_flops = fp16_flops
        self.fp8_flops = fp8_flops
        self.fp4_flops = fp4_flops
        self.mem = mem
        self.mem_bw = mem_bw
        self.nvlink_bw = nvlink_bw
        self.pcie_bw = pcie_bw
        self.discount_rate = discount_rate

    def get_fp16_flops(self):
        return self.fp16_flops * self.discount_rate * (self.sm - self.comm_sm) / self.sm

    def get_fp8_flops(self):
        return self.fp8_flops * self.discount_rate * (self.sm - self.comm_sm) / self.sm

    def get_fp4_flops(self):
        return self.fp4_flops * self.discount_rate * (self.sm - self.comm_sm) / self.sm

    def get_mem_bw(self):
        return self.mem_bw * self.discount_rate

    def get_nvlink_bw(self):
        return self.nvlink_bw * self.discount_rate

    def get_pcie_bw(self):
        return self.pcie_bw * self.discount_rate


def get_gpu_info(filename='./device/gpuinfo.csv',
                 discount_rate=0.85,
                 device_list=[],
                 decoding_mode=False, print_console=False):
    """Get gpu info from csv file.

    Args:
        filename (str, optional): gpu performance datasheet filepath. Defaults to './device/gpuinfo.csv'.
        discount_rate (float, optional): Estimate performance discount from Peak FLOPS and peak BW. Defaults to 0.85.
        device_list (list, optional): select dedicated gpu. Defaults to [].
        decoding_mode (bool, optional): Enable decoding mode to set comm_sm=0. Defaults to False.
        print_console (bool, optional): print result. Defaults to False.

    Returns:
        dict{GPU_perf}: gpu performance dict.
    """
    gpu_dict = {}
    df = pd.read_csv(filename)
    if print_console:
        print(df.set_index('gpu_type').to_markdown())
    if decoding_mode:
        df['comm_sm'] = 0
    for _, c in df.iterrows():
        key = c['gpu_type']
        gpu = GPU_perf(
            gpu_type=c['gpu_type'],
            sm=c['sm'], comm_sm=c['comm_sm'],
            fp16_flops=c['fp16'],
            fp8_flops=c['fp8'],
            fp4_flops=c['fp4'],
            mem=c['mem'],
            mem_bw=c['mem_bw'],
            nvlink_bw=c['nvlink_bw'],
            pcie_bw=c['pcie_bw'],
            gpu_per_node=c['gpu_per_node'],
            discount_rate=discount_rate)
        if (len(device_list) == 0) | (key in device_list):
            gpu_dict[key] = gpu
    return gpu_dict


def get_stage_gpu_info(config: Config, print_console=False):
    """Load prefill/decode GPU profiles independently.

    This keeps backward compatibility with existing APIs while enabling
    stage-specific hardware selection from a single Config object.

    Returns:
        Tuple[dict{GPU_perf}, dict{GPU_perf}]: (prefill_gpu_dict, decode_gpu_dict)
    """
    prefill_gpu_dict = get_gpu_info(
        filename=getattr(config, 'prefill_gpu_info_file', './device/gpu_info.csv'),
        discount_rate=getattr(config, 'prefill_discount_rate', 0.85),
        device_list=getattr(config, 'prefill_device_list', []),
        decoding_mode=False,
        print_console=print_console)

    decode_gpu_dict = get_gpu_info(
        filename=getattr(config, 'decode_gpu_info_file', './device/gpu_info.csv'),
        discount_rate=getattr(config, 'decode_discount_rate', 0.85),
        device_list=getattr(config, 'decode_device_list', []),
        decoding_mode=True,
        print_console=print_console)

    return prefill_gpu_dict, decode_gpu_dict

def gpu_category_idx(gpu_dict):
    gpu_category={}
    i = 0
    for key in gpu_dict.keys():
        gpu_category[key]=i
        i +=1
    return gpu_category

# 非吸收的版本


def is_gqa_model(args: ModelArgs):
    return getattr(args, 'attn_type', 'MLA').upper() == 'GQA'


def gqa_head_dim(args: ModelArgs):
    return args.qk_nope_head_dim + args.qk_rope_head_dim


def gqa_kv_cache_dim(args: ModelArgs):
    return args.n_kv_heads * (gqa_head_dim(args) + args.v_head_dim)


def gqa_flops(q_len, kv_len, args: ModelArgs, kv_cache_rate):
    q_proj = q_len * args.dim * args.n_heads * gqa_head_dim(args)
    k_proj = kv_len * args.dim * args.n_kv_heads * gqa_head_dim(args)
    v_proj = kv_len * args.dim * args.n_kv_heads * args.v_head_dim
    o_proj = q_len * args.n_heads * args.v_head_dim * args.dim

    k_proj = k_proj * (1 - kv_cache_rate)
    v_proj = v_proj * (1 - kv_cache_rate)
    gemm_sum = q_proj + k_proj + v_proj + o_proj

    attn_sum = args.n_heads * (
        q_len * gqa_head_dim(args) * kv_len +
        q_len * kv_len * args.v_head_dim)

    gemm_sum = gemm_sum * 2 / 1e9
    attn_sum = attn_sum * 2 / 1e9
    return gemm_sum + attn_sum, gemm_sum, attn_sum


def gqa_mem(args: ModelArgs):
    q_proj = args.dim * args.n_heads * gqa_head_dim(args)
    k_proj = args.dim * args.n_kv_heads * gqa_head_dim(args)
    v_proj = args.dim * args.n_kv_heads * args.v_head_dim
    o_proj = args.n_heads * args.v_head_dim * args.dim
    return (q_proj + k_proj + v_proj + o_proj) / 1024 / 1024


def gqa_elapse_time(args: ModelArgs,
                    gpu: GPU_perf,
                    seq_len,
                    kv_cache_rate,
                    tp=[2, 4, 8, 16, 32],
                    decoding_mode=True,
                    batchsize=1,
                    enable_gemm_fp4=True,
                    min_ar_time=0.015,
                    gqa_discount=0.7,
                    gqa_kernel_static_time=0.05,
                    print_console=False):
    if decoding_mode:
        _, gemm_flops, attn_fp16_flops = gqa_flops(1, seq_len, args, 1)
        gemm_flops *= batchsize
        attn_fp16_flops *= batchsize
    else:
        _, gemm_flops, attn_fp16_flops = gqa_flops(seq_len, seq_len, args, kv_cache_rate)

    gemm_fp8_t = gemm_flops / gpu.get_fp8_flops() / gqa_discount
    attn_fp16_t = attn_fp16_flops / gpu.get_fp16_flops() / gqa_discount
    load_t = gqa_mem(args) / gpu.get_mem_bw()

    total = gemm_fp8_t + attn_fp16_t + load_t
    if enable_gemm_fp4 and gpu.get_fp4_flops() != 0:
        gemm_fp4_t = gemm_flops / gpu.get_fp4_flops()
        total = gemm_fp4_t + attn_fp16_t

    ar_len = batchsize if decoding_mode else seq_len
    all_reduce_comm_size = ar_len * args.dim * 2 / 1024 / 1024
    all_reduce_t = all_reduce_comm_size / gpu.get_nvlink_bw() + min_ar_time

    tp_time = {}
    for v in tp:
        if v == 1:
            tp_time[v] = total + gqa_kernel_static_time
        else:
            tp_time[v] = total / v + all_reduce_t + gqa_kernel_static_time

    if print_console:
        print("[%8s]GQA_GEMM_FP8 Elapsed time(ms): %.3f" %
              (gpu.gpu_type, gemm_fp8_t))
        print("[%8s]GQA_ATTN_FP16 Elapsed time(ms): %.3f" %
              (gpu.gpu_type, attn_fp16_t))
    return total, tp_time


def attn_mem(args: ModelArgs):
    if is_gqa_model(args):
        return gqa_mem(args)
    return mla_mem(args)


def attn_kv_cache_dim(args: ModelArgs):
    if is_gqa_model(args):
        return gqa_kv_cache_dim(args)
    return args.kv_lora_rank + args.qk_rope_head_dim


def decode_kv_cache_storage_multiplier(args: ModelArgs, tp):
    if is_gqa_model(args):
        return 1
    return tp


def decode_kv_cache_load_divisor(args: ModelArgs, tp):
    if is_gqa_model(args):
        return max(1, tp)
    return 1


def decode_kv_cache_bytes(args: ModelArgs, seq_len, decode_len, tp):
    token_num = seq_len + decode_len
    kv_dim = attn_kv_cache_dim(args)
    return token_num * kv_dim * args.n_layers * decode_kv_cache_storage_multiplier(args, tp)


def decode_kv_load_time(args: ModelArgs, gpu: GPU_perf, seq_len, batchsize, tp):
    kv_cache = seq_len * attn_kv_cache_dim(args) * batchsize
    kv_cache = kv_cache / decode_kv_cache_load_divisor(args, tp)
    return kv_cache / 1024 / 1024 / 1024 / gpu.get_mem_bw() * 1000


def attn_elapse_time(args: ModelArgs, *func_args, **func_kwargs):
    if is_gqa_model(args):
        return gqa_elapse_time(args, *func_args, **func_kwargs)
    return mla_elapse_time(args, *func_args, **func_kwargs)


def mla_flops(q_len, kv_len, args: ModelArgs, kv_cache_rate):
    # calculate MACs and estimate Flops approx. 2xMAC.
    q_down_proj = q_len * args.dim * args.q_lora_rank  # wq_a
    q_up_proj = q_len * args.q_lora_rank * args.n_heads * \
        (args.qk_nope_head_dim + args.qk_rope_head_dim)  # wq_b
    kv_down_proj = kv_len * args.dim * \
        (args.kv_lora_rank + args.qk_rope_head_dim)  # wkv_a
    k_up_proj = kv_len * args.kv_lora_rank * \
        args.n_heads * args.qk_nope_head_dim  # w_uk
    v_up_proj = kv_len * args.kv_lora_rank * args.n_heads * args.v_head_dim  # w_uv

    kv_down_proj = kv_down_proj * (1 - kv_cache_rate)
    gemm_sum = q_down_proj + q_up_proj + kv_down_proj + k_up_proj + v_up_proj

    # 把它看成一个标准的args.n_heads的MHA
    mha = args.n_heads * (q_len * args.qk_rope_head_dim * kv_len  # QK_score_rope
                          + q_len * args.qk_nope_head_dim * kv_len  # QK_score_nope
                          + q_len * kv_len * args.v_head_dim)  # ScoreV
    wo = q_len * args.n_heads * args.v_head_dim * args.dim  # wo
    attn_sum = mha + wo
    # return flops by 2* Sum(MACs)
    GEMM_FP8_FLOPS = gemm_sum * 2/1e9
    ATTN_FP16_FLOPS = attn_sum * 2/1e9

    return GEMM_FP8_FLOPS+ATTN_FP16_FLOPS, GEMM_FP8_FLOPS, ATTN_FP16_FLOPS

# 矩阵吸收的版本


def mla_matabsob_flops(q_len, kv_len, args: ModelArgs, kv_cache_rate=0):
    # calculate MACs and estimate Flops approx. 2xMAC.
    q_down_proj = q_len * args.dim * args.q_lora_rank  # wq_a
    q_rope_up_proj = q_len * args.q_lora_rank * \
        args.n_heads * args.qk_rope_head_dim  # wq_b_rope
    q_absorb = q_len * args.n_heads * (args.q_lora_rank * args.qk_nope_head_dim  # wq_b
                                       + args.qk_nope_head_dim * args.kv_lora_rank)  # w_uk

    kv_down_proj = kv_len * args.dim * \
        (args.kv_lora_rank + args.qk_rope_head_dim)  # wkv_a
    kv_down_proj = kv_down_proj * (1 - kv_cache_rate)  # KV-Cache命中率修正
    gemm_sum = q_down_proj + q_rope_up_proj + q_absorb + kv_down_proj

    # 把它看成一个标准的args.n_heads的MQA
    mqa = args.n_heads * (q_len * args.qk_rope_head_dim * kv_len  # Score_rope
                          + q_len * args.kv_lora_rank * kv_len  # Score_nope
                          + q_len * kv_len * args.kv_lora_rank)  # Score V

    attn_up_proj = q_len * args.n_heads * args.v_head_dim * args.kv_lora_rank
    o_proj = q_len * args.n_heads * args.v_head_dim * args.dim
    attn_sum = mqa + attn_up_proj + o_proj

    # return flops by 2* Sum(MACs)
    gemm_sum = gemm_sum * 2/1e9
    attn_sum = attn_sum * 2/1e9

    return gemm_sum + attn_sum, gemm_sum, attn_sum


def mla_mem(args: ModelArgs):
    q_down_proj = args.dim * args.q_lora_rank  # wq_a
    q_up_proj = args.q_lora_rank * args.n_heads * \
        (args.qk_nope_head_dim + args.qk_rope_head_dim)  # wq_b
    kv_down_proj = args.dim * \
        (args.kv_lora_rank + args.qk_rope_head_dim)  # wkv_a
    k_up_proj = args.kv_lora_rank * args.n_heads * args.qk_nope_head_dim  # w_uk
    v_up_proj = args.kv_lora_rank * args.n_heads * args.v_head_dim  # w_uv
    wo = args.n_heads * args.v_head_dim * args.dim  # wo
    return (q_down_proj + q_up_proj + k_up_proj + kv_down_proj + v_up_proj + wo)/1024/1024


def mla_elapse_time(args: ModelArgs,
                    gpu: GPU_perf,
                    seq_len,
                    kv_cache_rate,
                    tp=[2, 4, 8, 16, 32],
                    decoding_mode=True,
                    batchsize=1,
                    enable_gemm_fp4=True,
                    min_ar_time=0.015,  # Allreduce的静态延迟
                    mla_discount=0.7,  # based on FlashMLA result on H800
                    mla_kernel_static_time=0.05,
                    print_console=False):
    if decoding_mode:
        # Decoding时计算为qlen=1, kv_cache_rate = 1
        _, gemm_flops, attn_fp16_flops = mla_matabsob_flops(
            1, seq_len, args, 1)
        gemm_flops *= batchsize
        attn_fp16_flops *= batchsize
    else:
        # prefill阶段使用非吸收的版本
        _, gemm_flops, attn_fp16_flops = mla_flops(
            seq_len, seq_len, args, kv_cache_rate)
    gemm_fp8_t = gemm_flops / gpu.get_fp8_flops() / mla_discount
    attn_fp16_t = attn_fp16_flops / gpu.get_fp16_flops() / mla_discount

    # load weight
    load_t = mla_mem(args) / gpu.get_mem_bw()

    total = gemm_fp8_t + attn_fp16_t + load_t

    if enable_gemm_fp4:
        if gpu.get_fp4_flops() == 0:
            if print_console:
                print('[%8s]This GPU does not support FP4' % gpu.gpu_type)
        else:
            gemm_fp4_t = gemm_flops / gpu.get_fp4_flops()
            total = gemm_fp4_t + attn_fp16_t

    ar_len = batchsize if decoding_mode else seq_len
    all_reduce_comm_size = ar_len * args.dim * 2 / 1024/1024  # fp16 take 2Bytes
    all_reduce_t = all_reduce_comm_size / gpu.get_nvlink_bw() + min_ar_time

    tp_time = {}
    for v in tp:
        if v == 1:
            tp_time[v] = total + mla_kernel_static_time
        else:
            tp_time[v] = total / v + all_reduce_t + mla_kernel_static_time

    if print_console:
        if enable_gemm_fp4 & (gpu.get_fp4_flops() != 0):
            print("[%8s]GEMM_FP4 Elapsed time(ms): %.3f" %
                  (gpu.gpu_type, gemm_fp4_t))
        print("[%8s]GEMM_FP8 Elapsed time(ms): %.3f" %
              (gpu.gpu_type, gemm_fp8_t))
        print("[%8s]ATTN_FP16 Elapsed time(ms): %.3f" %
              (gpu.gpu_type, attn_fp16_t))
        print("[%8s]Total Elapsed time(ms):%.3f" % (gpu.gpu_type, total))
        print("[%8s]AR Elapsed time(ms):%.3f" % (gpu.gpu_type, all_reduce_t))
        for v in tp:
            print("[%8s]TP[%2d] Elapsed time(ms):%.3f" %
                  (gpu.gpu_type, v, tp_time[v]))

    return total, tp_time


def prefill_mla(args: ModelArgs, gpu_dict, seq_len, kv_cache_rate, print_console=False):
    df = pd.DataFrame(columns=['GPU', 'TP1', 'TP4', 'TP8'])
    for key in gpu_dict.keys():
        tp1, tp_list = mla_elapse_time(args, gpu_dict[key],
                                       seq_len, kv_cache_rate,
                                       tp=[4, 8],
                                       decoding_mode=False,
                                       enable_gemm_fp4=True,
                                       print_console=print_console)
        df.loc[len(df)] = [gpu_dict[key].gpu_type, tp1] + \
            list(tp_list.values())
    if print_console:
        print(df.set_index('GPU').to_markdown(floatfmt=".3f"))
    return df


def prefill_gqa(args: ModelArgs, gpu_dict, seq_len, kv_cache_rate, print_console=False):
    df = pd.DataFrame(columns=['GPU', 'TP1', 'TP4', 'TP8'])
    for key in gpu_dict.keys():
        tp1, tp_list = gqa_elapse_time(args, gpu_dict[key],
                                       seq_len, kv_cache_rate,
                                       tp=[4, 8],
                                       decoding_mode=False,
                                       enable_gemm_fp4=True,
                                       print_console=print_console)
        df.loc[len(df)] = [gpu_dict[key].gpu_type, tp1] + list(tp_list.values())
    if print_console:
        print(df.set_index('GPU').to_markdown(floatfmt='.3f'))
    return df


def densmlp_flops(args: ModelArgs, seq_len):
    return 3 * seq_len * args.dim * args.inter_dim * 2/1e9


def densmlp_mem(args: ModelArgs):
    return 3 * args.dim * args.inter_dim / 1024/1024


def _prefill_dense_mlp(args: ModelArgs, gpu: GPU_perf, seq_len, print_console=False):
    gemm_flops = densmlp_flops(args, seq_len)
    if gpu.get_fp4_flops() != 0:
        gemm_time = gemm_flops / gpu.get_fp4_flops()
    else:
        gemm_time = gemm_flops / gpu.get_fp8_flops()

    load_time = densmlp_mem(args) / gpu.get_mem_bw()
    gemm_time = gemm_time + load_time
    if print_console:
        print("[%8s]Elapsed time(ms): %.3f" % (gpu.gpu_type, gemm_time))
    return gemm_time


def prefill_dense_mlp(args: ModelArgs, gpu_dict, seq_len, print_console=False):
    df = pd.DataFrame(columns=['GPU', 'DenseMLP'])
    for key in gpu_dict.keys():
        t = _prefill_dense_mlp(args, gpu_dict[key], seq_len, print_console=print_console)
        df.loc[len(df)] = [gpu_dict[key].gpu_type, t]
    if print_console:
        print(df.set_index('GPU').to_markdown(floatfmt=".3f"))
    return df


def moe_expert_flops(args: ModelArgs, seq_len):
    return 3 * seq_len * args.dim * args.moe_inter_dim * 2/1e9


def moe_expert_mem(args: ModelArgs):
    return 3 * args.dim * args.moe_inter_dim / 1024 / 1024


def _prefill_moe(args: ModelArgs, gpu: GPU_perf, seq_len, tp, dp):
    load_time = moe_expert_mem(args) / gpu.get_mem_bw()
    gemm_flops = gpu.get_fp4_flops() if gpu.get_fp4_flops() != 0 else gpu.get_fp8_flops()
    num_device = tp * dp
    num_shared_token = dp * seq_len / num_device
    shared_flops = moe_expert_flops(args, num_shared_token)
    shared_time = shared_flops / gemm_flops + load_time

    num_routed_token = seq_len * dp * args.n_activated_experts / num_device
    routed_flops = moe_expert_flops(args, num_routed_token)
    expert_num = math.ceil(args.n_routed_experts) / dp / tp
    routed_time = routed_flops / gemm_flops + load_time * expert_num

    return shared_time, routed_time


def prefill_moe(args: ModelArgs, gpu_dict, seq_len,
                tp_list=[4, 8],
                dp_list=[4, 8, 9],
                print_console=False):
    df = pd.DataFrame(columns=['GPU', 'TP', 'DP',
                      'Shared Expert', 'Routed Expert'])
    for key in gpu_dict.keys():
        for tp in tp_list:
            for dp in dp_list:
                s, r = _prefill_moe(args, gpu_dict[key], seq_len, tp, dp)
                df.loc[len(df)] = [gpu_dict[key].gpu_type, tp, dp, s, r]
    if print_console:
        df['TP'] = df['TP'].astype(int).astype(str)
        df['DP'] = df['DP'].astype(int).astype(str)
        print(df.set_index('GPU').to_markdown(floatfmt=".3f"))
    return df


def _prefill_alltoall(args: ModelArgs, gpu, seq_len, tp, static_latency=0.05):
    if gpu.gpu_per_node == 8:
        dp = gpu.gpu_per_node/tp
        dispatch_node = 4
        dispatch_size = (dispatch_node - 1) * dp * seq_len * \
            args.n_activated_experts / gpu.gpu_per_node * args.dim / 1024/1024
        comm_bw = gpu.get_pcie_bw() * gpu.gpu_per_node
    else:
        # NVL72
        expert_num = math.ceil(args.n_routed_experts / gpu.gpu_per_node)
        dispatch_prob = (args.n_routed_experts - expert_num) / \
            args.n_routed_experts
        dispatch_size = dispatch_prob * args.n_activated_experts * \
            seq_len/tp * args.dim / 1024/1024
        comm_bw = gpu.get_nvlink_bw()

    combine_size = 2 * dispatch_size  # fp16
    if gpu.get_fp4_flops != 0:
        dispatch_size = dispatch_size / 2
    dispatch_time = dispatch_size / comm_bw + static_latency
    combine_time = combine_size / comm_bw + static_latency
    return dispatch_time, combine_time


def prefill_alltoall(args: ModelArgs, gpu_dict, seq_len, print_console=False):
    df = pd.DataFrame(columns=['GPU', 'TP', 'Dispatch', 'Combine'])
    for tp in [4, 8]:
        for key in gpu_dict.keys():
            dispatch_time, combine_time = _prefill_alltoall(
                args, gpu_dict[key], seq_len, tp)
            df.loc[len(df)] = [key, tp, dispatch_time, combine_time]
    if print_console:
        df['TP'] = df['TP'].astype(int).astype(str)
        print(df.set_index('GPU').to_markdown(floatfmt=".3f"))
    return df


def _prefill_time(args: ModelArgs, gpu, seq_len, kv_cache_rate, tp, dp):
    dense_mla, tp_mla = attn_elapse_time(args, gpu,
                                         seq_len, kv_cache_rate,
                                         tp=[tp],
                                         decoding_mode=False,
                                         enable_gemm_fp4=True)
    dense_mlp = _prefill_dense_mlp(args, gpu, seq_len)
    shared, routed = _prefill_moe(args, gpu, seq_len, tp, dp)
    dispatch, combine = _prefill_alltoall(args, gpu, seq_len, tp)
    return dense_mla, dense_mlp, tp_mla[tp], shared, combine, routed, dispatch


def prefill_time(args: ModelArgs, gpu_dict, seq_len, kv_cache_rate, tp, dp, print_console=False):
    df = pd.DataFrame(columns=['GPU', 'MLA', 'DenseMLP', 'TP_MLA', 'Shared Expert',
                      'Combine', 'Overlap1', 'Routed Expert', 'Dispatch', 'Overlap2'])
    df2 = pd.DataFrame(columns=['GPU', 'Compute', 'Comm', 'Sum'])
    n_sparse_layers = args.n_layers - args.n_dense_layers
    df.loc[len(df)] = ['Layers', args.n_dense_layers, args.n_dense_layers,  # MLA+ DenseMLP
                       n_sparse_layers, n_sparse_layers, n_sparse_layers, n_sparse_layers,
                       n_sparse_layers, n_sparse_layers, n_sparse_layers]
    for key in gpu_dict.keys():
        dense_mla, dense_mlp, tp_mla, shared, combine, routed, dispatch = _prefill_time(
            args, gpu_dict[key], seq_len, kv_cache_rate, tp, dp)
        overlap1 = combine - (tp_mla + shared)
        overlap2 = dispatch - routed
        df.loc[len(df)] = [key, dense_mla, dense_mlp, tp_mla, shared,
                           combine, overlap1, routed, dispatch, overlap2]
        comp_time = args.n_dense_layers * \
            (dense_mla + dense_mlp) + n_sparse_layers * (tp_mla + shared + routed)
        comm_time = n_sparse_layers * (combine + dispatch)
        sum_time = comp_time
        if overlap1 > 0:
            sum_time += overlap1 * n_sparse_layers
        if overlap2 > 0:
            sum_time += overlap2 * n_sparse_layers
        df2.loc[len(df2)] = [key, comp_time, comm_time, sum_time]
    df = df.set_index('GPU').T
    df2 = df2.set_index('GPU').T
    if print_console:
        df['Layers'] = df['Layers'].astype(int).astype(str)
        print(df.to_markdown(floatfmt=".3f"))
        print('-----------SUM-------------')
        print(df2.to_markdown(floatfmt=".3f"))
    return df, df2

# Decoding


def decode_other_parameter_gb(args: ModelArgs):
    """Estimate non-attention, non-expert resident parameters in GB.

    This includes:
    - token embedding and LM head
    - dense MLP weights in dense layers
    - router/gate weights in sparse layers
    - normalization weights
    """
    sparse_layer_num = args.n_layers - args.n_dense_layers
    embedding_and_head_gb = 2 * args.vocab_size * args.dim / 1024 / 1024 / 1024
    dense_mlp_gb = densmlp_mem(args) * args.n_dense_layers / 1024
    router_gb = sparse_layer_num * args.dim * args.n_routed_experts / 1024 / 1024 / 1024
    norm_gb = ((2 * args.n_layers) + 1) * args.dim / 1024 / 1024 / 1024
    return embedding_and_head_gb + dense_mlp_gb + router_gb + norm_gb


def _decoding_batchsize(args: ModelArgs, gpu: GPU_perf, seq_len, decode_len, tp, expert_num):
    mem_util_rate = 0.9  # torch/activation等其它开销的折扣
    attn_param = attn_mem(args)
    expert_mem = moe_expert_mem(args)  # expert参数，单位MB
    others_parameter = decode_other_parameter_gb(args)
    kv_cache = decode_kv_cache_bytes(args, seq_len, decode_len, tp)
    mem = gpu.mem * mem_util_rate - others_parameter - attn_param * args.n_layers / tp / 1024
    mem -= expert_mem * \
        (args.n_layers - args.n_dense_layers) * expert_num / 1024
    return mem * 1024 * 1024 * 1024 / kv_cache


def decode_ffn_weight_gb(args: ModelArgs):
    dense_mlp_gb = densmlp_mem(args) * args.n_dense_layers / 1024
    sparse_layer_num = args.n_layers - args.n_dense_layers
    moe_expert_gb = moe_expert_mem(args) * sparse_layer_num * \
        (args.n_shared_experts + args.n_routed_experts) / 1024
    return dense_mlp_gb + moe_expert_gb


def min_cabinet_count_for_decode_ffn(args: ModelArgs,
                                     gpu: GPU_perf,
                                     weight_util_rate=0.9):
    cabinet_mem_gb = gpu.mem * gpu.gpu_per_node * weight_util_rate
    return max(1, math.ceil(decode_ffn_weight_gb(args) / cabinet_mem_gb))


def inter_stage_link_time(bs,
                          args: ModelArgs,
                          tp,
                          link_bw=100,
                          link_latency=0.005,
                          max_link_rails=8):
    activation_mb = bs * args.dim * 2 / 1024 / 1024
    rail_num = max(1, min(tp, max_link_rails))
    effective_bw = link_bw * rail_num
    one_way = activation_mb / effective_bw
    return 2 * (one_way + link_latency)


def decode_batchsize(args: ModelArgs, gpu_dict, seq_len, decode_len, tp):
    df = pd.DataFrame(columns=['GPU', 'EP320', 'EP144', 'EP72', 'EP34'])
    for key in gpu_dict.keys():
        item = key
        value = [item]
        for exp_num in [2, 3, 5, 9]:
            bs = _decoding_batchsize(
                args, gpu_dict[key], seq_len, decode_len, tp, exp_num)
            value.append(bs)
        df.loc[len(df)] = value
    print(df.set_index('GPU').to_markdown(floatfmt=".0f"))
    return df


def decode_mla(args: ModelArgs, gpu_dict, bs_list, seq_len, decode_len,
               expert_num=2,
               tp_list=None,
               print_console=False):
    df = pd.DataFrame(columns=['GPU', 'BatchSize',
                      'TP', 'LoadKVPerDevice', 'DenseMLA', 'SparseMLA'])
    if tp_list is None:
        tp_list = [1, 4, 8]
    for key in gpu_dict.keys():
        for bs in bs_list:
            dense_mla, sparse_mla = mla_elapse_time(args, gpu_dict[key],
                                                    seq_len, kv_cache_rate=1,
                                                    tp=tp_list,
                                                    batchsize=bs,
                                                    decoding_mode=True,
                                                    enable_gemm_fp4=True)
            for tp_num in tp_list:
                max_bs = _decoding_batchsize(
                    args, gpu_dict[key], seq_len, decode_len, expert_num=expert_num, tp=tp_num)
                if bs > max_bs:
                    continue
                else:
                    load_kv_time = decode_kv_load_time(
                        args, gpu_dict[key], seq_len, bs, tp_num)
                    df.loc[len(df)] = [gpu_dict[key].gpu_type, bs, tp_num,
                                       load_kv_time, dense_mla, sparse_mla[tp_num]]
    if print_console:
        df['BatchSize'] = df['BatchSize'].astype(int).astype(str)
        print(df.set_index('GPU').to_markdown(floatfmt=".3f"))
    return df


def decode_gqa(args: ModelArgs, gpu_dict, bs_list, seq_len, decode_len,
               expert_num=2,
               tp_list=None,
               print_console=False):
    df = pd.DataFrame(columns=['GPU', 'BatchSize',
                      'TP', 'LoadKVPerDevice', 'DenseMLA', 'SparseMLA'])
    if tp_list is None:
        tp_list = [1, 4, 8]
    for key in gpu_dict.keys():
        for bs in bs_list:
            dense_attn, sparse_attn = gqa_elapse_time(args, gpu_dict[key],
                                                      seq_len, kv_cache_rate=1,
                                                      tp=tp_list,
                                                      batchsize=bs,
                                                      decoding_mode=True,
                                                      enable_gemm_fp4=True)
            for tp_num in tp_list:
                max_bs = _decoding_batchsize(
                    args, gpu_dict[key], seq_len, decode_len, expert_num=expert_num, tp=tp_num)
                if bs > max_bs:
                    continue
                load_kv_time = decode_kv_load_time(
                    args, gpu_dict[key], seq_len, bs, tp_num)
                df.loc[len(df)] = [gpu_dict[key].gpu_type, bs, tp_num,
                                   load_kv_time, dense_attn, sparse_attn[tp_num]]
    if print_console:
        df['BatchSize'] = df['BatchSize'].astype(int).astype(str)
        print(df.set_index('GPU').to_markdown(floatfmt='.3f'))
    return df


def decode_attention(args: ModelArgs, *func_args, **func_kwargs):
    if is_gqa_model(args):
        return decode_gqa(args, *func_args, **func_kwargs)
    return decode_mla(args, *func_args, **func_kwargs)


def decode_dense_mlp(args: ModelArgs, gpu_dict, bs_list, seq_len, decode_len,
                     expert_num=2,
                     tp_list=None,
                     print_console=False):
    if tp_list is None:
        tp_list = [1, 4, 8]  # only used for calc max batchsize
    df = pd.DataFrame(columns=['GPU', 'BatchSize', 'TP', 'DenseMLP'])
    for key in gpu_dict.keys():
        for bs in bs_list:
            t = _prefill_dense_mlp(args, gpu_dict[key], bs)
            for tp_num in tp_list:
                max_bs = _decoding_batchsize(
                    args, gpu_dict[key], seq_len, decode_len, expert_num=expert_num, tp=tp_num)
                if bs > max_bs:
                    continue
                else:
                    df.loc[len(df)] = [gpu_dict[key].gpu_type, bs, tp_num, t]
    if print_console:
        df['BatchSize'] = df['BatchSize'].astype(int).astype(str)
        print(df[df['TP'] == 1][['GPU', 'BatchSize', 'DenseMLP']
                                ].set_index('GPU').to_markdown(floatfmt=".3f"))
    return df


def n_pow2_range(n:int):
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n = n+1
    return n

def _decode_moe_expert(args: ModelArgs, gpu: GPU_perf, bs, 
                       gemm_group_per_device, device_num):
    load_time = moe_expert_mem(args) / gpu.get_mem_bw()
    if gpu.get_fp4_flops() != 0:
        load_time = load_time /2
    gpu_flops = gpu.get_fp4_flops() if gpu.get_fp4_flops() != 0 else gpu.get_fp8_flops()
    
    total_expert = gemm_group_per_device * device_num
    m_per_group = bs * args.n_activated_experts * device_num / total_expert
   
    '''
    # TODO: 
    # 基于 group_gemm num 和 m_per_group 调整折扣因子
    # 可以基于Profile实测结果查表, 并将数据放在GPU_Perf结构题中
    # 此处简化以m_per_group估计如下
    '''

    #data from hs's profiling result
    flops_discounts = {
        1: 0.05,
        2: 0.05,
        4: 0.05,
        8: 0.05,
        16: 0.08,
        32: 0.1,
        64: 0.2,
        128: 0.35,
        256: 0.4,
        512: 0.6,
        1024: 0.7,
        2048: 0.7,
        4096: 0.7,
        8192: 0.7,
        16384: 0.7,
        32768: 0.7,
        65536: 0.7
    }

    # H20 exception based on hs's result
    if gpu.gpu_type.find('H20')!= -1 :
        flops_discounts = {
        1: 0.06,
        2: 0.06,
        4: 0.06,
        8: 0.12,
        16: 0.25,
        32: 0.45,
        64: 0.8,
        128: 0.9,
        256: 1.0,
        512: 1.0,
        1024: 1.0,
        2048: 1.0,
        4096: 1.0,
        8192: 1.0,
        16384: 1.0,
        32768: 1.0,
        65536: 1.0
    }

    gpu_flops = gpu_flops * flops_discounts[n_pow2_range(int(m_per_group))]
    
    shared_flops = moe_expert_flops(args, bs)
    shared_time = shared_flops / gpu_flops + load_time

    num_routed_token = bs * args.n_activated_experts
    routed_flops = moe_expert_flops(args, num_routed_token)
    routed_time = routed_flops / gpu_flops + load_time * gemm_group_per_device
    return shared_time, routed_time


def decode_moe_expert(args: ModelArgs, gpu_dict, 
                      bs_list, seq_len, decode_len, 
                      gemm_group_per_device,
                      device_num,
                      mbs=2, 
                      tp_list=None,
                      print_console=False):
    if tp_list is None:
        tp_list = [1, 4, 8]  # only used for calc max batchsize
    df = pd.DataFrame(columns=['GPU', 'BatchSize',
                      'TP', 'SharedExpert', 'RoutedExpert'])
    for gpu_key in gpu_dict.keys():
        for bs in bs_list:
            s, r = _decode_moe_expert(
                args, gpu_dict[gpu_key], bs/mbs, 
                gemm_group_per_device=gemm_group_per_device, 
                device_num=device_num)
            s *= mbs
            r *= mbs
            for tp_num in tp_list:
                max_bs = _decoding_batchsize(
                    args, gpu_dict[gpu_key], seq_len, decode_len, 
                    expert_num= gemm_group_per_device+1, tp=tp_num)
                if bs > max_bs:
                    continue
                else:
                    df.loc[len(df)] = [gpu_dict[gpu_key].gpu_type,
                                       str(bs), tp_num, s, r]
    if print_console:
        df['BatchSize'] = df['BatchSize'].astype(int).astype(str)
        print(df[df['TP'] == 1][['GPU', 'BatchSize', 'SharedExpert',
              'RoutedExpert']].set_index('GPU').to_markdown(floatfmt=".3f"))
    return df


def _moe_a2a(args: ModelArgs, gpu: GPU_perf, bs, expert_num, device_num, fp8_combine=False, static_latency=0.005, mbs=2):
    dispatch_size = bs * args.dim * args.n_activated_experts / 1024/1024
    if fp8_combine & (gpu.get_fp4_flops() != 0):  # 支持FP4GPU才能开启FP8 Combine
        combine_size = dispatch_size
    else:
        combine_size = dispatch_size * 2  # FP16
    if gpu.gpu_per_node == 8:
        comm_bw = gpu.get_pcie_bw()
        # single host deployment
        if args.n_routed_experts / (expert_num - 1) == gpu.gpu_per_node:
            comm_bw = gpu.get_nvlink_bw()
    #NVL72 /144 / 576
    elif (gpu.gpu_per_node in NVL_GPU_LIST) & (device_num >  gpu.gpu_per_node):
            comm_bw = gpu.get_pcie_bw()
    else:
        comm_bw = gpu.get_nvlink_bw()

    dispatch_t = dispatch_size / comm_bw + static_latency * mbs
    combine_t = combine_size / comm_bw + static_latency * mbs
    return dispatch_t, combine_t


def decode_a2a(args: ModelArgs, gpu_dict,
               bs_list, seq_len, decode_len,
               expert_num, device_num,
               mbs=2,
               tp_list=None,
               print_console=False, fp8_combine=False):
    if tp_list is None:
        tp_list = [1, 4, 8]  # only used for calc max batchsize
    df = pd.DataFrame(columns=['GPU', 'BatchSize',
                      'TP', 'Dispatch', 'Combine'])
    for key in gpu_dict.keys():
        for bs in bs_list:
            dispatch_time, combine_time = _moe_a2a(
                args, gpu_dict[key], bs, 
                expert_num=expert_num, device_num=device_num, 
                mbs=mbs, fp8_combine=fp8_combine)
            for tp_num in tp_list:
                max_bs = _decoding_batchsize(
                    args, gpu_dict[key], 
                    seq_len, decode_len, 
                    expert_num=expert_num, tp=tp_num)
                if bs > max_bs:
                    continue
                else:
                    df.loc[len(df)] = [gpu_dict[key].gpu_type, bs,
                                       tp_num, dispatch_time, combine_time]
    if print_console:
        df['BatchSize'] = df['BatchSize'].astype(int).astype(str)
        print(df[df['TP'] == 1][['GPU', 'BatchSize', 'Dispatch', 'Combine']].set_index(
            'GPU').to_markdown(floatfmt=".3f"))
    return df


def _decode_time(args: ModelArgs, gpu,
                 bs_list, seq_len, decode_len,
                 gemm_group_per_device,
                 device_num,
                 mbs=2,
                 tp_list=None,
                 fp8_combine=False,
                 print_console=False):

    expert_per_device = gemm_group_per_device + 1  # add shared expert
    mla = decode_attention(args, gpu, bs_list, seq_len,
                           decode_len,
                           expert_num=expert_per_device,
                           tp_list=tp_list)
    dense_mlp = decode_dense_mlp(
        args, gpu, bs_list, seq_len, decode_len,
        expert_num=expert_per_device,
        tp_list=tp_list)
    moe = decode_moe_expert(args, gpu, bs_list, seq_len,
                            decode_len, mbs=mbs,
                            gemm_group_per_device=gemm_group_per_device,
                            device_num=device_num,
                            tp_list=tp_list)
    a2a = decode_a2a(args, gpu, bs_list, seq_len, decode_len,
                     expert_num=expert_per_device, device_num=device_num,
                     tp_list=tp_list,
                     fp8_combine=fp8_combine, mbs=mbs)
    dfs = [mla, dense_mlp, moe, a2a]

    for decode_df in dfs:
        decode_df['BatchSize'] = decode_df['BatchSize'].astype(int).astype(str)
    df = reduce(lambda left, right: pd.merge(left, right, on=[
                'GPU', 'BatchSize', 'TP'], how='left'), dfs)
    if print_console:
        print(df.set_index('GPU').to_markdown(floatfmt=".3f"))
    return df


def decode_time(args: ModelArgs, gpu_dict,
                bs_list, seq_len, decode_len,
                gemm_group_per_device,
                device_num,
                tps_limit=0,
            tp_list=None,
                fp8_combine=False,
                print_console=False):

    df = _decode_time(args, gpu_dict, bs_list, seq_len, decode_len,
                      gemm_group_per_device=gemm_group_per_device,
                      device_num=device_num,
                tp_list=tp_list,
                      fp8_combine=fp8_combine)

    def overlap_adjust(r):
        if r['Delta'] > 0:
            return r['TPOT_O'] + r['Delta'] * (args.n_layers - args.n_dense_layers)
        else:
            return r['TPOT_O']

    # 修正TP执行时间, 按照加载FP8的KV计算
    df['DenseMLA'] = df['DenseMLA'] + df['LoadKVPerDevice']
    df['SparseMLA'] = df['SparseMLA'] + df['LoadKVPerDevice']
    df['COMP_SUM'] = df['SparseMLA'] + df['SharedExpert'] + df['RoutedExpert']
    df['COMM_SUM'] = df['Dispatch'] + df['Combine']
    df['Delta'] = df['COMM_SUM'] - df['SparseMLA'] - df['SharedExpert']
    df['TPOT_O'] = (df['DenseMLA'] + df['DenseMLP']) * args.n_dense_layers
    df['TPOT_O'] += (df['SparseMLA'] + df['SharedExpert'] +
                     df['RoutedExpert']) * (args.n_layers - args.n_dense_layers)

    df['TPOT'] = df.apply(lambda row:  overlap_adjust(row), axis=1)
    df = df[['GPU', 'TP', 'BatchSize', 'LoadKVPerDevice', 'DenseMLA', 'DenseMLP', 'SparseMLA', 'Combine',
             'SharedExpert', 'RoutedExpert', 'Dispatch', 'COMP_SUM', 'COMM_SUM', 'Delta', 'TPOT', 'TPOT_O']]
    df['TPS'] = 1000 / df['TPOT']
    df['TPS_O'] = 1000 / df['TPOT_O']
    df['Total'] = df['TPS'] * df['BatchSize'].astype(int)
    df['Total_O'] = df['TPS_O'] * df['BatchSize'].astype(int)
    df['Comm_Impact'] = (df['Total_O'] - df['Total']) / df['Total_O']

    df = df[df['TPS'] > tps_limit]
    if print_console:
        print(df.set_index('GPU').T.to_markdown(floatfmt=".3f"))
    return df


def decode_time_with_ep_list(args: ModelArgs, gpu_dict,
                             config: Config,
                             tps_limit=0,
                             tp_list=None,
                             fp8_combine=False,
                             print_console=False):
    df_list = []
    for device_num in config.eplist:
        gemm_group_per_device = math.ceil(args.n_routed_experts / device_num)
        df = decode_time(args, gpu_dict, config.bs_list, config.seq_len,
                         config.decode_len,
                         gemm_group_per_device=gemm_group_per_device,
                         device_num=device_num,
                         tp_list=tp_list,
                         fp8_combine=fp8_combine,
                         tps_limit=tps_limit,
                         print_console=False)
        df['EP'] = device_num
        df_list.append(df)
    dd = pd.concat(df_list)
    dd.reset_index(inplace=True, drop=True)
    order = ['GPU', 'TP', 'EP', 'BatchSize', 'LoadKVPerDevice', 'DenseMLA', 'DenseMLP', 'SparseMLA',
             'Combine', 'SharedExpert', 'RoutedExpert', 'Dispatch', 'COMP_SUM',
             'COMM_SUM', 'Delta', 'TPOT', 'TPOT_O', 'TPS', 'TPS_O', 'Total',
             'Total_O', 'Comm_Impact']
    dd = dd[order]
    dd['BatchSize'] = dd['BatchSize'].astype(int)
    return dd


def df_filter(df,gpu,device_num=0, bs=0,tps_limit=0, value_list=[]):
    df_o = df[df['GPU'] == gpu] 
    if bs > 0:
        df_o = df_o[df_o['BatchSize'] == str(bs)]
    if tps_limit > 0:
        df_o = df_o[df_o['TPS'] > tps_limit]
    if device_num > 0:
        df_o = df_o[df_o['EP'] == device_num]
    if len(value_list) > 0:
        df_o = df_o[value_list]
    return df_o


def decode_disaggregated_time(args: ModelArgs,
                              attn_gpu: GPU_perf,
                              ffn_gpu: GPU_perf,
                              bs_list,
                              seq_len,
                              decode_len,
                              gemm_group_per_device,
                              device_num,
                              tps_limit=0,
                              tp_list=None,
                              fp8_combine=False,
                              cx9_bw=100,
                              cx9_latency=0.005,
                              print_console=False):
    if tp_list is None:
        tp_list = [1, 2, 4, 8]

    expert_per_device = gemm_group_per_device + 1
    mla = decode_attention(args, {attn_gpu.gpu_type: attn_gpu},
                           bs_list, seq_len, decode_len,
                           expert_num=expert_per_device,
                           tp_list=tp_list)
    dense_mlp = decode_dense_mlp(args, {ffn_gpu.gpu_type: ffn_gpu},
                                 bs_list, seq_len, decode_len,
                                 expert_num=expert_per_device,
                                 tp_list=tp_list)
    moe = decode_moe_expert(args, {ffn_gpu.gpu_type: ffn_gpu},
                            bs_list, seq_len, decode_len,
                            gemm_group_per_device=gemm_group_per_device,
                            device_num=device_num,
                            tp_list=tp_list)
    a2a = decode_a2a(args, {ffn_gpu.gpu_type: ffn_gpu},
                     bs_list, seq_len, decode_len,
                     expert_num=expert_per_device,
                     device_num=device_num,
                     tp_list=tp_list,
                     fp8_combine=fp8_combine)

    dfs = [mla, dense_mlp, moe, a2a]
    for decode_df in dfs:
        decode_df['BatchSize'] = decode_df['BatchSize'].astype(int).astype(str)

    df = reduce(lambda left, right: pd.merge(left, right, on=['BatchSize', 'TP'], how='inner'),
                [mla[['BatchSize', 'TP', 'LoadKVPerDevice', 'DenseMLA', 'SparseMLA']],
                 dense_mlp[['BatchSize', 'TP', 'DenseMLP']],
                 moe[['BatchSize', 'TP', 'SharedExpert', 'RoutedExpert']],
                 a2a[['BatchSize', 'TP', 'Dispatch', 'Combine']]])

    def overlap_adjust(r):
        if r['Delta'] > 0:
            return r['TPOT_O'] + r['Delta'] * (args.n_layers - args.n_dense_layers)
        else:
            return r['TPOT_O']

    df['Attn_GPU'] = attn_gpu.gpu_type
    df['FFN_GPU'] = ffn_gpu.gpu_type
    df['DenseMLA'] = df['DenseMLA'] + df['LoadKVPerDevice']
    df['SparseMLA'] = df['SparseMLA'] + df['LoadKVPerDevice']
    df['COMP_SUM'] = df['SparseMLA'] + df['SharedExpert'] + df['RoutedExpert']
    df['COMM_SUM'] = df['Dispatch'] + df['Combine']
    df['Delta'] = df['COMM_SUM'] - df['SparseMLA'] - df['SharedExpert']
    df['TPOT_O'] = (df['DenseMLA'] + df['DenseMLP']) * args.n_dense_layers
    df['TPOT_O'] += (df['SparseMLA'] + df['SharedExpert'] +
                     df['RoutedExpert']) * (args.n_layers - args.n_dense_layers)
    df['TPOT'] = df.apply(lambda row: overlap_adjust(row), axis=1)

    batch_sizes = df['BatchSize'].astype(int)
    df['InterStage'] = [inter_stage_link_time(bs, args, tp, link_bw=cx9_bw,
                                              link_latency=cx9_latency)
                        for bs, tp in zip(batch_sizes, df['TP'])]
    df['TPOT_Disagg'] = df['TPOT'] + df['InterStage'] * args.n_layers
    df['TPS'] = 1000 / df['TPOT']
    df['TPS_O'] = 1000 / df['TPOT_O']
    df['TPS_Disagg'] = 1000 / df['TPOT_Disagg']
    df['Total'] = df['TPS'] * batch_sizes
    df['Total_O'] = df['TPS_O'] * batch_sizes
    df['Total_Disagg'] = df['TPS_Disagg'] * batch_sizes
    df['Comm_Impact'] = (df['Total_O'] - df['Total']) / df['Total_O']
    df['Disagg_Impact'] = (df['Total'] - df['Total_Disagg']) / df['Total']

    df = df[df['TPS_Disagg'] > tps_limit]
    order = ['Attn_GPU', 'FFN_GPU', 'TP', 'BatchSize', 'LoadKVPerDevice', 'DenseMLA', 'DenseMLP',
             'SparseMLA', 'Combine', 'SharedExpert', 'RoutedExpert', 'Dispatch',
             'InterStage', 'COMP_SUM', 'COMM_SUM', 'Delta', 'TPOT', 'TPOT_O',
             'TPOT_Disagg', 'TPS', 'TPS_O', 'TPS_Disagg', 'Total', 'Total_O',
             'Total_Disagg', 'Comm_Impact', 'Disagg_Impact']
    df = df[order]
    if print_console:
        print(df.to_markdown(floatfmt='.3f'))
    return df


def profile_rubin_nvl72_disaggregated(args: ModelArgs,
                                      config: Config,
                                      gpu_dict,
                                      decode_ffn_cabinet_list,
                                      tp_list=None,
                                      attn_gpu_type='Rubin-NVL72',
                                      ffn_gpu_type='Rubin-NVL72',
                                      cx9_bw=100,
                                      cx9_latency=0.005,
                                      fp8_combine=False,
                                      tps_limit=0,
                                      print_console=False):
    """Profile a split deployment.

    Prefill attention/FFN and decode attention stay on one Rubin-NVL72 cabinet.
    Decode FFN is placed on one or more Rubin-NVL72 cabinets connected via CX9.
    """
    if tp_list is None:
        tp_list = [1, 2, 4, 8]
    tp_list = [tp for tp in tp_list if 0 < tp <= 8]
    if len(tp_list) == 0:
        raise ValueError('tp_list must contain at least one TP value within 1..8')

    attn_gpu = gpu_dict[attn_gpu_type]
    ffn_gpu = gpu_dict[ffn_gpu_type]
    min_cabinet_num = min_cabinet_count_for_decode_ffn(args, ffn_gpu)

    prefill_rows = []
    decode_frames = []
    for tp in tp_list:
        dp = max(1, attn_gpu.gpu_per_node // tp)
        _, prefill_summary = prefill_time(args,
                                          {attn_gpu.gpu_type: attn_gpu},
                                          config.seq_len,
                                          config.kv_cache_rate,
                                          tp=tp,
                                          dp=dp,
                                          print_console=False)
        prefill_rows.append({
            'Prefill_GPU': attn_gpu.gpu_type,
            'TP': tp,
            'DP': dp,
            'Prefill_Cabinets': 1,
            'Prefill_Total': prefill_summary.at['Sum', attn_gpu.gpu_type]
        })

    for cabinet_num in decode_ffn_cabinet_list:
        device_num = cabinet_num * ffn_gpu.gpu_per_node
        gemm_group_per_device = max(1, math.ceil(args.n_routed_experts / device_num))
        df = decode_disaggregated_time(args,
                                       attn_gpu,
                                       ffn_gpu,
                                       config.bs_list,
                                       config.seq_len,
                                       config.decode_len,
                                       gemm_group_per_device=gemm_group_per_device,
                                       device_num=device_num,
                                       tps_limit=tps_limit,
                                       tp_list=tp_list,
                                       fp8_combine=fp8_combine,
                                       cx9_bw=cx9_bw,
                                       cx9_latency=cx9_latency,
                                       print_console=False)
        if df.empty:
            continue
        df['FFN_Cabinets'] = cabinet_num
        df['FFN_Min_Cabinets'] = min_cabinet_num
        df['Weight_Headroom'] = cabinet_num - min_cabinet_num
        df['CX9_BW_GBs'] = cx9_bw
        df['CX9_Latency_ms'] = cx9_latency
        decode_frames.append(df)

    prefill_df = pd.DataFrame(prefill_rows)
    if decode_frames:
        decode_df = pd.concat(decode_frames).reset_index(drop=True)
    else:
        decode_df = pd.DataFrame(columns=['Attn_GPU', 'FFN_GPU', 'TP', 'BatchSize',
                                          'FFN_Cabinets', 'FFN_Min_Cabinets'])

    if print_console:
        print(prefill_df.to_markdown(floatfmt='.3f'))
        if not decode_df.empty:
            print(decode_df.to_markdown(floatfmt='.3f'))
    return prefill_df, decode_df



def df_sort(df,value,ascending=False):
    if ascending:
        df_o = df.groupby(['GPU','BatchSize','EP'],as_index=False)\
            .apply(lambda t: t[t[value]==t[value].min()]).sort_values([value],ascending=True).reset_index(drop=True)
    else:
        df_o = df.groupby(['GPU','BatchSize','EP'],as_index=False)\
            .apply(lambda t: t[t[value]==t[value].max()]).sort_values([value],ascending=False).reset_index(drop=True)
    return df_o

def color_negative_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'red' if val < 0 else 'black'
    return 'color: %s' % color

def color_positive_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for positive
    strings, black otherwise.
    """
    color = 'red' if val > 0 else 'black'
    return 'color: %s' % color

def gpu_category_color(s,props):
    colors = ['color:darkred','color:steelblue','color:green','color:black','color:m',
              'color:darkgoldenrod','color:darkgreen','color:crimson','color:brwon','color:sienna',
              'color:navy','color:pink','color:gray','color:darkviolet']
    gpu_idx = props[s]
    return colors[gpu_idx]

def highlight_max(data, color='yellow'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)



def draw(df, gpu_dict,
         comp_name,comp_val_list, 
         val_list, val_unit_name,
         title, width=8, height=4,
         filename='',savefig=False):
    def _df_filter(df, gpu, comp_name, comp_val, value_list):
        df1 = df[(df['GPU'] == gpu) & (df[comp_name] == comp_val)][value_list]
        return df1
    
    sns.color_palette("Paired")
    num_gpu = len(gpu_dict)
    fig_height = height * num_gpu
    fig_width = width * len(val_list)
    fig, axs = plt.subplots(nrows=num_gpu, ncols=len(val_list), figsize=(fig_width, fig_height))

    fig.suptitle(title, y=0.9,fontsize='large')
    value_list = val_list + ['index_value']
    cnt = 0
    for key in gpu_dict.keys():
        axt = axs[cnt]
        for i in range(0,len(val_list)):
            axt[i].legend(comp_val_list)
        for comp_v in comp_val_list:
            df1 = _df_filter(df, key , comp_name, comp_v, value_list)
            for i in range(0,len(val_list)):
                sns.lineplot(x='index_value', y=val_list[i],label=str(comp_v), data=df1,  ax=axt[i])
                axt[i].set_ylabel(val_unit_name)
                axt[i].set_xlabel(key+'('+val_list[i]+')')
        cnt += 1

    plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=None, wspace=0.2, hspace=0.2)
    if savefig:
        if filename=="":
            filename = './figures/'+title.replace(' ','_')+'.png'
        plt.savefig(filename,bbox_inches='tight', pad_inches=0.05)
    plt.show()