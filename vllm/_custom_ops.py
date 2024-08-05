from typing import Dict, Optional, Tuple
from tqdm.auto import tqdm

import torch

from vllm.benchmark import BENCHMARKER
try:
    from vllm._C import cache_ops as vllm_cache_ops
    from vllm._C import ops as vllm_ops
    from vllm._C import kvc_ops
except ImportError:
    pass


# activation ops
def silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    vllm_ops.silu_and_mul(out, x)


def gelu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    vllm_ops.gelu_and_mul(out, x)


def gelu_tanh_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    vllm_ops.gelu_tanh_and_mul(out, x)


def gelu_fast(out: torch.Tensor, x: torch.Tensor) -> None:
    vllm_ops.gelu_fast(out, x)


def gelu_new(out: torch.Tensor, x: torch.Tensor) -> None:
    vllm_ops.gelu_new(out, x)


# page attention ops
@BENCHMARKER.wrap()
def paged_attention_v1(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    max_context_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    kv_scale: float,
) -> None:
    vllm_ops.paged_attention_v1(out, query, key_cache, value_cache,
                                num_kv_heads, scale, block_tables,
                                context_lens, block_size, max_context_len,
                                alibi_slopes, kv_cache_dtype, kv_scale)


@BENCHMARKER.wrap()
def paged_attention_v2(
    out: torch.Tensor,
    exp_sum: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    max_context_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    kv_scale: float,
) -> None:
    vllm_ops.paged_attention_v2(out, exp_sum, max_logits, tmp_out, query,
                                key_cache, value_cache, num_kv_heads, scale,
                                block_tables, context_lens, block_size,
                                max_context_len, alibi_slopes, kv_cache_dtype,
                                kv_scale)


# paged attention ops with KV-Compress cache
@BENCHMARKER.wrap()
def paged_attention_kvc_v1(
    out: torch.Tensor,
    kv_metric_out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    kv_position: torch.Tensor,
    last_position: torch.Tensor,
    kv_metric_buffer_len: torch.Tensor,
    block_size: int,
    max_context_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    kv_scale: float,
) -> None:
    vllm_ops.kvcompress_paged_attention_v1(out, kv_metric_out, query, key_cache,
                                           value_cache, num_kv_heads, scale,
                                           block_tables, context_lens,
                                           kv_position, last_position,
                                           kv_metric_buffer_len, block_size,
                                           max_context_len, alibi_slopes,
                                           kv_cache_dtype, kv_scale)


@BENCHMARKER.wrap()
def paged_attention_kvc_v2(
    out: torch.Tensor,
    kv_metric_out: torch.Tensor,
    exp_sum: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    tmp_kv_metric_out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    kv_position: torch.Tensor,
    last_position: torch.Tensor,
    kv_metric_buffer_len: torch.Tensor,
    block_size: int,
    max_context_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    kv_scale: float,
) -> None:
    vllm_ops.kvcompress_paged_attention_v2(out, kv_metric_out, exp_sum,
                                           max_logits, tmp_out,
                                           tmp_kv_metric_out,
                                           query, key_cache, value_cache,
                                           num_kv_heads, scale, block_tables,
                                           context_lens, kv_position,
                                           last_position, kv_metric_buffer_len,
                                           block_size, max_context_len,
                                           alibi_slopes, kv_cache_dtype,
                                           kv_scale)


# pos encoding ops
def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    vllm_ops.rotary_embedding(positions, query, key, head_size, cos_sin_cache,
                              is_neox)


def batched_rotary_embedding(positions: torch.Tensor, query: torch.Tensor,
                             key: torch.Tensor, head_size: int,
                             cos_sin_cache: torch.Tensor, is_neox: bool,
                             rot_dim: int,
                             cos_sin_cache_offsets: torch.Tensor) -> None:
    vllm_ops.batched_rotary_embedding(positions, query, key, head_size,
                                      cos_sin_cache, is_neox, rot_dim,
                                      cos_sin_cache_offsets)


# layer norm ops
def rms_norm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
             epsilon: float) -> None:
    vllm_ops.rms_norm(out, input, weight, epsilon)


def fused_add_rms_norm(input: torch.Tensor, residual: torch.Tensor,
                       weight: torch.Tensor, epsilon: float) -> None:
    vllm_ops.fused_add_rms_norm(input, residual, weight, epsilon)


# quantization ops
# awq
def awq_dequantize(qweight: torch.Tensor, scales: torch.Tensor,
                   zeros: torch.Tensor, split_k_iters: int, thx: int,
                   thy: int) -> torch.Tensor:
    return vllm_ops.awq_dequantize(qweight, scales, zeros, split_k_iters, thx,
                                   thy)


def awq_gemm(input: torch.Tensor, qweight: torch.Tensor, qzeros: torch.Tensor,
             scales: torch.Tensor, split_k_iters: int) -> torch.Tensor:
    return vllm_ops.awq_gemm(input, qweight, qzeros, scales, split_k_iters)


# gptq
def gptq_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
              b_gptq_qzeros: torch.Tensor, b_gptq_scales: torch.Tensor,
              b_g_idx: torch.Tensor, use_exllama: bool,
              bit: int) -> torch.Tensor:
    return vllm_ops.gptq_gemm(a, b_q_weight, b_gptq_qzeros, b_gptq_scales,
                              b_g_idx, use_exllama, bit)


def gptq_shuffle(q_weight: torch.Tensor, q_perm: torch.Tensor,
                 bit: int) -> None:
    vllm_ops.gptq_shuffle(q_weight, q_perm, bit)


# squeezellm
def squeezellm_gemm(vec: torch.Tensor, mat: torch.Tensor, mul: torch.Tensor,
                    lookup_table: torch.Tensor) -> None:
    vllm_ops.squeezellm_gemm(vec, mat, mul, lookup_table)


# marlin
def marlin_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
                b_scales: torch.Tensor, workspace: torch.Tensor, size_m: int,
                size_n: int, size_k: int) -> torch.Tensor:
    return vllm_ops.marlin_gemm(a, b_q_weight, b_scales, workspace, size_m,
                                size_n, size_k)


# fp8
def scaled_fp8_quant(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    scale = torch.zeros(1, device=input.device, dtype=torch.float32)
    output = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    vllm_ops.scaled_fp8_quant(output, input, scale)
    return output, scale


# moe
def moe_align_block_size(topk_ids: torch.Tensor, num_experts: int,
                         block_size: int, sorted_token_ids: torch.Tensor,
                         experts_ids: torch.Tensor,
                         num_tokens_post_pad: torch.Tensor) -> None:
    vllm_ops.moe_align_block_size(topk_ids, num_experts, block_size,
                                  sorted_token_ids, experts_ids,
                                  num_tokens_post_pad)


@BENCHMARKER.wrap()
def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    kv_scale: float,
) -> None:
    vllm_cache_ops.reshape_and_cache(key, value, key_cache, value_cache,
                                     slot_mapping, kv_cache_dtype, kv_scale)


@BENCHMARKER.wrap()
def reshape_and_cache_kvc(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    kv_metrics: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_metric_head_bias: torch.Tensor,
    kv_cache_dtype: str,
    kv_scale: float,
) -> None:
    vllm_cache_ops.kvcompress_reshape_and_cache(key, value, key_cache,
                                                value_cache, kv_metrics,
                                                slot_mapping,
                                                kv_metric_head_bias,
                                                kv_cache_dtype, kv_scale)


def copy_blocks(key_caches: torch.Tensor, value_caches: torch.Tensor,
                block_mapping: torch.Tensor) -> None:
    vllm_cache_ops.copy_blocks(key_caches, value_caches, block_mapping)


def swap_blocks(src: torch.Tensor, dst: torch.Tensor,
                block_mapping: Dict[int, int]) -> None:
    vllm_cache_ops.swap_blocks(src, dst, block_mapping)


def convert_fp8(output: torch.Tensor, input: torch.Tensor) -> None:
    vllm_cache_ops.convert_fp8(output, input)


#TODO: cuda_utils, custom_ar

# KV-Compress
MAX_INT = 2147483000
def ref_schedule_cache_evictions(
    out_evicted_kv_indices: torch.Tensor,
    out_evicted_logical_indices: torch.Tensor,
    out_evicted_kv_count: torch.Tensor,
    remaining_token_count: torch.Tensor,
    evicted_kv_offsets: torch.Tensor,
    sorted_indices: torch.Tensor,
    seq_block_offsets: torch.Tensor,
    layer_by_block: torch.Tensor,
    head_by_block: torch.Tensor,
    virtual_block_num_by_block: torch.Tensor,
    evicted_blocks_per_seq: torch.Tensor,
    context_lens: torch.Tensor,
    hanging_token_count: torch.Tensor,
    kv_position: torch.Tensor,
    last_position: torch.Tensor,
    protected_window: torch.Tensor,
    block_size: int,
    evict_evenly_per_layer: bool,
    control_layers: torch.Tensor,
    max_evicted_kv: int,
    null_eviction_index: int,
    truncate: bool,
):
    if evict_evenly_per_layer:
        raise NotImplementedError
    # assert (
    #     head_by_block[16] != head_by_block[18]
    #     or layer_by_block[16] != layer_by_block[18]
    #     or virtual_block_num_by_block[16] != virtual_block_num_by_block[18]
    #     or sorted_indices[16] % block_size != sorted_indices[18] % block_size
    # )
    # print(head_by_block[16] != head_by_block[21])
    # print(layer_by_block[16] != layer_by_block[21])
    # print(virtual_block_num_by_block[16] != virtual_block_num_by_block[21])
    # print(sorted_indices[16] % block_size != sorted_indices[21] % block_size)
    # for j in [16, 21]:
    #     block_num = sorted_indices[j] // block_size
    #     block_offset = sorted_indices[j] % block_size

    #     layer_idx = layer_by_block[block_num]
    #     head_idx = head_by_block[block_num]
    #     virtual_block_num = virtual_block_num_by_block[block_num]

    #     kv_idx = virtual_block_num * block_size + block_offset
    #     print(f'DEBUG: {kv_idx}')

    num_seqs, num_layers, num_kv_heads = out_evicted_kv_count.shape
    total_blocks, = layer_by_block.shape
    out_evicted_kv_indices.fill_(MAX_INT)
    out_evicted_kv_count.fill_(0)
    # print(f'hanging_toks:\n{hanging_token_count}')
    # print(f'sequence_block_offsets:\n{seq_block_offsets}')
    # print(f'evicted_blocks_per_seq:\n{evicted_blocks_per_seq}')
    # print(f'{out_evicted_logical_indices.shape=}')
    # print(f'evicted_kv_offsets:\n{evicted_kv_offsets}')
    # print(f'context_lens:\n{context_lens}')

    # evictable_kv = torch.zeros_like(context_lens)
    # next_seq_offset = 0
    # seq_idx = -1
    # for i in range(kv_position.size(0)):
    #     if i >= next_seq_offset:
    #         seq_idx += 1
    #         next_seq_offset = seq_block_offsets[seq_idx + 1] if seq_idx < seq_block_offsets.size(0) - 1 else float('inf')
    #     evictable_kv[seq_idx, layer_by_block[i], head_by_block[i]] += (
    #         (kv_position >= 0) & (kv_position <= last_position[0] - protected_window)
    #     )[i].sum()

    # print(f'evictable_keys:\n{evictable_kv}')
    pos_flat = kv_position.flatten()

    for i in tqdm(range(num_seqs)):
        # blocks_to_evict = min(evicted_blocks_per_seq[i], )
        remaining_kv = torch.ones((num_layers, num_kv_heads), device=hanging_token_count.device) * hanging_token_count[i]
        evicted_blocks = 0
        tot_iter = torch.zeros_like(remaining_kv)
        current_pos = last_position[i]
        j = seq_block_offsets[i] * block_size
        end_j = (seq_block_offsets[i+1] if i+1 < len(seq_block_offsets) else total_blocks) * block_size
        print(f'start, end: {j}, {end_j}')
        total_evictable_keys = end_j - j
        assert total_evictable_keys % block_size == 0
        protected_blocks = ((protected_window[i] + block_size - 1) // block_size) * num_layers * num_kv_heads
        max_evictable_blocks = total_evictable_keys // block_size
        max_evictable_blocks -= protected_blocks
        assert max_evictable_blocks >= evicted_blocks_per_seq[i], f"Expected at least {evicted_blocks_per_seq[i]} evictable blocks, got {max_evictable_blocks}"
        evicted_kv_indices = []
        while evicted_blocks < evicted_blocks_per_seq[i] and j < end_j:
            block_num = sorted_indices[j] // block_size
            block_offset = sorted_indices[j] % block_size

            layer_idx = layer_by_block[block_num]
            head_idx = head_by_block[block_num]
            virtual_block_num = virtual_block_num_by_block[block_num]
            kv_pos = kv_position[block_num, block_offset]

            kv_idx = virtual_block_num * block_size + block_offset
            # print(f'kv_idx: {sorted_indices[j]}, v_kv_idx: {kv_idx}, j: {j}, i: {i}, l: {layer_idx}, h: {head_idx}, evicted: {out_evicted_kv_count[i, layer_idx, head_idx] + 1}, remaining: {remaining_kv[layer_idx, head_idx]}, blk#: {block_num}, blk%: {block_offset}')
            #assert kv_idx.item() not in {x.item() for x in out_evicted_kv_indices[i, layer_idx, head_idx]}
            if kv_idx >= context_lens[i, layer_idx, head_idx].item():
                j += 1
                continue

            if kv_pos > current_pos - protected_window[i]:
                j += 1
                continue

            evicted_kv_indices.append(sorted_indices[j])

            evicted_idx = evicted_kv_offsets[i, layer_idx, head_idx] + out_evicted_kv_count[i, layer_idx, head_idx]
            out_evicted_logical_indices[evicted_idx] = kv_idx
            out_evicted_kv_indices[evicted_idx] = head_idx + num_kv_heads * (layer_idx + num_layers * i)
            out_evicted_kv_count[i, layer_idx, head_idx] += 1

            j += 1
            remaining_kv[layer_idx, head_idx] -= 1
            if remaining_kv[layer_idx, head_idx] == 0:
                remaining_kv[layer_idx, head_idx] = block_size
                evicted_blocks += 1
                assert (out_evicted_kv_count > 0).any()
                # print(f'evicted {evicted_blocks}/{evicted_blocks_per_seq[i]} blocks')

            tot_iter[layer_idx, head_idx] += 1

        # print(f'tot_iter: {tot_iter}')
        # print(f'remaining_kv:\n{remaining_kv}')
        assert evicted_blocks == evicted_blocks_per_seq[i], "schedule_cache_evictions loop failed"
        assert pos_flat[torch.tensor(evicted_kv_indices, dtype=torch.long)].max() <= (current_pos) - protected_window[i], "schedule_cache_evictions loop failed"

    # print(f'evicted_count pre-truncate:\n{out_evicted_kv_count}')
    if truncate:
        no_evicted_kv = out_evicted_kv_count < hanging_token_count
        out_evicted_kv_count[:] = torch.where(
            no_evicted_kv,
            torch.zeros_like(out_evicted_kv_count),
            out_evicted_kv_count - (out_evicted_kv_count - hanging_token_count) % block_size,
        )

        alloc_kv_count = ((context_lens + block_size - 1) // block_size) * block_size
        for i in range(num_seqs):
            for layer_idx in range(num_layers):
                for head_idx in range(num_kv_heads):
                    start_offset = evicted_kv_offsets[i, layer_idx, head_idx] + out_evicted_kv_count[i, layer_idx, head_idx]
                    end_offset = evicted_kv_offsets[i, layer_idx, head_idx] + alloc_kv_count[i, layer_idx, head_idx]
                    # print(f'({i}, {layer_idx}, {head_idx}): {start_offset=}, {end_offset=}, {alloc_kv_count[i, layer_idx, head_idx]=}')
                    out_evicted_logical_indices[start_offset:end_offset] = null_eviction_index
                    out_evicted_kv_indices[start_offset:end_offset] = head_idx + num_kv_heads * (layer_idx + num_layers * i)

    print(f'evicted_count:\n{out_evicted_kv_count}')
    for i in range(num_seqs):
        for l in range(num_layers):
            for h in range(num_kv_heads):
                offset = evicted_kv_offsets[i,l,h]
                evicted = out_evicted_kv_count[i,l,h]
                evicted_logical_indices = out_evicted_logical_indices[offset:offset+evicted]
                print(evicted_logical_indices)
                assert not (evicted_logical_indices == MAX_INT).any()


@BENCHMARKER.wrap()
def schedule_cache_evictions(
    out_evicted_kv_indices: torch.Tensor,
    out_evicted_logical_indices: torch.Tensor,
    out_evicted_kv_count: torch.Tensor,
    out_evicted_kv_offsets: torch.Tensor,
    sorted_indices: torch.Tensor,
    seq_block_offsets: torch.Tensor,
    layer_by_block: torch.Tensor,
    head_by_block: torch.Tensor,
    logical_block_num_by_block: torch.Tensor,
    evicted_blocks_per_seq: torch.Tensor,
    context_lens: torch.Tensor,
    hanging_token_count: torch.Tensor,
    kv_position: torch.Tensor,
    last_position: torch.Tensor,
    protected_window_size: torch.Tensor,
    block_size: int,
    max_evicted_kv: int,
    null_eviction_index: int,
    truncate: bool,
    evict_evenly_per_layer: bool = False,
    control_layers: Optional[torch.Tensor] = None,
) -> None:
    if evict_evenly_per_layer and block_size > 1:
        raise RuntimeError(f"cannot evict evenly across layers when block_size > 1 (got {block_size=})")
    # if evict_evenly_per_layer:
    #     assert (evicted_blocks_per_seq % context_lens.size(0) == 0).all()

    # never filter evictions by protected window
    # print(f'{kv_position.shape=},{last_position=},{kv_position.max()=},{protected_window_size=}')
    # print(f'{kv_position=}')
    # last_position[:] = kv_position.max() + protected_window_size
    # raise
    # TODO PPL increasing as protected window increased below
    # protected_window_size = 0
    out_evicted_kv_indices.zero_()
    # print(max_evicted_kv)
    # print(seq_block_offsets.max(), seq_block_offsets.max() * block_size)
    # for i in range(seq_block_offsets.size(0) - 1):
    #     curr_offset = seq_block_offsets[i]
    #     next_offset = seq_block_offsets[i+1]
    #     curr_blocks = ((context_lens[:,i] + block_size - 1) // block_size).sum()
    #     assert curr_offset + curr_blocks == next_offset, f"{curr_offset=}, {curr_blocks=}, {next_offset=}, {(curr_offset + curr_blocks)=}"
    # curr_offset = seq_block_offsets[i-1]
    # curr_blocks = ((context_lens[:,i-1] + block_size - 1) // block_size).sum()
    # assert curr_offset + curr_blocks <= max_evicted_kv, f"{curr_offset=}, {curr_blocks=}, {max_evicted_kv=}, {(curr_offset + curr_blocks)=}"

    kvc_ops.schedule_cache_evictions(
    # ref_schedule_cache_evictions(
        out_evicted_kv_indices,
        out_evicted_logical_indices,
        out_evicted_kv_count,
        hanging_token_count.clone(),
        out_evicted_kv_offsets,
        sorted_indices.type(torch.int).contiguous(),
        seq_block_offsets.type(torch.int).contiguous(),
        layer_by_block.contiguous(),
        head_by_block.contiguous(),
        logical_block_num_by_block.contiguous(),
        evicted_blocks_per_seq.contiguous(),
        context_lens.transpose(0, 1).contiguous(),
        hanging_token_count.contiguous(),
        kv_position.type(torch.int),
        last_position.type(torch.int),
        protected_window_size,
        block_size,
        evict_evenly_per_layer,
        control_layers,
        max_evicted_kv,
        null_eviction_index,
        truncate,
    )
    # TODO fails half the time when protected_window is set to 50 above
    # assert not (out_evicted_kv_indices > last_position.max() - protected_window_size).any()


def validate_evictions(evicted_kv_indices, evicted_kv_count, max_int):
    valid_mask = evicted_kv_indices < max_int
    selected_mask = (
        torch.arange(evicted_kv_indices.shape[-1],
                     device=evicted_kv_indices.device,
                     dtype=evicted_kv_indices.dtype)[None,None,None]
        < evicted_kv_count[...,None]
    )
    assert not (selected_mask & ~valid_mask).any()


def ref_count_block_evictions(
    evicted_block_count: torch.Tensor,
    evicted_logical_indices: torch.Tensor,
    evicted_kv_offsets: torch.Tensor,
    hanging_token_count: torch.Tensor,
    block_size: int,
    null_value: int,
    evicted_blocks_per_seq: Optional[torch.Tensor] = None,
    debug_counts: Optional[torch.Tensor] = None,
    debug_logical_indices = None,
):
    end_total = evicted_logical_indices.size(0)
    for s in range(evicted_kv_offsets.size(0)):
        end_s = evicted_kv_offsets[s+1,0,0] if s+1 < evicted_kv_offsets.size(0) else end_total
        for l in range(evicted_kv_offsets.size(1)):
            end_l = evicted_kv_offsets[s,l+1,0] if l+1 < evicted_kv_offsets.size(1) else end_s
            for h in range(evicted_kv_offsets.size(2)):
                start = evicted_kv_offsets[s,l,h]
                end = evicted_kv_offsets[s,l,h+1] if h+1 < evicted_kv_offsets.size(2) else end_l
                curr_slice = evicted_logical_indices[start:end]
                assert ((curr_slice.view(-1, block_size) == null_value).any(dim=-1) == (curr_slice.view(-1, block_size) == null_value).all(dim=-1)).all()
                assert (curr_slice[:(curr_slice != null_value).sum()] != null_value).all()
                evicted_kvs = (curr_slice != null_value).sum()
                evicted_block_count[s,l,h] = (
                    evicted_kvs // block_size
                )

                if s == l == h == 0:
                    print(f'hiiiii: {hanging_token_count[s,l,h]=}, {end=}, {end-block_size+hanging_token_count[s,l,h]=}')
                    print(evicted_logical_indices[end-block_size+hanging_token_count[s,l,h]:end])
                # set logical indices for empty KV slots in last evicted block to null_value
                end_evict = start + evicted_kvs
                evicted_logical_indices[end_evict-block_size+hanging_token_count[s,l,h]:end_evict] = null_value

                if s == l == h == 0 and block_size - hanging_token_count[s,l,h] > 0:
                    print(f'heyo {(debug_logical_indices != evicted_logical_indices).any()=}\n{evicted_logical_indices[min(end-block_size*2,0):end]}')
                    assert (evicted_logical_indices != debug_logical_indices).any()
        assert evicted_block_count[s].sum() == evicted_blocks_per_seq[s]

    if debug_counts is not None:
        print(torch.where(debug_counts != evicted_block_count))


def count_block_evictions(
    evicted_block_count: torch.Tensor,
    evicted_logical_indices: torch.Tensor,
    evicted_kv_offsets: torch.Tensor,
    hanging_token_count: torch.Tensor,
    block_size: int,
    null_value: int,
    evicted_blocks_per_seq: Optional[torch.Tensor] = None,
):
    # evicted_block_count_kernel = evicted_block_count.clone()
    # evicted_logical_indices_kernel = evicted_logical_indices.contiguous().clone()
    # assert evicted_block_count.dtype == torch.int
    # assert evicted_logical_indices.dtype == torch.int
    # assert evicted_kv_offsets.dtype == torch.int
    kvc_ops.count_block_evictions(
        evicted_block_count,
        evicted_logical_indices.contiguous(),
        evicted_kv_offsets.contiguous(),
        hanging_token_count.contiguous(),
        block_size,
        null_value,
    )
    # yo = evicted_logical_indices.clone()
    # ref_count_block_evictions(
    #     evicted_block_count,
    #     evicted_logical_indices,
    #     evicted_kv_offsets,
    #     hanging_token_count,
    #     block_size,
    #     null_value,
    #     evicted_blocks_per_seq,
    #     # debug_counts=evicted_block_count_kernel,
    #     debug_logical_indices=yo,
    # )
    # assert (evicted_block_count_kernel == evicted_block_count).all()
    # assert (evicted_logical_indices_kernel == evicted_logical_indices).all()
    # print((evicted_block_count == evicted_block_count_kernel.transpose(0, 1).flatten().view(evicted_block_count.shape)).all())
    # print((evicted_block_count == evicted_block_count_kernel.transpose(1, 2).flatten().view(evicted_block_count.shape)).all())
    # print((evicted_block_count == evicted_block_count_kernel.transpose(0, 2).flatten().view(evicted_block_count.shape)).all())
    # print(f"HII: {evicted_block_count - evicted_block_count_kernel=}")
    # print((evicted_block_count == evicted_block_count_kernel).sum(), evicted_block_count.numel())


def ref_schedule_t1_cache_moves(
    out_cache_moves_idx: torch.Tensor,
    out_cache_moves_count: torch.Tensor,
    evicted_logical_indices: torch.Tensor,
    evicted_kv_count: torch.Tensor,
    evicted_kv_offsets: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
):
    num_seqs, num_layers, num_kv_heads = evicted_kv_count.shape
    for i in range(num_seqs):
        for layer_idx in range(num_layers):
            for j in range(num_kv_heads):
                move_count = 0
                evict_count = 0
                # print(f'counts: {evicted_kv_count[i, layer_idx, j]}')
                # print(f'idxs: {evicted_kv_indices[i, layer_idx, j]}')
                # print(f'ctx_len: {context_lens[layer_idx,i,j]}')
                start_head_offset = evicted_kv_offsets[i, layer_idx, j]
                end_head_offset = start_head_offset + evicted_kv_count[i,layer_idx,j] - 1
                for k in range(evicted_kv_count[i, layer_idx, j].item()):
                    src_idx = context_lens[layer_idx,i,j] - k - 1

                    end_src_idx = evicted_logical_indices[end_head_offset - evict_count]
                    dst_idx = evicted_logical_indices[start_head_offset + move_count]

                    if src_idx <= dst_idx:
                        break

                    # print(f'seq: {i}, k: {k}, layer: {layer_idx}, kv_head: {j}, dst: {dst_idx}, src: {src_idx}, end_src: {end_src_idx}')
                    if src_idx <= end_src_idx:
                        evict_count += 1
                        continue

                    src_block_num = block_tables[layer_idx, i, j, src_idx // block_size]
                    dst_block_num = block_tables[layer_idx, i, j, dst_idx // block_size]

                    physical_src_idx = src_block_num * block_size + src_idx % block_size
                    physical_dst_idx = dst_block_num * block_size + dst_idx % block_size
                    # print(f'moving: {physical_src_idx}({src_idx}) -> {physical_dst_idx}({dst_idx})')

                    out_cache_moves_idx[start_head_offset + move_count, 0] = physical_dst_idx
                    out_cache_moves_idx[start_head_offset + move_count, 1] = physical_src_idx
                    move_count += 1

                out_cache_moves_count[i,layer_idx,j] = move_count


@BENCHMARKER.wrap()
def schedule_cache_moves(
    out_cache_moves_indices: torch.Tensor,
    out_cache_moves_count: torch.Tensor,
    evicted_logical_indices: torch.Tensor,
    evicted_kv_count: torch.Tensor,
    evicted_kv_offsets: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
) -> None:
    out_cache_moves_indices.fill_(0)
    kvc_ops.schedule_t1_cache_moves(
    # ref_schedule_t1_cache_moves(
        out_cache_moves_indices,
        out_cache_moves_count,
        evicted_logical_indices.contiguous(),
        evicted_kv_count.contiguous(),
        evicted_kv_offsets.contiguous(),
        block_tables.contiguous(),
        context_lens.contiguous(),
        block_size,
    )


def ref_execute_cache_moves(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_metrics: torch.Tensor,
    kv_positions: torch.Tensor,
    cache_moves_idx: torch.Tensor,
    cache_moves_count: torch.Tensor,
    evicted_kv_offsets: torch.Tensor,
    blocks_per_head: int,
    threads_per_head: int,
):
    # cache_move_dst = cache_moves_idx[:,0].type(torch.long)
    # cache_move_src = cache_moves_idx[:,1].type(torch.long)
    # cat = torch.cat([cache_move_dst[cache_move_dst >= 0], cache_move_src[cache_move_src >= 0]])
    # assert len(cat) == len(cat.unique())
    block_size = v_cache.shape[2]
    num_seqs, num_layers, num_kv_heads = cache_moves_count.shape
    for i in range(num_seqs):
        for layer_idx in range(num_layers):
            for j in range(num_kv_heads):
                moves = []
                cache_moves_offset = evicted_kv_offsets[i,layer_idx,j]
                for k in range(cache_moves_count[i,layer_idx,j].item()):
                    dst_block_num = cache_moves_idx[cache_moves_offset + k, 0] // block_size
                    dst_block_offset = cache_moves_idx[cache_moves_offset + k, 0] % block_size
                    src_block_num = cache_moves_idx[cache_moves_offset + k, 1] // block_size
                    src_block_offset = cache_moves_idx[cache_moves_offset + k, 1] % block_size
                    kv_metrics[dst_block_num, dst_block_offset] = kv_metrics[src_block_num, src_block_offset]
                    assert kv_positions[dst_block_num, dst_block_offset] == kv_positions.view(-1)[cache_moves_idx[cache_moves_offset + k, 0]]
                    kv_positions[dst_block_num, dst_block_offset] = kv_positions[src_block_num, src_block_offset]
                    k_cache[dst_block_num, :, dst_block_offset] = k_cache[src_block_num, :, src_block_offset]
                    v_cache[dst_block_num, :, dst_block_offset] = v_cache[src_block_num, :, src_block_offset]
                    moves.append(f'({src_block_num}, {src_block_offset})->({dst_block_num}, {dst_block_offset})')
                    assert (k_cache[dst_block_num, :, dst_block_offset] == k_cache[src_block_num, :, src_block_offset]).all()
                # print(f'l {layer_idx}, s {i}, h {j}: {", ".join(moves)}')


@BENCHMARKER.wrap()
def execute_cache_moves(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_metrics: torch.Tensor,
    kv_position: torch.Tensor,
    cache_moves_indices: torch.Tensor,
    cache_moves_count: torch.Tensor,
    evicted_kv_offsets: torch.Tensor,
    blocks_per_head: int,
    threads_per_head: int,
) -> None:
    # init_kv_pos = kv_position.clone()
    # kv_pos = kv_position.clone()
    # assert kv_position.dtype == torch.int
    # assert cache_moves_count.dtype == torch.int
    # ref_execute_cache_moves(
    #     k_cache,
    #     v_cache,
    #     kv_metrics,
    #     kv_pos,
    #     cache_moves_indices,
    #     cache_moves_count,
    #     evicted_kv_offsets,
    #     blocks_per_head,
    #     threads_per_head,
    # )
    kvc_ops.execute_cache_moves(
        k_cache,
        v_cache,
        kv_metrics,
        kv_position.contiguous(),
        cache_moves_indices.contiguous(),
        cache_moves_count.contiguous(),
        evicted_kv_offsets.contiguous(),
        blocks_per_head,
        threads_per_head,
    )
    # fail_to_move = torch.where(((kv_position == init_kv_pos) & (kv_position != kv_pos)).flatten())[0]
    # for i in range(10):
    #     fail_to_move_idx = torch.where(cache_moves_indices[:,0] == fail_to_move[i])[0][0]
    #     last_offset = evicted_kv_offsets[evicted_kv_offsets <= fail_to_move_idx].max()
    #     print(f"{fail_to_move_idx=}")
    #     print(f"{last_offset=}")
    #     print(f"{torch.where(evicted_kv_offsets == last_offset)=}")
    # print(f"{len(fail_to_move)}/{kv_position.numel()}={len(fail_to_move)/kv_position.numel()}")
    # incorrect_move = torch.where(((kv_position != init_kv_pos) & (kv_position != kv_pos)).flatten())[0]

    # print(f"{fail_to_move=}")
    # print(f"{incorrect_move=}")

    # print(f"{(kv_pos != kv_position).sum()=}")
    # print(f"{torch.where(kv_pos != kv_position)[0].unique()}")
    # print(f"{torch.where(kv_pos != kv_position)[1].unique()}")
    # assert ((kv_position == kv_pos).flatten()[evicted_kv_offsets[0,0,0]:evicted_kv_offsets[0,0,1]]).all()
    # assert (kv_position == kv_pos).all(), f"{evicted_kv_offsets=}\n{torch.where(kv_pos != kv_position)}"

