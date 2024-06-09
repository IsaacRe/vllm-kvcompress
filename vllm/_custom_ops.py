from typing import Dict, Optional, Tuple

import torch

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
    block_size: int,
    max_context_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    kv_metric_buffer_len: int,
    kv_scale: float,
) -> None:
    vllm_ops.kvcompress_paged_attention_v1(out, kv_metric_out, query, key_cache,
                                           value_cache, num_kv_heads, scale,
                                           block_tables, context_lens,
                                           kv_position, last_position,
                                           block_size, max_context_len,
                                           alibi_slopes, kv_cache_dtype,
                                           kv_metric_buffer_len, kv_scale)


def paged_attention_kvc_v2(
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
    kv_position: torch.Tensor,
    last_position: torch.Tensor,
    block_size: int,
    max_context_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    kv_metric_buffer_len: int,
    kv_scale: float,
) -> None:
    vllm_ops.kvcompress_paged_attention_v2(out, exp_sum, max_logits, tmp_out,
                                           query, key_cache, value_cache,
                                           num_kv_heads, scale, block_tables,
                                           context_lens, kv_position,
                                           last_position, block_size,
                                           max_context_len, alibi_slopes,
                                           kv_cache_dtype,
                                           kv_metric_buffer_len, kv_scale)


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

def ref_schedule_cache_evictions(
    out_evicted_kv_indices: torch.Tensor,  # store virtual kv indices
    out_evicted_kv_count: torch.Tensor,
    sorted_indices: torch.Tensor,
    seq_block_offsets: torch.Tensor,
    layer_by_block: torch.Tensor,
    head_by_block: torch.Tensor,
    virtual_block_num_by_block: torch.Tensor,
    evicted_blocks_per_seq: torch.Tensor,
    context_lens: torch.Tensor,
    hanging_token_count: torch.Tensor,
    block_size: int,
    gt_indices: torch.Tensor,
    gt_count: torch.Tensor,
    max_int: int,
):
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

    num_seqs, num_layers, num_kv_heads, _ = out_evicted_kv_indices.shape
    total_blocks, = layer_by_block.shape
    out_evicted_kv_indices.fill_(max_int)
    out_evicted_kv_count.fill_(0)
    # print(f'hanging_toks:\n{hanging_token_count}')
    # print(f'sequence_block_offsets:\n{seq_block_offsets}')
    # print(f'evicted_blocks_per_seq:\n{evicted_blocks_per_seq}')
    for i in range(num_seqs):
        remaining_kv = torch.ones((num_layers, num_kv_heads), device=out_evicted_kv_count.device) * hanging_token_count[i]
        evicted_blocks = 0
        tot_iter = torch.zeros_like(remaining_kv)
        j = seq_block_offsets[i] * block_size
        end_j = (seq_block_offsets[i+1] if i+1 < len(seq_block_offsets) else total_blocks) * block_size
        print(f'start, end: {j}, {end_j}')
        while evicted_blocks < evicted_blocks_per_seq[i] and j < end_j:
            block_num = sorted_indices[j] // block_size
            block_offset = sorted_indices[j] % block_size

            layer_idx = layer_by_block[block_num]
            head_idx = head_by_block[block_num]
            virtual_block_num = virtual_block_num_by_block[block_num]

            kv_idx = virtual_block_num * block_size + block_offset
            # print(f'kv_idx: {sorted_indices[j]}, v_kv_idx: {kv_idx}, j: {j}, i: {i}, l: {layer_idx}, h: {head_idx}, evicted: {out_evicted_kv_count[i, layer_idx, head_idx] + 1}, remaining: {remaining_kv[layer_idx, head_idx]}, blk#: {block_num}, blk%: {block_offset}')
            assert kv_idx.item() not in {x.item() for x in out_evicted_kv_indices[i, layer_idx, head_idx]}
            if kv_idx >= context_lens[i, layer_idx, head_idx].item():
                j += 1
                continue

            out_evicted_kv_indices[i, layer_idx, head_idx, out_evicted_kv_count[i, layer_idx, head_idx]] = kv_idx
            out_evicted_kv_count[i, layer_idx, head_idx] += 1
            # assert kv_idx == gt_indices[i, layer_idx, head_idx, out_evicted_kv_count[i, layer_idx, head_idx]], (
            #     f"{i=}, {layer_idx=}, {head_idx=}, {out_evicted_kv_count[i, layer_idx, head_idx]=}\n"
            #     f"{gt_indices[i, layer_idx, head_idx, out_evicted_kv_count[i, layer_idx, head_idx]]=}\n"
            #     f"{out_evicted_kv_indices[i, layer_idx, head_idx, out_evicted_kv_count[i, layer_idx, head_idx]]=}"
            # )
            # assert out_evicted_kv_count[i, layer_idx, head_idx] == gt_count[i, layer_idx, head_idx], (
            #     f"{i=}, {layer_idx=}, {head_idx=}\n{gt_count[i, layer_idx, head_idx]=}\n{out_evicted_kv_count[i, layer_idx, head_idx]=}"
            # )

            j += 1
            remaining_kv[layer_idx, head_idx] -= 1
            if remaining_kv[layer_idx, head_idx] == 0:
                remaining_kv[layer_idx, head_idx] = block_size
                evicted_blocks += 1
                print(f'evicted {evicted_blocks}/{evicted_blocks_per_seq[i]} blocks')

            tot_iter[layer_idx, head_idx] += 1

        assert out_evicted_kv_count[i].sum() >= evicted_blocks_per_seq[i] * block_size

        print(f'tot_iter: {tot_iter}')
        print(f'remaining_kv:\n{remaining_kv}')
        assert evicted_blocks == evicted_blocks_per_seq[i], "schedule_cache_evictions loop failed"



def schedule_cache_evictions(
    out_evicted_kv_indices: torch.Tensor,
    out_evicted_kv_count: torch.Tensor,
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
    block_size: int,
    protected_window_size: int,
) -> None:
    kvc_ops.schedule_cache_evictions(
        out_evicted_kv_indices,
        out_evicted_kv_count,
        sorted_indices.type(torch.int).contiguous(),
        seq_block_offsets.type(torch.int).contiguous(),
        layer_by_block.contiguous(),
        head_by_block.contiguous(),
        logical_block_num_by_block.contiguous(),
        evicted_blocks_per_seq.contiguous(),
        context_lens.transpose(0, 1).contiguous(),
        hanging_token_count.contiguous(),
        kv_position,
        last_position,
        block_size,
        protected_window_size,
    )
    torch.ones(1).to(0)


def validate_evictions(evicted_kv_indices, evicted_kv_count, max_int):
    valid_mask = evicted_kv_indices < max_int
    selected_mask = (
        torch.arange(evicted_kv_indices.shape[-1],
                     device=evicted_kv_indices.device,
                     dtype=evicted_kv_indices.dtype)[None,None,None]
        < evicted_kv_count[...,None]
    )
    assert not (selected_mask & ~valid_mask).any()


def ref_schedule_t1_cache_moves(
    out_cache_moves_idx: torch.Tensor,
    out_cache_moves_count: torch.Tensor,
    evicted_kv_indices: torch.Tensor,
    evicted_kv_count: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
):
    num_seqs, num_layers, num_kv_heads, _ = evicted_kv_indices.shape
    for i in range(num_seqs):
        for layer_idx in range(num_layers):
            for j in range(num_kv_heads):
                move_count = 0
                evict_count = 0
                # print(f'counts: {evicted_kv_count[i, layer_idx, j]}')
                # print(f'idxs: {evicted_kv_indices[i, layer_idx, j]}')
                # print(f'ctx_len: {context_lens[layer_idx,i,j]}')
                for k in range(evicted_kv_count[i, layer_idx, j].item()):
                    src_idx = context_lens[layer_idx,i,j] - k - 1
                    end_src_idx = evicted_kv_indices[i, layer_idx, j, evicted_kv_count[i, layer_idx, j] - 1 - evict_count]
                    dst_idx = evicted_kv_indices[i, layer_idx, j, move_count]

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

                    out_cache_moves_idx[i, layer_idx, j, move_count, 0] = physical_dst_idx
                    out_cache_moves_idx[i, layer_idx, j, move_count, 1] = physical_src_idx
                    move_count += 1
                
                out_cache_moves_count[i,layer_idx,j] = move_count


def schedule_cache_moves(
    out_cache_moves_indices: torch.Tensor,
    out_cache_moves_count: torch.Tensor,
    evicted_kv_indices: torch.Tensor,
    evicted_kv_count: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
) -> None:
    ref_schedule_t1_cache_moves(  # kvc_ops.schedule_t1_cache_moves(
        out_cache_moves_indices,
        out_cache_moves_count,
        evicted_kv_indices.contiguous(),
        evicted_kv_count.contiguous(),
        block_tables.contiguous(),
        context_lens.contiguous(),
        block_size,
    )
    torch.ones(1).to(0)


def execute_cache_moves(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_metrics: torch.Tensor,
    kv_position: torch.Tensor,
    cache_moves_indices: torch.Tensor,
    cache_moves_count: torch.Tensor,
    blocks_per_head: int,
    threads_per_head: int,
) -> None:
    kvc_ops.execute_cache_moves(
        k_cache,
        v_cache,
        kv_metrics,
        kv_position,
        cache_moves_indices,
        cache_moves_count,
        blocks_per_head,
        threads_per_head,
    )
    torch.ones(1).to(0)
