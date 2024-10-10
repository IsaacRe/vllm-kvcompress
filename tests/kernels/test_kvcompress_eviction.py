import random
from typing import List, Optional, Tuple
import time

import pytest
import torch
import numpy as np

from vllm._C import kvc_ops
from vllm.utils import get_max_shared_memory_bytes
from vllm.utils import is_hip
from allclose_default import get_default_atol, get_default_rtol

MAX_INT = 2147483000
FLOAT32_BYTES = torch.finfo(torch.float).bits // 8
# This will change depending on the compute capability.
# - 512 as a buffer
MAX_SEQ_LEN = get_max_shared_memory_bytes() // FLOAT32_BYTES - 512
# There may not be enough gpu memory due to large NUM_BLOCKS.
# Reduce NUM_BLOCKS when it happens.
NUM_EVICTED = 50
PARTITION_SIZE = 512
NUM_LAYERS = 8
# flshattF and tritonflashattF supported: {torch.float16, torch.bfloat16}
DTYPES = [torch.half, torch.bfloat16, torch.float
          ] if not is_hip() else [torch.half, torch.bfloat16]
NUM_GEN_SEQS = [7]  # Arbitrary values for testing
NUM_PREFILL_SEQS = [3]  # Arbitrary values for testing
NUM_HEADS = [8, 40]  # Arbitrary values for testing

# FlashAttention forward only supports head dimension at most 128
# https://github.com/ROCmSoftwarePlatform/flash-attention/blob/3d2b6f5d037782cc2c906909a46fb7e2e1b48b25/csrc/flash_attn_rocm/flash_api.cpp#L62
HEAD_SIZES = [64, 80, 96, 112, 128, 256
              ] if not is_hip() else [64, 80, 96, 112, 128]

BLOCK_SIZES = [32, 16]
USE_ALIBI = [False, True]
KV_CACHE_DTYPE = ["auto", "fp8_e5m2"]
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]

    # nonstexpr int num_seqs = 7;
    # constexpr int num_layers = 8;
    # constexpr int num_kv_heads = 8,40;
    # const int total_kv_heads = num_layers * num_kv_heads;
    # constexpr int max_evicted_blocks = 50;
    # constexpr int blocks_per_head = 321;
    # constexpr int block_size = 32;

# void schedule_cache_evictions(
#     torch::Tensor& evicted_kv_indices,        // [num_seqs, num_layers, num_kv_heads, max_evicted_tokens]
#     torch::Tensor& evicted_kv_count,          // [num_seqs, num_layers, num_kv_heads]
#     torch::Tensor& sorted_indices,            // [total_blocks * BLOCK_SIZE] sorted indices of concat([metrics_0, ..., metrics_N]) where metrics_i[j] is eviction metric for kv j%BLOCK_SIZE of block j/BLOCK_SIZE in sequence i
#     torch::Tensor& seq_block_offsets,         // [num_seqs]
#     torch::Tensor& layer_by_block,            // [total_blocks]  TODO: could use uint8
#     torch::Tensor& head_by_block,             // [total_blocks]  TODO: could use uint8
#     torch::Tensor& virtual_block_num_by_block,  // [total_blocks]
#     torch::Tensor& evicted_blocks_per_seq,    // [num_seqs]
#     torch::Tensor& context_lens,              // [num_seqs, num_layers, num_kv_heads]
#     torch::Tensor& hanging_token_count,       // [num_seqs, num_layers, num_kv_heads]
#     torch::Tensor& kv_position,               // [total_blocks, BLOCK_SIZE]
#     torch::Tensor& last_position,             // [num_seqs]
#     const int block_size,
#     const int protected_window_size,
#     const bool evict_evenly_per_layer,
#     const c10::optional<torch::Tensor>& control_layers)
def ref_schedule_cache_evictions(
    out_evicted_kv_indices: torch.Tensor,
    out_evicted_logical_indices: torch.Tensor,
    out_evicted_kv_count: torch.Tensor,
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
    block_size: int,
    protected_window: int,
    max_evicted_kv: int,
    null_eviction_index: int,
    truncate: bool,
    evict_evenly_per_layer: bool,
    control_layers: torch.Tensor,
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
    print(f'hanging_toks:\n{hanging_token_count}')
    print(f'sequence_block_offsets:\n{seq_block_offsets}')
    print(f'evicted_blocks_per_seq:\n{evicted_blocks_per_seq}')
    print(f'{out_evicted_logical_indices.shape=}')
    print(f'evicted_kv_offsets:\n{evicted_kv_offsets}')
    print(f'context_lens:\n{context_lens}')

    evictable_kv = torch.zeros_like(context_lens)
    next_seq_offset = 0
    seq_idx = -1
    for i in range(kv_position.size(0)):
        if i >= next_seq_offset:
            seq_idx += 1
            next_seq_offset = seq_block_offsets[seq_idx + 1] if seq_idx < seq_block_offsets.size(0) - 1 else float('inf')
        evictable_kv[seq_idx, layer_by_block[i], head_by_block[i]] += (
            (kv_position >= 0) & (kv_position <= last_position[0] - protected_window)
        )[i].sum()

    print(f'evictable_keys:\n{evictable_kv}')

    for i in range(num_seqs):
        # blocks_to_evict = min(evicted_blocks_per_seq[i], )
        remaining_kv = torch.ones((num_layers, num_kv_heads)) * hanging_token_count[i]
        evicted_blocks = 0
        tot_iter = torch.zeros_like(remaining_kv)
        current_pos = last_position[i]
        j = seq_block_offsets[i] * block_size
        end_j = (seq_block_offsets[i+1] if i+1 < len(seq_block_offsets) else total_blocks) * block_size
        print(f'start, end: {j}, {end_j}')
        while evicted_blocks < evicted_blocks_per_seq[i] and j < end_j:
            block_num = sorted_indices[j] // block_size
            block_offset = sorted_indices[j] % block_size

            layer_idx = layer_by_block[block_num]
            head_idx = head_by_block[block_num]
            virtual_block_num = virtual_block_num_by_block[block_num]
            kv_pos = kv_position[block_num, block_offset]

            kv_idx = virtual_block_num * block_size + block_offset
            print(f'kv_idx: {sorted_indices[j]}, v_kv_idx: {kv_idx}, j: {j}, i: {i}, l: {layer_idx}, h: {head_idx}, evicted: {out_evicted_kv_count[i, layer_idx, head_idx] + 1}, remaining: {remaining_kv[layer_idx, head_idx]}, blk#: {block_num}, blk%: {block_offset}')
            #assert kv_idx.item() not in {x.item() for x in out_evicted_kv_indices[i, layer_idx, head_idx]}
            if kv_idx >= context_lens[i, layer_idx, head_idx].item():
                j += 1
                continue

            if kv_pos > current_pos - protected_window:
                j += 1
                continue

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
                print(f'evicted {evicted_blocks}/{evicted_blocks_per_seq[i]} blocks')

            tot_iter[layer_idx, head_idx] += 1

        print(f'tot_iter: {tot_iter}')
        print(f'remaining_kv:\n{remaining_kv}')
        assert evicted_blocks == evicted_blocks_per_seq[i], "schedule_cache_evictions loop failed"

    print(f'evicted_count pre-truncate:\n{out_evicted_kv_count}')
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
                    print(f'({i}, {layer_idx}, {head_idx}): {start_offset=}, {end_offset=}, {alloc_kv_count[i, layer_idx, head_idx]=}')
                    out_evicted_logical_indices[start_offset:end_offset] = null_eviction_index
                    out_evicted_kv_indices[start_offset:end_offset] = head_idx + num_kv_heads * (layer_idx + num_layers * i)
    
    assert not (out_evicted_kv_indices == MAX_INT).any()
    print(f'evicted_count:\n{out_evicted_kv_count}')
    for i in range(num_seqs):
        for l in range(num_layers):
            for h in range(num_kv_heads):
                offset = evicted_kv_offsets[i,l,h]
                evicted = out_evicted_kv_count[i,l,h]
                evicted_logical_indices = out_evicted_logical_indices[offset:offset+evicted]
                print(evicted_logical_indices)
                assert not (evicted_logical_indices == MAX_INT).any()

    print(f'INDICES:\n{out_evicted_kv_indices}')
    print(f'LOGICAL_INDICES:\n{out_evicted_logical_indices}')
    print(out_evicted_kv_count)

# void schedule_t1_cache_moves(
#   torch::Tensor& cache_moves_idx,           // [num_seqs, num_layers, num_kv_heads, max_evicted_tokens, 2]  (virtual token indices)
#   torch::Tensor& cache_moves_count,         // [num_seqs, num_layers, num_kv_heads]
#   torch::Tensor& evicted_kv_indices,        // [num_seqs, num_layers, num_kv_heads, max_evicted_tokens]
#   torch::Tensor& evicted_kv_count,          // [num_seqs, num_layers, num_kv_heads]
#   torch::Tensor& block_tables,              // [num_layers, num_seqs, num_kv_heads, max_num_blocks_per_seq]
#   torch::Tensor& context_lens,              // [num_layers, num_seqs, num_kv_heads]
#   const int block_size,
#   const int layer_idx)
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
    print(f'context_lens:\n{context_lens.transpose(0, 1)}')
    print(f'evicted_logical_indices:\n{evicted_logical_indices}')
    alloc_kv_count = (context_lens.transpose(0, 1) + block_size - 1) // block_size
    num_seqs, num_layers, num_kv_heads = evicted_kv_count.shape
    for i in range(num_seqs):
        for layer_idx in range(num_layers):
            for j in range(num_kv_heads):
                move_count = 0
                evict_count = 0
                print(f'counts: {evicted_kv_count[i, layer_idx, j]}')
                #print(f'idxs: {evicted_kv_indices[i, layer_idx, j]}')
                print(f'ctx_len: {context_lens[layer_idx,i,j]}')
                start_head_offset = evicted_kv_offsets[i, layer_idx, j]
                end_head_offset = start_head_offset + evicted_kv_count[i,layer_idx,j] - 1
                for k in range(evicted_kv_count[i, layer_idx, j].item()):
                    src_idx = context_lens[layer_idx,i,j] - k - 1

                    end_src_idx = evicted_logical_indices[end_head_offset - evict_count]
                    dst_idx = evicted_logical_indices[start_head_offset + move_count]

                    print(f'HIIII: {src_idx, dst_idx}')
                    if src_idx <= dst_idx:
                        break
                    
                    print(f'seq: {i}, k: {k}, layer: {layer_idx}, kv_head: {j}, dst: {dst_idx}, src: {src_idx}, end_src: {end_src_idx}')
                    if src_idx <= end_src_idx:
                        evict_count += 1
                        continue
                    
                    src_block_num = block_tables[layer_idx, i, j, src_idx // block_size]
                    dst_block_num = block_tables[layer_idx, i, j, dst_idx // block_size]
                    
                    physical_src_idx = src_block_num * block_size + src_idx % block_size
                    physical_dst_idx = dst_block_num * block_size + dst_idx % block_size
                    print(f'moving: {physical_src_idx}({src_idx}) -> {physical_dst_idx}({dst_idx})')

                    out_cache_moves_idx[start_head_offset + move_count, 0] = physical_dst_idx
                    out_cache_moves_idx[start_head_offset + move_count, 1] = physical_src_idx
                    move_count += 1
                
                out_cache_moves_count[i,layer_idx,j] = move_count
    
    print(evicted_logical_indices)
    print(out_cache_moves_idx)


# void schedule_t1_cache_moves(
#   torch::Tensor& cache_moves_idx,           // [num_seqs, num_kv_heads, max_evicted_tokens, 2]  (virtual token indices)
#   torch::Tensor& cache_moves_count,         // [num_seqs, num_kv_heads]
#   torch::Tensor& evicted_kv_indices,        // [num_seqs, num_layers, num_kv_heads, max_evicted_tokens]
#   torch::Tensor& evicted_kv_count,          // [num_seqs, num_layers, num_kv_heads]
#   torch::Tensor& block_tables,              // [num_seqs, num_kv_heads, max_num_blocks_per_seq]
#   torch::Tensor& context_lens,              // [num_seqs, num_kv_heads]
#   const int block_size,
#   const int layer_idx)
def ref_schedule_t2_cache_moves(
    out_cache_moves_idx: torch.Tensor,
    out_cache_moves_count: torch.Tensor,
    evicted_kv_indices: torch.Tensor,
    evicted_kv_count: torch.Tensor,
    t1_block_tables: torch.Tensor,
    t2_block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    layer_idx: int,
):
    num_seqs, _, num_kv_heads, _ = evicted_kv_indices.shape
    for i in range(num_seqs):
        for j in range(num_kv_heads):
            move_count = 0
            evict_count = 0
            print(f'counts: {evicted_kv_count[i, layer_idx, j]}')
            print(f'idxs: {evicted_kv_indices[i, layer_idx, j]}')
            print(f'ctx_len: {context_lens[i,j]}')
            print(f'move_counts:\n{out_cache_moves_count}')
            for k in range(evicted_kv_count[i, layer_idx, j].item()):
                src_idx = context_lens[i,j] - k - 1
                end_src_idx = evicted_kv_indices[i, layer_idx, j, evicted_kv_count[i, layer_idx, j] - 1 - evict_count]
                dst_idx = evicted_kv_indices[i, layer_idx, j, move_count]

                if src_idx <= dst_idx:
                    break
                
                print(f'seq: {i}, k: {k}, layer: {layer_idx}, kv_head: {j}, dst: {dst_idx}, src: {src_idx}, end_src: {end_src_idx}')
                if src_idx <= end_src_idx:
                    evict_count += 1
                    continue
                
                t2_src_block_num = t1_block_tables[i, src_idx // block_size]
                src_block_num = t2_block_tables[t2_src_block_num, j]
                t2_dst_block_num = t1_block_tables[i, dst_idx // block_size]
                dst_block_num = t2_block_tables[t2_dst_block_num, j]
                
                physical_src_idx = src_block_num * block_size + src_idx % block_size
                physical_dst_idx = dst_block_num * block_size + dst_idx % block_size
                print(f'moving: {physical_src_idx}({src_idx}) -> {physical_dst_idx}({dst_idx})')

                out_cache_moves_idx[i, j, move_count, 0] = physical_dst_idx
                out_cache_moves_idx[i, j, move_count, 1] = physical_src_idx
                move_count += 1
            
            out_cache_moves_count[i,j] = move_count


# void execute_cache_moves(
#   torch::Tensor& k_cache,               // [num_blocks, head_size/x, block_size, x]
#   torch::Tensor& v_cache,               // [num_blocks, head_size, block_size]
#   torch::Tensor& cache_moves_idx,       // [num_seqs, num_kv_heads, max_num_moves, 2] indexes into [num_blocks, block_size]
#   torch::Tensor& cache_moves_count,     // [num_seqs, num_kv_heads]
#   int blocks_per_head,
#   int threads_per_head)
def ref_execute_cache_moves(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_metrics: torch.Tensor,
    kv_positions: torch.Tensor,
    cache_moves_idx: torch.Tensor,
    cache_moves_count: torch.Tensor,
    evicted_kv_offsets: torch.Tensor,
    block_size: int
):
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
                    kv_positions[dst_block_num, dst_block_offset] = kv_positions[src_block_num, src_block_offset]
                    k_cache[dst_block_num, :, dst_block_offset] = k_cache[src_block_num, :, src_block_offset]
                    v_cache[dst_block_num, :, dst_block_offset] = v_cache[src_block_num, :, src_block_offset]
                    moves.append(f'({src_block_num}, {src_block_offset})->({dst_block_num}, {dst_block_offset})')
                    assert (k_cache[dst_block_num, :, dst_block_offset] == k_cache[src_block_num, :, src_block_offset]).all()
                print(f'l {layer_idx}, s {i}, h {j}: {", ".join(moves)}')


# @pytest.mark.parametrize("num_seqs", NUM_GEN_SEQS)
# @pytest.mark.parametrize("num_kv_heads", NUM_HEADS)
# @pytest.mark.parametrize("block_size", BLOCK_SIZES)
# @pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
# @pytest.mark.parametrize("seed", SEEDS)
# @pytest.mark.parametrize("device", CUDA_DEVICES)
def test_kvcompress_schedule_evictions(
    kv_cache_factory,
    # num_seqs: int,
    # num_kv_heads: int,
    # block_size: int,
    # seed: int,
    # device: str,
):
    # device = 'cuda:0'
    # seed = 1
    # kv_cache_dtype = KV_CACHE_DTYPE[0]
    # dtype = DTYPES[0]
    # [block_size = BLOCK_SIZES[0]
    # head_size = HEAD_SIZES[0]
    # num_heads = NUM_HEADS[1]
    # num_seqs = NUM_GEN_SEQS[0]
    # print(MAX_SEQ_LEN)
    num_seqs = 7
    num_kv_heads = 8
    head_size = 112
    block_size = 16
    kv_cache_dtype = "float"
    seed = 4
    device = 'cuda:0'

    protected_window = 0
    num_seqs = 3
    num_kv_heads = 2
    block_size = 4
    MAX_SEQ_LEN = 5
    NUM_LAYERS = 2
    head_size = 2

    # num_seqs = 3
    # num_kv_heads = 2
    # block_size = 2
    # MAX_SEQ_LEN = 8
    # NUM_LAYERS = 2
    # head_size = 1
    # num_seqs = 2
    # num_kv_heads = 1
    # block_size = 2
    # MAX_SEQ_LEN = 2
    # NUM_LAYERS = 2
    # head_size = 1

    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)

    print(f'num_layers: {NUM_LAYERS}, num_seqs: {num_seqs}, num_heads: {num_kv_heads}, block_size: {block_size}')

    # each sequence will have same number of hanging tokens
    max_seq_blocks = (MAX_SEQ_LEN - 1) // block_size  # leave room for hanging tokens
    hanging_token_count = torch.tensor([
        [
            [random.randint(1, block_size) for _ in range(num_kv_heads)]
            for _ in range(num_seqs)
        ]
        for _ in range(NUM_LAYERS)
    ], dtype=torch.int)
    context_lens_by_layer = torch.tensor([
        [
            [random.randint(0, max_seq_blocks) * block_size for _ in range(num_kv_heads)]
            for _ in range(num_seqs)
        ]
        for _ in range(NUM_LAYERS)
    ], dtype=torch.int) + hanging_token_count

    block_tables_by_layer = []
    key_cache_by_layer = []
    value_cache_by_layer = []
    max_num_blocks_per_seq_per_layer = []
    for l in range(NUM_LAYERS):
        context_lens = context_lens_by_layer[l]
        hanging_token_count[l,-1,-1] = block_size if MAX_SEQ_LEN % block_size == 0 else MAX_SEQ_LEN % block_size
        context_lens[-1,-1] = MAX_SEQ_LEN
        max_context_len = MAX_SEQ_LEN
        assert hanging_token_count.max() <= block_size
        assert hanging_token_count.min() > 0

    total_blocks = ((context_lens_by_layer + block_size - 1) // block_size).sum().item()
    block_nums = np.random.choice(total_blocks, total_blocks, replace=False)
    kv_head_idxs = torch.empty(total_blocks, dtype=torch.int)
    seq_idxs = torch.empty(total_blocks, dtype=torch.int)
    layer_idxs = torch.empty(total_blocks, dtype=torch.int)
    virtual_block_nums = -torch.ones(total_blocks, dtype=torch.int)
    last_positions = torch.tensor([MAX_SEQ_LEN - 1] * num_seqs, dtype=torch.int)
    block_idx = 0
    for l in range(NUM_LAYERS):
        context_lens = context_lens_by_layer[l]
        max_context_len = MAX_SEQ_LEN

        # Create block_tables/seq_idxs.
        max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
        max_num_blocks_per_seq_per_layer.append(max_num_blocks_per_seq)
        total_layer_blocks = ((context_lens + block_size - 1) // block_size).sum().item()
        block_tables = []
        for i in range(num_seqs):
            # Add to block_tables/head_idxs
            kv_head_block_tables = []
            for j in range(num_kv_heads):
                ctx_len = context_lens[i,j].item()
                num_blocks = (ctx_len + (block_size - 1)) // block_size

                assert num_blocks <= max_num_blocks_per_seq
                kv_head_idxs[block_nums[block_idx:block_idx+num_blocks]] = j
                seq_idxs[block_nums[block_idx:block_idx+num_blocks]] = i
                layer_idxs[block_nums[block_idx:block_idx+num_blocks]] = l
                virtual_block_nums[block_nums[block_idx:block_idx+num_blocks]] = \
                    torch.arange(num_blocks, dtype=torch.int)
                kv_head_block_tables.append(
                    torch.concat([
                        torch.tensor(block_nums[block_idx:block_idx+num_blocks]),
                        torch.empty(max_num_blocks_per_seq - num_blocks)
                    ]).type(torch.int)
                )
                block_idx += num_blocks
            block_tables.append(torch.stack(kv_head_block_tables))

        # Add to block_tables_by_layer
        block_tables_by_layer.append(torch.stack(block_tables))

        # Create the KV caches.
        key_caches, value_caches = kv_cache_factory(total_layer_blocks, block_size, 1,
                                                    1, head_size,
                                                    kv_cache_dtype, None, seed,
                                                    device)
        key_cache, value_cache = key_caches[0], value_caches[0]
        if key_cache.numel() == 0:  # low-dim debug
            key_cache = value_cache.clone().unsqueeze(-1)
        key_cache = key_cache.squeeze(1)
        value_cache = value_cache.squeeze(1)
        key_cache[:] = torch.arange(key_cache.numel(), dtype=torch.float32).reshape(key_cache.shape)
        value_cache[:] = torch.arange(value_cache.numel(), dtype=torch.float32).reshape(value_cache.shape)

        key_cache_by_layer.append(key_cache)
        value_cache_by_layer.append(value_cache)

    # Set KV positions to be the logical block number * block size + block offset
    kv_positions = -torch.ones((total_blocks, block_size), dtype=torch.int)
    allocated_mask = virtual_block_nums >= 0
    kv_positions[allocated_mask] = (
        (virtual_block_nums * block_size)[:,None] +
        torch.arange(block_size, device=device, dtype=torch.int)[None]
    )[allocated_mask]

    # Add to metrics_by_layer, setting empty token slots to inf to prevent eviction
    all_kv_metrics = torch.rand(total_blocks, block_size)

    print(f'context_lens: {context_lens_by_layer}')
    print(f'hanging_tokens: {hanging_token_count}')

    # Merge KV cache and KV metrics across layers
    key_cache = torch.concat(key_cache_by_layer)
    value_cache = torch.concat(value_cache_by_layer)

    num_blocks_by_layer = (
        (context_lens_by_layer + block_size - 1) // block_size
    )

    # Create evicted_blocks_per_seq
    num_blocks_by_seq = num_blocks_by_layer.sum(dim=0).sum(dim=1)  # [num_seqs]
    protected_window_blocks = ((protected_window + block_size - 1) // block_size) * NUM_LAYERS * num_kv_heads
    evicted_blocks_per_seq = torch.tensor(
        [random.randint(0, n) for n in num_blocks_by_seq - protected_window_blocks],
        dtype=torch.int,
    )
    evicted_blocks_per_seq[-1] = num_blocks_by_seq[-1] - protected_window_blocks
    if num_seqs > 1:
        evicted_blocks_per_seq[-2] = 0

    # Compute seq_block_offsets
    seq_block_offsets = torch.concat([
        torch.tensor([0], dtype=torch.int),
        num_blocks_by_layer.sum(dim=0).sum(dim=1).cumsum(dim=0)[:-1],
    ]).type(torch.int)

    # Merge layer metrics, seq and head indices and virtual block numbers
    all_seq_idxs = seq_idxs
    all_kv_head_idxs = kv_head_idxs
    all_layer_idxs = layer_idxs
    all_virtual_block_nums = virtual_block_nums
    all_kv_positions = kv_positions

    # Stack block tables
    all_block_tables = torch.stack(block_tables_by_layer)

    # DEBUG
    # print(f'all_layers [{all_layer_idxs.shape}]:\n{all_layer_idxs}')
    # print(f'all_seqs [{all_seq_idxs.shape}]:\n{all_seq_idxs}')
    # print(f'all_heads [{all_kv_head_idxs.shape}]:\n{all_kv_head_idxs}')
    # print(f'all_vblks [{all_virtual_block_nums.shape}]:\n{all_virtual_block_nums}')
    # combos = set()
    # print(f'layers: {all_layer_idxs.shape}, seqs: {all_seq_idxs.shape}, heads: {all_kv_head_idxs.shape}, blknums: {all_virtual_block_nums.shape}')
    # for idx in range(len(all_seq_idxs)):
    #     e = (all_layer_idxs[idx].item(), all_seq_idxs[idx].item(), all_kv_head_idxs[idx].item(), all_virtual_block_nums[idx].item())
    #     print(e)
    #     assert e not in combos
    #     combos.add(e)

    
    # prob_0 = 13 // block_size
    # prob_1 = 21 // block_size
    # print(f'DEBUG 13: blk#={13 // block_size}')
    # print(f'DEBUG 21: blk#={21 // block_size}')

    # Sort indices by metric low to high (lowest will be evicted first)
    sorted_indices_ = all_kv_metrics.flatten().sort(dim=0).indices

    # Sort indices by seq_idx
    sorted_indices = sorted_indices_.gather(
        dim=0,
        index=all_seq_idxs
            .repeat_interleave(block_size)
            .gather(dim=0, index=sorted_indices_)
            .sort(dim=0)
            .indices
            .type(torch.int64),
    ).type(torch.long)

    # for some sort result seems to be stable even without passing stable=True
    sort_indices_2 = sorted_indices_.gather(
        dim=0,
        index=all_seq_idxs
            .repeat_interleave(block_size)
            .gather(dim=0, index=sorted_indices_)
            .sort(dim=0, stable=True)
            .indices
            .type(torch.int64),
    ).type(torch.long)

    a = all_kv_metrics.view(-1)[sorted_indices]
    b = all_kv_metrics.view(-1)[sort_indices_2]
    assert (a == b).all()

    sorted_indices = sorted_indices.type(torch.int)

    # # Sort layer and head indices and virtual block numbers by final ordering
    # assert sorted_indices.max().item() < len(all_kv_head_idxs) * block_size
    # assert sorted_indices.max().item() < len(all_layer_idxs) * block_size
    # assert sorted_indices.max().item() < len(all_virtual_block_nums) * block_size
    # head_by_block = all_kv_head_idxs.repeat_interleave(block_size)
    # layer_by_block = all_layer_idxs.repeat_interleave(block_size)
    # virtual_block_num_by_block = all_virtual_block_nums.repeat_interleave(block_size)

    # print(f'head [{head_by_block.shape}]:\n{head_by_block}')
    # print(f'seq [{all_seq_idxs.repeat_interleave(block_size).shape}]:\n{all_seq_idxs.repeat_interleave(block_size)}')
    # print(f'vblk [{virtual_block_num_by_block.shape}]:\n{virtual_block_num_by_block}')
    # print(f'layer [{layer_by_block.shape}]:\n{layer_by_block}')
    # combos = set()
    # all_seqs_repeat = all_seq_idxs.repeat_interleave(block_size)
    # for idx in range(len(head_by_block)):
    #     e = (head_by_block[idx].item(), all_seqs_repeat[idx].item(), virtual_block_num_by_block[idx].item(), layer_by_block[idx].item(), idx % block_size)
    #     print(f'idx: {idx}, {e}')
    #     assert e not in combos
    #     combos.add(e)


    # assert (
    #     head_by_block[16] != head_by_block[18]
    #     or layer_by_block[16] != layer_by_block[18]
    #     or virtual_block_num_by_block[16] != virtual_block_num_by_block[18]
    #     or sorted_indices[16] % block_size != sorted_indices[18] % block_size
    # )

    # Call schedule_evictions kernel
    evicted_kv_offsets = (
        ((context_lens_by_layer.transpose(0, 1) + block_size - 1) // block_size) * block_size
    ).flatten().cumsum(dim=0)
    print(f'evicted offsets flat:\n{evicted_kv_offsets}')
    max_evicted_kv = evicted_kv_offsets[-1].item()
    evicted_kv_offsets = torch.cat([torch.zeros_like(evicted_kv_offsets[:1]), evicted_kv_offsets[:-1]])
    evicted_kv_offsets = evicted_kv_offsets.view(
        *context_lens_by_layer.transpose(0, 1).shape
    ).type(torch.int)

    ref_evicted_kv_indices = torch.ones((max_evicted_kv,), dtype=torch.int) * MAX_INT
    ref_evicted_logical_indices = torch.empty_like(ref_evicted_kv_indices)
    ref_evicted_kv_count = torch.empty((num_seqs, NUM_LAYERS, num_kv_heads), dtype=torch.int)
    out_evicted_kv_indices = torch.empty_like(ref_evicted_kv_indices)
    out_evicted_logical_indices = torch.empty_like(ref_evicted_kv_indices)
    out_evicted_kv_count = torch.empty_like(ref_evicted_kv_count)

    """
    torch::Tensor& evicted_kv_indices,        // [num_seqs, num_layers, num_kv_heads, max_evicted_tokens]
    torch::Tensor& evicted_kv_count,          // [num_seqs, num_layers, num_kv_heads]
    torch::Tensor& sorted_indices,            // [total_blocks * BLOCK_SIZE] sorted indices of concat([metrics_0, ..., metrics_N]) where metrics_i[j] is eviction metric for kv j%BLOCK_SIZE of block j/BLOCK_SIZE in sequence i
    torch::Tensor& seq_block_offsets,         // [num_seqs]
    torch::Tensor& layer_by_block,            // [total_blocks]  TODO: could use uint8
    torch::Tensor& head_by_block,             // [total_blocks]  TODO: could use uint8
    torch::Tensor& virtual_block_num_by_block,  // [total_blocks]
    torch::Tensor& evicted_blocks_per_seq,    // [num_seqs]
    torch::Tensor& context_lens,              // [num_seqs, num_layers, num_kv_heads]
    torch::Tensor& hanging_token_count,       // [num_seqs, num_layers, num_kv_heads]
    torch::Tensor& kv_position,               // [total_blocks, BLOCK_SIZE]
    torch::Tensor& last_position,             // [num_seqs]
    const int block_size,
    const int protected_window_size,
    const bool evict_evenly_per_layer,
    const c10::optional<torch::Tensor>& control_layers)"""

    truncate = True
    ref_schedule_cache_evictions(
        ref_evicted_kv_indices,
        ref_evicted_logical_indices,
        ref_evicted_kv_count,
        evicted_kv_offsets,
        sorted_indices,
        seq_block_offsets,
        all_layer_idxs,
        all_kv_head_idxs,
        all_virtual_block_nums,
        evicted_blocks_per_seq,
        context_lens_by_layer.transpose(0, 1),
        hanging_token_count.transpose(0, 1),
        all_kv_positions,
        last_positions,
        block_size,
        protected_window,
        max_evicted_kv,
        MAX_INT,
        truncate,
        False,
        None,
    )
    print(ref_evicted_kv_count)
    for i, t in enumerate([
        out_evicted_kv_indices,
        out_evicted_logical_indices,
        out_evicted_kv_count,
        evicted_kv_offsets,
        sorted_indices,
        seq_block_offsets,
        all_layer_idxs,
        all_kv_head_idxs,
        all_virtual_block_nums,
        evicted_blocks_per_seq,
        context_lens_by_layer,
        hanging_token_count,
        all_kv_positions,
        last_positions,
    ]):
        assert t.dtype == torch.int, f'{i}: {t.dtype}'
    kvc_ops.schedule_cache_evictions(
        out_evicted_kv_indices,
        out_evicted_logical_indices,
        out_evicted_kv_count,
        hanging_token_count.transpose(0, 1).clone(),
        evicted_kv_offsets,
        sorted_indices,
        seq_block_offsets,
        all_layer_idxs,
        all_kv_head_idxs,
        all_virtual_block_nums,
        evicted_blocks_per_seq,
        context_lens_by_layer.transpose(0, 1).contiguous(),
        hanging_token_count.transpose(0, 1).contiguous(),
        all_kv_positions,
        last_positions,
        block_size,
        protected_window,
        False, #True,
        None, #torch.tensor([], dtype=torch.int),
        max_evicted_kv,
        MAX_INT,
        truncate,
    )

    assert (torch.clamp(ref_evicted_kv_count - hanging_token_count.transpose(0, 1), min=0) % block_size == 0).all()
    print(f'out:\n{ref_evicted_kv_count // block_size}')
    print(f'ref:\n{evicted_blocks_per_seq}')

    mask = ref_evicted_logical_indices < MAX_INT
    # mask = ~(
    #     torch.arange(out_evicted_logical_indices.shape[-1])[None,None,None]
    #             .to(out_evicted_logical_indices.device)
    #     >= out_evicted_logical_indices[...,None]
    # )


    # assert out_evicted_logical_indices[mask].max() <= last_positions.max() - protected_window, f'{out_evicted_logical_indices[mask].max()}, {MAX_INT}'

    print(out_evicted_logical_indices[mask])
    print(ref_evicted_logical_indices[mask])
    print(out_evicted_kv_indices)
    print(ref_evicted_kv_indices)
    print(out_evicted_kv_count)
    print(ref_evicted_kv_count)
    assert torch.allclose(out_evicted_logical_indices[mask], ref_evicted_logical_indices[mask])
    assert torch.allclose(out_evicted_kv_indices, ref_evicted_kv_indices)
    assert torch.allclose(out_evicted_kv_count, ref_evicted_kv_count)

    # # Remove hanging evictions (KV evictions that dont fully free a block) from eviction counts
    # no_evicted_kv = ref_evicted_kv_count < hanging_token_count.transpose(0, 1)
    # truncated_evicted_kv_count = torch.where(
    #     no_evicted_kv,
    #     torch.zeros_like(ref_evicted_kv_count),
    #     ref_evicted_kv_count - (ref_evicted_kv_count - hanging_token_count.transpose(0, 1)) % block_size,
    # )

    # print(context_lens_by_layer.transpose(0, 1))
    # print(truncated_evicted_kv_count)

    # print(f'num_blocks_by_layer:\n{num_blocks_by_layer}')
    # print(f'sorted_indices [{sorted_indices.shape}]:\n{sorted_indices}')
    # print(f'layer_by_block [{all_layer_idxs.shape}]:\n{all_layer_idxs}')
    # print(f'head_by_block [{all_kv_head_idxs.shape}]:\n{all_kv_head_idxs}')
    # print(f'virtual_blk_num [{all_virtual_block_nums.shape}]:\n{all_virtual_block_nums}')
    # print(f'evicted_blk_per_seq [{evicted_blocks_per_seq.shape}]:\n{evicted_blocks_per_seq}')
    # print(f'hanging_tok_count [{hanging_token_count.shape}]:\n{hanging_token_count}')
    # print('-- schedule_evictions --')
    # print(f'evicted_indices [{ref_evicted_kv_indices.shape}]:\n{ref_evicted_kv_indices}')
    # print(f'context_lens:\n{context_lens_by_layer.transpose(0,1)}')
    # print(f'hanging_tokens:\n{hanging_token_count.transpose(0,1)}')
    # print(f'evicted_count [{ref_evicted_kv_count.shape}]:\n{ref_evicted_kv_count}')
    # print(f'evict - hanging:\n{ref_evicted_kv_count - hanging_token_count.transpose(0, 1)}')
    # print(f'evict - hanging %:\n{(ref_evicted_kv_count - hanging_token_count.transpose(0, 1)) % block_size}')
    # print(f'evict - (evict - hanging) %:\n{ref_evicted_kv_count - (ref_evicted_kv_count - hanging_token_count.transpose(0, 1)) % block_size}')
    # print(f'evicted_count(trunc):\n{truncated_evicted_kv_count}')

    # # Sort evicted_kv_indices per head
    # evicted_kv_indices_sorted = ref_evicted_kv_indices.clone()
    # x = torch.arange(evicted_kv_indices_sorted.shape[-1])[None,None,None] >= truncated_evicted_kv_count[...,None]
    # assert x.shape == evicted_kv_indices_sorted.shape
    # evicted_kv_indices_sorted[x] = MAX_INT  # set values 
    # evicted_kv_indices_sorted = evicted_kv_indices_sorted.sort(dim=-1).values
    # print(f'ARANGE:\n{torch.stack([torch.arange(evicted_kv_indices_sorted.shape[-1])[None,None,None].expand(*x.shape), truncated_evicted_kv_count[...,None].expand(*x.shape)], dim=-1)}')
    # print(f'MASK:\n{x}')
    # print(f'SORTED:\n{evicted_kv_indices_sorted}')

    #print(truncated_evicted_kv_count[0,0,1])
    #print(f'HERE: {set(e.item() for e in evicted_kv_indices_sorted[0,0,1][:truncated_evicted_kv_count[0,0,1]])}')

    # Sort evicted indices
    logical_index_sort = ref_evicted_logical_indices.sort(dim=0).indices

    seq_layer_head_logical_index_sort = logical_index_sort[
        ref_evicted_kv_indices[logical_index_sort].sort(dim=0, stable=True).indices
    ]
    evicted_logical_indices_sort = ref_evicted_logical_indices[seq_layer_head_logical_index_sort]

    print(ref_evicted_kv_count)

    assert (ref_evicted_kv_indices == ref_evicted_kv_indices[seq_layer_head_logical_index_sort]).all()

    for i in range(num_seqs):
        for l in range(NUM_LAYERS):
            for h in range(num_kv_heads):
                offset = evicted_kv_offsets[i,l,h]
                evicted = ref_evicted_kv_count[i,l,h]
                evicted_logical_indices = evicted_logical_indices_sort[offset:offset+evicted]
                print(evicted_logical_indices, evicted)
                assert not (evicted_logical_indices == MAX_INT).any()

    # Begin layer-dependent code

    # Call schedule_moves kernel
    ref_cache_moves_idx = -torch.ones((max_evicted_kv, 2), dtype=torch.int)
    ref_cache_moves_count = torch.empty((num_seqs, NUM_LAYERS, num_kv_heads), dtype=torch.int)
    out_cache_moves_idx = -torch.ones_like(ref_cache_moves_idx, dtype=torch.int)
    out_cache_moves_count = torch.zeros_like(ref_cache_moves_count, dtype=torch.int)

    evicted_kv_count = ref_evicted_kv_count.clone()

    ref_schedule_t1_cache_moves(
        ref_cache_moves_idx,
        ref_cache_moves_count,
        evicted_logical_indices_sort,
        evicted_kv_count,
        evicted_kv_offsets,
        all_block_tables,
        context_lens_by_layer,
        block_size,
    )
    for i, t in enumerate([
        out_cache_moves_idx,
        out_cache_moves_count,
        evicted_logical_indices_sort,
        evicted_kv_count,
        evicted_kv_offsets,
        all_block_tables,
        context_lens_by_layer,
    ]):
        assert t.dtype == torch.int, f'{i}: {t.dtype}'
        print(t.shape)
    kvc_ops.schedule_t1_cache_moves(
        out_cache_moves_idx,
        out_cache_moves_count,
        evicted_logical_indices_sort,
        evicted_kv_count,
        evicted_kv_offsets,
        all_block_tables,
        context_lens_by_layer,
        block_size,
    )

    # print(f'block_tables [{block_tables_by_layer[l].shape}]:\n{block_tables_by_layer[l]}')
    # print(f'context_lens [{context_lens_by_layer[l].shape}]:\n{context_lens_by_layer[l]}')
    # print(f'evicted_indices_sorted [{evicted_kv_indices_sorted.shape}]:\n{evicted_kv_indices_sorted}')
    # print('-- schedule_moves --')
    # print(f'cache_moves_idx [{ref_cache_moves_idx.shape}]:\n{ref_cache_moves_idx}')
    # print(f'cache_moves_count [{ref_cache_moves_count.shape}]:\n{ref_cache_moves_count}')

    print(f'-----\n{out_cache_moves_count}\n---- VS ----\n{ref_cache_moves_count}')
    print(f'-----\n{out_cache_moves_idx}\n---- VS ----\n{ref_cache_moves_idx}')
    print(context_lens_by_layer.transpose(0, 1))

    # check equivalence of python and kernel implementations
    mask = ref_cache_moves_idx >= 0
    assert torch.allclose(out_cache_moves_idx[mask], ref_cache_moves_idx[mask])
    assert torch.allclose(out_cache_moves_count, ref_cache_moves_count)

    # check alignment b/t eviction schedule (by virtual index) and move schedule (by physical index)
    
    print(ref_evicted_kv_count)
    for l in range(NUM_LAYERS):
        for s in range(num_seqs):
            for h in range(num_kv_heads):
                offset = evicted_kv_offsets[s,l,h]
                evicted = ref_evicted_kv_count[s,l,h]
                evicted_logical_indices = evicted_logical_indices_sort[offset:offset+evicted]
                print(evicted_logical_indices)
                assert not (evicted_logical_indices == MAX_INT).any()
                for evict_idx in range(ref_evicted_kv_count[s,l,h].item()):
                    v_idx = evicted_logical_indices_sort[offset + evict_idx]
                    print(f'{l, s, h} ({offset=}): {evict_idx=}, {v_idx=}')
                    v_blk = v_idx // block_size
                    p_blk = block_tables_by_layer[l][s,h,v_blk]
                    p_idx = p_blk * block_size + v_idx % block_size
                    for move_idx in range(ref_cache_moves_count[s,l,h].item()):
                        src = ref_cache_moves_idx[offset + move_idx, 1]

                        # evicted keys should never be moved
                        assert src != p_idx

    # Call execute_moves kernel
    ref_key_cache = key_cache.clone()
    ref_value_cache = value_cache.clone()
    ref_kv_metrics = all_kv_metrics.clone()
    ref_kv_positions = all_kv_positions.clone()
    out_key_cache = ref_key_cache.clone()
    out_value_cache = ref_value_cache.clone()
    out_kv_metrics = ref_kv_metrics.clone()

    ref_execute_cache_moves(
        ref_key_cache,
        ref_value_cache,
        ref_kv_metrics,
        ref_kv_positions,
        ref_cache_moves_idx,
        ref_cache_moves_count,
        evicted_kv_offsets,
        block_size,
    )
    for i, t in enumerate([
        all_kv_positions,
        ref_cache_moves_idx,
        ref_cache_moves_count,
        evicted_kv_offsets,
    ]):
        assert t.dtype == torch.int, f'{i}: {t.dtype}'
    kvc_ops.execute_cache_moves(
        out_key_cache,
        out_value_cache,
        out_kv_metrics,
        all_kv_positions,
        ref_cache_moves_idx,
        ref_cache_moves_count,
        evicted_kv_offsets,
        1, 1,
    )
    print(ref_kv_metrics)
    print(out_kv_metrics)
    print(all_kv_metrics)
    print(torch.where(ref_kv_metrics != out_kv_metrics))
    print(torch.where(all_kv_metrics != out_kv_metrics))
    assert (ref_kv_metrics == out_kv_metrics).all()

    print('-- execeute_moves --')

    print(ref_cache_moves_count)
    
    # check for equality with original KV cache for all moved KVs
    for s in range(num_seqs):
        for l in range(NUM_LAYERS):
            for h in range(num_kv_heads):
                moves = []
                offset = evicted_kv_offsets[s,l,h]
                for move_idx in range(ref_cache_moves_count[s,l,h].item()):
                    dst_idx = ref_cache_moves_idx[offset + move_idx, 0]
                    src_idx = ref_cache_moves_idx[offset + move_idx, 1]
                    dst_blk = dst_idx // block_size
                    src_blk = src_idx // block_size
                    dst_offset = dst_idx % block_size
                    src_offset = src_idx % block_size
                    moves.append(f'({src_blk}, {src_offset})->({dst_blk}, {dst_offset})')

                    assert (ref_key_cache[dst_blk,:,dst_offset,:] == key_cache[src_blk,:,src_offset,:]).all().item(), f'{dst_blk.item()=}, {dst_offset.item()=}, {src_blk.item()=}, {src_offset.item()=}'
                    assert (ref_value_cache[dst_blk,:,dst_offset] == value_cache[src_blk,:,src_offset]).all().item()

                print(f'l {l}, s {s}, h {h}: {", ".join(moves)}')

    # print(f'cache_moves idx: \n{ref_cache_moves_idx}')
    # print(f'cache_moves: \n{ref_cache_moves_count}')
    # print(f'in_k_cache [{key_cache_by_layer[l].shape}]:\n{key_cache_by_layer[l]}')
    # print(f'out_k_cache [{ref_key_cache.shape}]:\n{ref_key_cache}\n---- VS ----\n{out_key_cache}')
    # print(f'in_v_cache [{value_cache_by_layer[l].shape}]:\n{value_cache_by_layer[l]}')
    # print(f'out_v_cache [{ref_value_cache.shape}]:\n{ref_value_cache}\n---- VS ----\n{out_value_cache}')

    print(torch.where(ref_key_cache != out_key_cache))
    print(ref_key_cache.shape)  # [blocks, block_size / x, head, x]
    failing_idxs = torch.where((ref_key_cache != out_key_cache).any(dim=-1).any(dim=-1).any(dim=-1))[0]
    print(f'{out_key_cache[failing_idxs]}\n---- VS ----\n{ref_key_cache[failing_idxs]}\nEQ:\n{(ref_key_cache == out_key_cache)[failing_idxs]}')
    print(f'failing_idxs: {failing_idxs}')
    print(f'block tables: {all_block_tables}')

    assert torch.allclose(ref_value_cache, out_value_cache)
    assert torch.allclose(ref_key_cache, out_key_cache)

    # set KVs from freed blocks to -1 so they register as evicted below (not necessary in practice)
    for i in range(num_seqs):
        for l in range(NUM_LAYERS):
            for j in range(num_kv_heads):
                evicted_keys = ref_evicted_kv_count[i,l,j]
                freed_blocks = (evicted_keys + block_size - 1) // block_size
                total_blocks = (context_lens_by_layer[l,i,j] + block_size - 1) // block_size
                for v_blk in range(total_blocks - freed_blocks, total_blocks):
                    p_blk = all_block_tables[l,i,j,v_blk]
                    ref_key_cache[p_blk] = -1
                    ref_value_cache[p_blk] = -1

    # Final check - check that values match for every head of k/v caches for all non-evicted tokens
    
    # setup remaining (unevicted KVs in each block)
    remaining_kvs_per_block = [
        torch.ones((num_seqs, num_kv_heads, max_num_blocks_per_seq), dtype=torch.int) * block_size
        for max_num_blocks_per_seq in max_num_blocks_per_seq_per_layer
    ]
    print(f'ctx_lens:\n {context_lens_by_layer}')
    for s in range(num_seqs):
        for l in range(NUM_LAYERS):
            for h in range(num_kv_heads):
                last_v_blk = (context_lens_by_layer[l,s,h].item() + block_size - 1) // block_size - 1
                # print(remaining_kvs_per_block[l].shape, hanging_token_count.shape)
                # print((s, h, last_v_blk), (l,s,h))
                remaining_kvs_per_block[l][s, h, last_v_blk] = hanging_token_count[l,s,h]

    # set up cache moves map
    cache_moves_map = [
        [
            [
                {
                    ref_cache_moves_idx[evicted_kv_offsets[s,l,h] + i, 1].item():
                    ref_cache_moves_idx[evicted_kv_offsets[s,l,h] + i, 0].item()
                    for i in range(ref_cache_moves_count[s,l,h])
                }
                for h in range(num_kv_heads)
            ]
            for s in range(num_seqs)
        ]
        for l in range(NUM_LAYERS)
    ]
    inv_cache_moves_map = [
        [
            [
                {
                    j: i for i, j in cache_moves_map[l][s][h].items()
                }
                for h in range(num_kv_heads)
            ]
            for s in range(num_seqs)
        ]
        for l in range(NUM_LAYERS)
    ]

    total_evicted_blocks_by_seq = torch.zeros(num_seqs, dtype=torch.int)
    current_seq_idx = 0
    done_evicting_for_current_seq = torch.zeros((NUM_LAYERS, num_kv_heads), dtype=torch.bool)


    # all_kv_metrics = torch.concat(metrics_by_layer)
    # all_seq_idxs = torch.concat(seq_idxs_by_layer)
    # all_kv_head_idxs = torch.concat(kv_head_idxs_by_layer)
    # all_layer_idxs = torch.concat(layer_idxs_by_layer)
    # all_virtual_block_nums = torch.concat(virtual_block_nums_by_layer)
    #print(f'HERE:\n{torch.stack([torch.arange(len(sorted_indices)), all_seq_idxs.repeat_interleave(block_size)[sorted_indices], all_layer_idxs.repeat_interleave(block_size)[sorted_indices], all_kv_head_idxs.repeat_interleave(block_size)[sorted_indices]])}')

    seqs_ = []
    layers_ = []
    heads_ = []
    v_idxs_ = []
    evicted_ = []

    # setup evicted indices by head
    all_evicted_indices = [
        [
            [
                {
                    e.item() for e in
                    evicted_logical_indices_sort[
                        evicted_kv_offsets[s,l,h]:evicted_kv_offsets[s,l,h]+ref_evicted_kv_count[s,l,h]
                    ]
                }
                for h in range(num_kv_heads)
            ]
            for l in range(NUM_LAYERS)
        ]
        for s in range(num_seqs)
    ]
    print(f"FINAL logical indices:\n{evicted_logical_indices_sort}")
    print(all_evicted_indices)

    #print(cache_moves_map[1][0][2])

    # problem idx: layer 0 sequence 0 head 0

    for idx, i in enumerate(sorted_indices):
        block_idx = i // block_size
        block_offset = i % block_size
        s = all_seq_idxs[block_idx]  # array is sorted by seq_idx, metric (ascending)
        l = all_layer_idxs[block_idx]
        h = all_kv_head_idxs[block_idx]
        v_blk = all_virtual_block_nums[block_idx].item()
        v_idx = v_blk * block_size + block_offset

        evict_scheduled = v_idx.item() in all_evicted_indices[s][l][h]

        p_blk = all_block_tables[l,s,h,v_blk]
        p_idx = (p_blk * block_size + block_offset).item()

        seqs_.append(s.item())
        layers_.append(l.item())
        heads_.append(h.item())
        v_idxs_.append(v_idx.item())
        evicted_.append(evict_scheduled)

        try:

            # skip empty evicted token slots
            #print(context_lens_by_layer.shape)
            if v_idx >= context_lens_by_layer[l,s,h].item():
                continue

            # check if this KV was moved, following reference
            print(f'checking move: {p_idx} -> {cache_moves_map[l][s][h]}')
            try:
                print(f'equal? {p_idx == list(cache_moves_map[l][s][h].keys())[0]}')
                print(f'types: {type(p_idx)} vs {type(list(cache_moves_map[l][s][h].keys())[0])}')
            except:
                pass

            moved_p_blk = p_blk
            moved_block_offset = block_offset
            if p_idx in cache_moves_map[l][s][h]:
                print(f'MOVE: sort_idx {idx}, seq {s}, layer {l}, head {h}, v_idx {v_idx}, p_idx {p_idx}, blk_idx {block_idx}, blk_offset {block_offset}, ctx_len {context_lens_by_layer[l,s,h]} adj_ctx_len {(context_lens_by_layer[l,s,h] - ref_evicted_kv_count[s,l,h]).item()}')
                assert not evict_scheduled  # evicted KVs are never moved
                moved_p_idx = cache_moves_map[l][s][h][p_idx]
                moved_p_blk = moved_p_idx // block_size
                moved_block_offset = moved_p_idx % block_size

                # when a move occurs, decrement remaining_kv_per_block of src block
                remaining_kvs_per_block[l][s,h,v_blk] -= 1
                if remaining_kvs_per_block[l][s,h,v_blk] == 0:
                    remaining_kvs_per_block[l][s,h,v_blk] = block_size
                    total_evicted_blocks_by_seq[s] += 1
                    print(f"Freed new block: {total_evicted_blocks_by_seq[s]}")

            # TODO layer 0, seq 0, head 1 is not entering here: adjusted context length is 4 while v_idx is 3
            # register eviction if block is within the eviction range
            elif v_idx >= (context_lens_by_layer[l,s,h] - ref_evicted_kv_count[s,l,h]).item():
                print(f'EVICTION (oob): sort_idx {idx}, seq {s}, layer {l}, head {h}, v_idx {v_idx}, p_idx {p_idx}, blk_idx {block_idx}, blk_offset {block_offset}, ctx_len {context_lens_by_layer[l,s,h]} adj_ctx_len {(context_lens_by_layer[l,s,h] - ref_evicted_kv_count[s,l,h]).item()}')
                assert evict_scheduled
                remaining_kvs_per_block[l][s,h,v_blk] -= 1
                if remaining_kvs_per_block[l][s,h,v_blk] == 0:
                    remaining_kvs_per_block[l][s,h,v_blk] = block_size
                    total_evicted_blocks_by_seq[s] += 1
                    print(f"Freed new block: {total_evicted_blocks_by_seq[s]}")
                continue

            if s != current_seq_idx:
                current_seq_idx = s
                done_evicting_for_current_seq.fill_(False)
            
            #print(key_cache_by_layer[l].shape)
            key = key_cache[p_blk,:,block_offset,:].flatten()
            value = value_cache[p_blk,:,block_offset]

            key_kvc_cache = ref_key_cache[moved_p_blk,:,moved_block_offset,:].flatten()
            value_kvc_cache = ref_value_cache[moved_p_blk,:,moved_block_offset]

            if (
                not (
                    torch.allclose(key_kvc_cache, key) and
                    torch.allclose(value_kvc_cache, value)
                )
            ):
                print(f'EVICTION: sort_idx {idx}, seq {s}, layer {l}, head {h}, v_idx {v_idx}, blk_idx {block_idx}, blk_offset {block_offset}, ctx_len {context_lens_by_layer[l,s,h]} adj_ctx_len {(context_lens_by_layer[l,s,h] - ref_evicted_kv_count[s,l,h]).item()}')
                if not evict_scheduled:
                    print(f'p_blk: {p_blk}, offset: {block_offset}, m_p_blk: {moved_p_blk}, m_offset: {moved_block_offset}')
                    print(f'{key}\n{key_kvc_cache}\n{value}\n{value_kvc_cache}')
                assert evict_scheduled
                # Any unequal KVs outside the eviction range should have been moved into.
                # Make sure this is the case.
                assert p_idx in inv_cache_moves_map[l][s][h]

                # Don't reduce remaining_kvs_per_block since a new KV now populates the slot

                # Once we stop evicting we should not start evicting until sequence changes.
                # This is because indices within each unique seq_idx should be sorted
                # by ascending metric and eviction priority is given by ascending metric
                assert not done_evicting_for_current_seq[l,h], "restarted evicting for head"
            else:
                print(f'equal: sort_idx {idx}, seq {s}, layer {l}, head {h}, v_idx {v_idx}, blk_idx {block_idx}, blk_offset {block_offset}, ctx_len {context_lens_by_layer[l,s,h]} adj_ctx_len {(context_lens_by_layer[l,s,h] - ref_evicted_kv_count[s,l,h]).item()}')
                assert not evict_scheduled
                
                # if the KV is within the protected range it was not a candidate for eviction
                # in the first place, so eviction for this head is not necessarily concluded
                if all_kv_positions.view(-1)[i] <= last_positions[s] - protected_window:
                    done_evicting_for_current_seq[l,h] = True

        except Exception as e:
            print(seqs_)
            print(layers_)
            print(heads_)
            print(v_idxs_)
            print(evicted_)

            raise e

    print(f'Remaining KVs:\n{remaining_kvs_per_block[0][0]}\n{remaining_kvs_per_block[1][0]}')
    print(f'evict idxs:\n{evicted_logical_indices_sort[evicted_kv_offsets[0,0,0]:evicted_kv_offsets[0,0,0]+50]}')
    print(f'evict count:\n{ref_evicted_kv_count[0]}')
    print(f'ctx lens:\n{context_lens_by_layer[:,0]}')
    for i in range(num_seqs):
        print(f'seq {i}: evicted {total_evicted_blocks_by_seq[i]}/{evicted_blocks_per_seq[i]}')
        assert total_evicted_blocks_by_seq[i].item() == evicted_blocks_per_seq[i].item()
