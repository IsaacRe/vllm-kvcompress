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
#   torch::Tensor& evicted_kv_indices,        // [num_seqs, num_layers, num_kv_heads, max_evicted_tokens]
#   torch::Tensor& evicted_kv_count,          // [num_seqs, num_layers, num_kv_heads]
#   torch::Tensor& sorted_indices,            // [total_blocks * BLOCK_SIZE] sorted indices of concat([metrics_0, ..., metrics_N]) where metrics_i[j] is eviction metric for kv j%BLOCK_SIZE of block j/BLOCK_SIZE in sequence i
#   torch::Tensor& seq_block_offsets,         // [num_seqs]
#   torch::Tensor& layer_by_block,            // [total_blocks]
#   torch::Tensor& head_by_block,             // [total_blocks]
#   torch::Tensor& virtual_block_num_by_block,  // [total_blocks]
#   torch::Tensor& evicted_blocks_per_seq)    // [num_seqs]
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
    out_evicted_kv_indices.fill_(MAX_INT)
    out_evicted_kv_count.fill_(0)
    print(f'hanging_toks:\n{hanging_token_count}')
    print(f'sequence_block_offsets:\n{seq_block_offsets}')
    print(f'evicted_blocks_per_seq:\n{evicted_blocks_per_seq}')
    for i in range(num_seqs):
        remaining_kv = torch.ones((num_layers, num_kv_heads)) * hanging_token_count[i]
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
            print(f'kv_idx: {sorted_indices[j]}, v_kv_idx: {kv_idx}, j: {j}, i: {i}, l: {layer_idx}, h: {head_idx}, evicted: {out_evicted_kv_count[i, layer_idx, head_idx] + 1}, remaining: {remaining_kv[layer_idx, head_idx]}, blk#: {block_num}, blk%: {block_offset}')
            assert kv_idx.item() not in {x.item() for x in out_evicted_kv_indices[i, layer_idx, head_idx]}
            if kv_idx >= context_lens[i, layer_idx, head_idx].item():
                j += 1
                continue

            out_evicted_kv_indices[i, layer_idx, head_idx, out_evicted_kv_count[i, layer_idx, head_idx]] = kv_idx
            out_evicted_kv_count[i, layer_idx, head_idx] += 1

            j += 1
            remaining_kv[layer_idx, head_idx] -= 1
            if remaining_kv[layer_idx, head_idx] == 0:
                remaining_kv[layer_idx, head_idx] = block_size
                evicted_blocks += 1
                print(f'evicted {evicted_blocks}/{evicted_blocks_per_seq[i]} blocks')

            tot_iter[layer_idx, head_idx] += 1

        print(f'tot_iter: {tot_iter}')
        print(f'remaining_kv:\n{remaining_kv}')
        assert evicted_blocks == evicted_blocks_per_seq[i], "schedule_cache_evictions loop failed"


# void schedule_t1_cache_moves(
#   torch::Tensor& cache_moves_idx,           // [num_seqs, num_kv_heads, max_evicted_tokens, 2]  (virtual token indices)
#   torch::Tensor& cache_moves_count,         // [num_seqs, num_kv_heads]
#   torch::Tensor& evicted_kv_indices,        // [num_seqs, num_layers, num_kv_heads, max_evicted_tokens]
#   torch::Tensor& evicted_kv_count,          // [num_seqs, num_layers, num_kv_heads]
#   torch::Tensor& block_tables,              // [num_seqs, num_kv_heads, max_num_blocks_per_seq]
#   torch::Tensor& context_lens,              // [num_seqs, num_kv_heads]
#   const int block_size,
#   const int layer_idx)
def ref_schedule_t1_cache_moves(
    out_cache_moves_idx: torch.Tensor,
    out_cache_moves_count: torch.Tensor,
    evicted_kv_indices: torch.Tensor,
    evicted_kv_count: torch.Tensor,
    block_tables: torch.Tensor,
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
            for k in range(evicted_kv_count[i, layer_idx, j].item()):
                src_idx = context_lens[i,j] - k - 1
                end_src_idx = evicted_kv_indices[i, layer_idx, j, evicted_kv_count[i, layer_idx, j] - 1 - move_count - evict_count]
                dst_idx = evicted_kv_indices[i, layer_idx, j, move_count]

                if src_idx <= dst_idx:
                    break
                
                print(f'seq: {i}, k: {k}, layer: {layer_idx}, kv_head: {j}, dst: {dst_idx}, src: {src_idx}, end_src: {end_src_idx}')
                if src_idx <= end_src_idx:
                    evict_count += 1
                    continue
                
                src_block_num = block_tables[i, j, src_idx // block_size]
                dst_block_num = block_tables[i, j, dst_idx // block_size]
                
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
    cache_moves_idx: torch.Tensor,
    cache_moves_count: torch.Tensor,
    block_size: int
):
    num_seqs, num_kv_heads = cache_moves_count.shape
    for i in range(num_seqs):
        for j in range(num_kv_heads):
            for k in range(cache_moves_count[i,j].item()):
                dst_block_num = cache_moves_idx[i,j,k,0] // block_size
                dst_block_offset = cache_moves_idx[i,j,k,0] % block_size
                src_block_num = cache_moves_idx[i,j,k,1] // block_size
                src_block_offset = cache_moves_idx[i,j,k,1] % block_size

                k_cache[dst_block_num, :, dst_block_offset] = k_cache[src_block_num, :, src_block_offset]
                v_cache[dst_block_num, :, dst_block_offset] = v_cache[src_block_num, :, src_block_offset]


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
    # block_size = BLOCK_SIZES[0]
    # head_size = HEAD_SIZES[0]
    # num_heads = NUM_HEADS[1]
    # num_seqs = NUM_GEN_SEQS[0]
    # print(MAX_SEQ_LEN)
    num_seqs = 7
    num_kv_heads = 8
    head_size = 112
    block_size = 16
    kv_cache_dtype = torch.float
    seed = 4
    device = 'cuda:0'

    num_seqs = 3
    num_kv_heads = 4
    block_size = 8
    MAX_SEQ_LEN = 8
    NUM_LAYERS = 2
    head_size = 8

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
    metrics_by_layer = []
    seq_idxs_by_layer = []
    layer_idxs_by_layer = []
    kv_head_idxs_by_layer = []
    virtual_block_nums_by_layer = []
    key_cache_by_layer = []
    value_cache_by_layer = []
    for l in range(NUM_LAYERS):
        context_lens = context_lens_by_layer[l]
        hanging_token_count[l,-1,-1] = block_size if MAX_SEQ_LEN % block_size == 0 else MAX_SEQ_LEN % block_size
        context_lens[-1,-1] = MAX_SEQ_LEN
        max_context_len = MAX_SEQ_LEN
        assert hanging_token_count.max() <= block_size
        assert hanging_token_count.min() > 0

        # Create block_tables/seq_idxs.
        max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
        block_tables = []
        total_blocks = ((context_lens + block_size - 1) // block_size).sum().item()
        kv_head_idxs = torch.empty(total_blocks, dtype=torch.int)
        seq_idxs = torch.empty(total_blocks, dtype=torch.int)
        virtual_block_nums = torch.empty(total_blocks, dtype=torch.int)
        block_idx = 0
        block_nums = np.random.choice(total_blocks, total_blocks, replace=False)
        for i in range(num_seqs):
            # Add to block_tables/head_idxs
            kv_head_block_tables = []
            for j in range(num_kv_heads):
                ctx_len = context_lens[i,j].item()
                num_blocks = (ctx_len + (block_size - 1)) // block_size

                assert num_blocks <= max_num_blocks_per_seq
                kv_head_idxs[block_nums[block_idx:block_idx+num_blocks]] = j
                seq_idxs[block_nums[block_idx:block_idx+num_blocks]] = i
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

        # Add to layer_idxs_by_layer
        layer_idxs_by_layer.append(torch.ones(total_blocks, dtype=torch.int) * l)

        # Create the KV caches.
        key_caches, value_caches = kv_cache_factory(total_blocks, block_size, 1,
                                                    1, head_size,
                                                    kv_cache_dtype, None, seed,
                                                    device)
        key_cache, value_cache = key_caches[0], value_caches[0]
        if key_cache.numel() == 0:  # low-dim debug
            key_cache = value_cache.clone().unsqueeze(-1)
        key_cache = key_cache.squeeze(1)
        value_cache = value_cache.squeeze(1)

        key_cache_by_layer.append(key_cache)
        value_cache_by_layer.append(value_cache)

        # Add to seq_idxs_by_layer
        seq_idxs_by_layer.append(seq_idxs)

        # Add to kv_head_idxs_by_layer
        kv_head_idxs_by_layer.append(kv_head_idxs)

        # Add to virtual_block_nums_by_layer
        virtual_block_nums_by_layer.append(virtual_block_nums)

        # Add to metrics_by_layer, setting empty token slots to inf to prevent eviction
        metrics_by_layer.append(torch.rand(total_blocks * block_size))

    print(f'context_lens: {context_lens_by_layer}')
    print(f'hanging_tokens: {hanging_token_count}')


    num_blocks_by_layer = (
        (context_lens_by_layer + block_size - 1) // block_size
    )

    # Create evicted_blocks_per_seq
    num_blocks_by_seq = num_blocks_by_layer.sum(dim=0).sum(dim=1)  # [num_seqs]
    evicted_blocks_per_seq = torch.tensor(
        [random.randint(0, n) for n in num_blocks_by_seq],
        dtype=torch.int,
    )
    evicted_blocks_per_seq[-1] = num_blocks_by_seq[-1]
    if num_seqs > 1:
        evicted_blocks_per_seq[-2] = 0

    # Compute seq_block_offsets
    seq_block_offsets = torch.concat([
        torch.tensor([0], dtype=torch.int),
        num_blocks_by_layer.sum(dim=0).sum(dim=1).cumsum(dim=0)[:-1],
    ]).type(torch.int)

    # Merge layer metrics, seq and head indices and virtual block numbers
    all_kv_metrics = torch.concat(metrics_by_layer)
    all_seq_idxs = torch.concat(seq_idxs_by_layer)
    all_kv_head_idxs = torch.concat(kv_head_idxs_by_layer)
    all_layer_idxs = torch.concat(layer_idxs_by_layer)
    all_virtual_block_nums = torch.concat(virtual_block_nums_by_layer)

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
    sorted_indices = all_kv_metrics.sort(dim=0).indices

    # Sort indices by seq_idx
    sorted_indices = sorted_indices.gather(
        dim=0,
        index=all_seq_idxs
            .repeat_interleave(block_size)
            .gather(dim=0, index=sorted_indices)
            .sort(dim=0)
            .indices
            .type(torch.int64),
    ).type(torch.int)

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
    max_evicted_tokens = min(
        (evicted_blocks_per_seq[None,:,None] * block_size - block_size + hanging_token_count).max().item(),
        max([context_lens_by_layer[l].max().item() for l in range(NUM_LAYERS)]),
    )
    ref_evicted_kv_indices = torch.ones(
        (num_seqs, NUM_LAYERS, num_kv_heads, max_evicted_tokens),
        dtype=torch.int
    ) * MAX_INT
    ref_evicted_kv_count = torch.empty((num_seqs, NUM_LAYERS, num_kv_heads), dtype=torch.int)
    out_evicted_kv_indices = torch.empty_like(ref_evicted_kv_indices)
    out_evicted_kv_count = torch.empty_like(ref_evicted_kv_count)

    ref_schedule_cache_evictions(
        ref_evicted_kv_indices,
        ref_evicted_kv_count,
        sorted_indices,
        seq_block_offsets,
        all_layer_idxs,
        all_kv_head_idxs,
        all_virtual_block_nums,
        evicted_blocks_per_seq,
        context_lens_by_layer.transpose(0, 1),
        hanging_token_count.transpose(0, 1),
        block_size,
    )
    for i, t in enumerate([
        out_evicted_kv_indices,
        out_evicted_kv_count,
        sorted_indices,
        seq_block_offsets,
        all_layer_idxs,
        all_kv_head_idxs,
        all_virtual_block_nums,
        evicted_blocks_per_seq,
        context_lens_by_layer,
        hanging_token_count,
    ]):
        assert t.dtype == torch.int, f'{i}: {t.dtype}'
    kvc_ops.schedule_cache_evictions(
        out_evicted_kv_indices,
        out_evicted_kv_count,
        sorted_indices,
        seq_block_offsets,
        all_layer_idxs,
        all_kv_head_idxs,
        all_virtual_block_nums,
        evicted_blocks_per_seq,
        context_lens_by_layer.transpose(0, 1).contiguous(),
        hanging_token_count.transpose(0, 1).contiguous(),
        block_size,
    )

    mask = ref_evicted_kv_indices < MAX_INT
    print(out_evicted_kv_indices[mask])
    print(ref_evicted_kv_indices[mask])
    print(out_evicted_kv_indices)
    print(ref_evicted_kv_indices)
    print(out_evicted_kv_count)
    print(ref_evicted_kv_count)
    assert torch.allclose(out_evicted_kv_indices[mask], ref_evicted_kv_indices[mask])
    assert torch.allclose(out_evicted_kv_count, ref_evicted_kv_count)

    # Remove hanging evictions (KV evictions that dont fully free a block) from eviction counts
    no_evicted_kv = ref_evicted_kv_count < hanging_token_count.transpose(0, 1)
    truncated_evicted_kv_count = torch.where(
        no_evicted_kv,
        torch.zeros_like(ref_evicted_kv_count),
        ref_evicted_kv_count - (ref_evicted_kv_count - hanging_token_count.transpose(0, 1)) % block_size,
    )

    print(f'num_blocks_by_layer:\n{num_blocks_by_layer}')
    print(f'sorted_indices [{sorted_indices.shape}]:\n{sorted_indices}')
    print(f'layer_by_block [{all_layer_idxs.shape}]:\n{all_layer_idxs}')
    print(f'head_by_block [{all_kv_head_idxs.shape}]:\n{all_kv_head_idxs}')
    print(f'virtual_blk_num [{all_virtual_block_nums.shape}]:\n{all_virtual_block_nums}')
    print(f'evicted_blk_per_seq [{evicted_blocks_per_seq.shape}]:\n{evicted_blocks_per_seq}')
    print(f'hanging_tok_count [{hanging_token_count.shape}]:\n{hanging_token_count}')
    print('-- schedule_evictions --')
    print(f'evicted_indices [{ref_evicted_kv_indices.shape}]:\n{ref_evicted_kv_indices}')
    print(f'context_lens:\n{context_lens_by_layer.transpose(0,1)}')
    print(f'hanging_tokens:\n{hanging_token_count.transpose(0,1)}')
    print(f'evicted_count [{ref_evicted_kv_count.shape}]:\n{ref_evicted_kv_count}')
    print(f'evict - hanging:\n{ref_evicted_kv_count - hanging_token_count.transpose(0, 1)}')
    print(f'evict - hanging %:\n{(ref_evicted_kv_count - hanging_token_count.transpose(0, 1)) % block_size}')
    print(f'evict - (evict - hanging) %:\n{ref_evicted_kv_count - (ref_evicted_kv_count - hanging_token_count.transpose(0, 1)) % block_size}')
    print(f'evicted_count(trunc):\n{truncated_evicted_kv_count}')

    # Sort evicted_kv_indices per head
    evicted_kv_indices_sorted = ref_evicted_kv_indices.sort(dim=-1).values

    ref_key_cache_by_layer = []
    ref_value_cache_by_layer = []
    ref_cache_moves_by_layer = []
    ref_cache_move_cnt_by_layer = []

    # Remaining eviction steps are applied layerwise
    for l in range(NUM_LAYERS):
        print(f'----- LAYER {l} -----')

        # Call schedule_moves kernel
        ref_cache_moves_idx = -torch.ones((num_seqs, num_kv_heads, max_evicted_tokens, 2), dtype=torch.int)
        ref_cache_moves_count = torch.empty((num_seqs, num_kv_heads), dtype=torch.int)
        out_cache_moves_idx = -torch.ones_like(ref_cache_moves_idx, dtype=torch.int)
        out_cache_moves_count = torch.empty_like(ref_cache_moves_count, dtype=torch.int)

        ref_schedule_t1_cache_moves(
            ref_cache_moves_idx,
            ref_cache_moves_count,
            evicted_kv_indices_sorted,
            truncated_evicted_kv_count,
            block_tables_by_layer[l],
            context_lens_by_layer[l],
            block_size,
            l,
        )
        for i, t in enumerate([
            out_cache_moves_idx,
            out_cache_moves_count,
            evicted_kv_indices_sorted,
            truncated_evicted_kv_count,
            block_tables_by_layer[l],
            context_lens_by_layer[l],
        ]):
            assert t.dtype == torch.int, f'{i}: {t.dtype}'
        kvc_ops.schedule_t1_cache_moves(
            out_cache_moves_idx,
            out_cache_moves_count,
            evicted_kv_indices_sorted.contiguous(),
            truncated_evicted_kv_count.contiguous(),
            block_tables_by_layer[l].contiguous(),
            context_lens_by_layer[l].contiguous(),
            block_size,
            l,
        )

        print(f'block_tables [{block_tables_by_layer[l].shape}]:\n{block_tables_by_layer[l]}')
        print(f'context_lens [{context_lens_by_layer[l].shape}]:\n{context_lens_by_layer[l]}')
        print(f'evicted_indices_sorted [{evicted_kv_indices_sorted.shape}]:\n{evicted_kv_indices_sorted}')
        print('-- schedule_moves --')
        print(f'cache_moves_idx [{ref_cache_moves_idx.shape}]:\n{ref_cache_moves_idx}')
        print(f'cache_moves_count [{ref_cache_moves_count.shape}]:\n{ref_cache_moves_count}')

        print(f'-----\n{out_cache_moves_count}\n---- VS ----\n{ref_cache_moves_count}')
        print(f'-----\n{out_cache_moves_idx}\n---- VS ----\n{ref_cache_moves_idx}')

        mask = ref_cache_moves_idx >= 0
        assert torch.allclose(out_cache_moves_idx[mask], ref_cache_moves_idx[mask])
        assert torch.allclose(out_cache_moves_count, ref_cache_moves_count)

        # Save cache moves for final correctness check
        ref_cache_moves_by_layer.append(ref_cache_moves_idx)
        ref_cache_move_cnt_by_layer.append(ref_cache_moves_count)

        # Call execute_moves kernel
        ref_key_cache = key_cache_by_layer[l].clone()
        ref_value_cache = value_cache_by_layer[l].clone()
        out_key_cache = ref_key_cache.clone()
        out_value_cache = ref_value_cache.clone()

        ref_execute_cache_moves(
            ref_key_cache,
            ref_value_cache,
            ref_cache_moves_idx,
            ref_cache_moves_count,
            block_size,
        )
        for i, t in enumerate([
            ref_cache_moves_idx,
            ref_cache_moves_count,
        ]):
            assert t.dtype == torch.int, f'{i}: {t.dtype}'
        kvc_ops.execute_cache_moves(
            out_key_cache,
            out_value_cache,
            ref_cache_moves_idx,
            ref_cache_moves_count,
            1, 1,
        )

        assert torch.allclose(ref_key_cache, out_key_cache)
        assert torch.allclose(ref_value_cache, out_value_cache)

        print('-- execeute_moves --')
        print(f'in_k_cache [{key_cache_by_layer[l].shape}]:\n{key_cache_by_layer[l]}')
        print(f'out_k_cache [{ref_key_cache.shape}]:\n{ref_key_cache}')
        print(f'in_v_cache [{value_cache_by_layer[l].shape}]:\n{value_cache_by_layer[l]}')
        print(f'out_v_cache [{ref_value_cache.shape}]:\n{ref_value_cache}')

        # set KVs from freed blocks to -1 so they register as evicted below (not necessary in practice)
        for i in range(num_seqs):
            for j in range(num_kv_heads):
                evicted_keys = truncated_evicted_kv_count[i,l,j]
                freed_blocks = (evicted_keys + block_size - 1) // block_size
                total_blocks = (context_lens_by_layer[l,i,j] + block_size - 1) // block_size
                for v_blk in range(total_blocks - freed_blocks, total_blocks):
                    p_blk = block_tables_by_layer[l][i,j,v_blk]
                    ref_key_cache[p_blk] = -1
                    ref_value_cache[p_blk] = -1

        ref_key_cache_by_layer.append(ref_key_cache)
        ref_value_cache_by_layer.append(ref_value_cache)

    # Final check - check that values match for every head of k/v caches for all non-evicted tokens
    
    # setup remaining (unevicted KVs in each block)
    remaining_kvs_per_block = [
        torch.ones_like(blk_tbl) * block_size for blk_tbl in block_tables_by_layer
    ]
    print(f'ctx_lens:\n {context_lens_by_layer}')
    for s in range(num_seqs):
        for l in range(NUM_LAYERS):
            for h in range(num_kv_heads):
                last_v_blk = (context_lens_by_layer[l,s,h].item() + block_size - 1) // block_size - 1
                print(remaining_kvs_per_block[l].shape, hanging_token_count.shape)
                print((s, h, last_v_blk), (l,s,h))
                remaining_kvs_per_block[l][s, h, last_v_blk] = hanging_token_count[l,s,h]

    # set up cache moves map
    cache_moves_map = [
        [
            [
                {
                    ref_cache_moves_by_layer[l][s,h,i,1].item(): ref_cache_moves_by_layer[l][s,h,i,0].item()
                    for i in range(ref_cache_move_cnt_by_layer[l][s,h])
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

    for idx, i in enumerate(sorted_indices):
        block_idx = i // block_size
        block_offset = i % block_size
        s = all_seq_idxs[block_idx]  # array is sorted by seq_idx, metric (ascending)
        l = all_layer_idxs[block_idx]
        h = all_kv_head_idxs[block_idx]
        v_blk = all_virtual_block_nums[block_idx].item()
        v_idx = v_blk * block_size + block_offset

        # skip empty evicted token slots
        #print(context_lens_by_layer.shape)
        if v_idx >= context_lens_by_layer[l,s,h].item():
            continue

        # register eviction if block is within the eviction range
        if v_idx >= (context_lens_by_layer[l,s,h] - truncated_evicted_kv_count[s,l,h]).item():
            print(f'EVICTION (oob): sort_idx {idx}, seq {s}, layer {l}, head {h}, v_idx {v_idx}, blk_idx {block_idx}, blk_offset {block_offset}')
            remaining_kvs_per_block[l][s,h,v_blk] -= 1
            if remaining_kvs_per_block[l][s,h,v_blk] == 0:
                remaining_kvs_per_block[l][s,h,v_blk] = block_size
                total_evicted_blocks_by_seq[s] += 1
                print(f"Freed new block: {total_evicted_blocks_by_seq[s]}")
            continue

        # check if this KV was moved, following reference
        if v_idx in cache_moves_map[l][s][h]:
            print(f'MOVE: sort_idx {idx}, seq {s}, layer {l}, head {h}, v_idx {v_idx}, blk_idx {block_idx}, blk_offset {block_offset}')
            v_idx = cache_moves_map[l][s][h][v_idx]
            v_blk = v_idx // block_size
            block_offset = v_idx % block_size

        if s != current_seq_idx:
            current_seq_idx = s
            done_evicting_for_current_seq.fill_(False)
        
        p_blk = block_tables_by_layer[l][s,h,v_blk]
        #print(key_cache_by_layer[l].shape)
        key = key_cache_by_layer[l][p_blk,:,block_offset,:].flatten()
        value = value_cache_by_layer[l][p_blk,:,block_offset]

        key_kvc_cache = ref_key_cache_by_layer[l][p_blk,:,block_offset,:].flatten()
        value_kvc_cache = ref_value_cache_by_layer[l][p_blk,:,block_offset]

        if (
            not (
                torch.allclose(key_kvc_cache, key) and
                torch.allclose(value_kvc_cache, value)
            )
        ):
            print(f'EVICTION: sort_idx {idx}, seq {s}, layer {l}, head {h}, v_idx {v_idx}, blk_idx {block_idx}, blk_offset {block_offset}')
            remaining_kvs_per_block[l][s,h,v_blk] -= 1
            if remaining_kvs_per_block[l][s,h,v_blk] == 0:
                remaining_kvs_per_block[l][s,h,v_blk] = block_size
                total_evicted_blocks_by_seq[s] += 1
                print(f"Freed new block: {total_evicted_blocks_by_seq[s]}")

            # Once we stop evicting we should not start evicting until sequence changes.
            # This is because indices within each unique seq_idx should be sorted
            # by ascending metric and metric is used for eviction priority (descending)
            assert not done_evicting_for_current_seq[l,h]
        else:
            print(f'equal: sort_idx {idx}, seq {s}, layer {l}, head {h}, v_idx {v_idx}, blk_idx {block_idx}, blk_offset {block_offset}')
            done_evicting_for_current_seq[l,h] = True

    for i in range(num_seqs):
        print(f'seq {i}: evicted {total_evicted_blocks_by_seq[i]}/{evicted_blocks_per_seq[i]}')
        assert total_evicted_blocks_by_seq[i].item() == evicted_blocks_per_seq[i].item()
