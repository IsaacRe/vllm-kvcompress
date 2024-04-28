import random
from typing import List, Optional, Tuple

import pytest
import torch
import numpy as np
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask

from vllm._C import ops, cache_ops
from vllm.utils import get_max_shared_memory_bytes
from vllm.utils import is_hip
from allclose_default import get_default_atol, get_default_rtol

FLOAT32_BYTES = torch.finfo(torch.float).bits // 8
# This will change depending on the compute capability.
# - 512 as a buffer
MAX_SEQ_LEN = get_max_shared_memory_bytes() // FLOAT32_BYTES - 512
PARTITION_SIZE = 512
# flshattF and tritonflashattF supported: {torch.float16, torch.bfloat16}
DTYPES = [torch.half, torch.bfloat16, torch.float
          ] if not is_hip() else [torch.half, torch.bfloat16]
NUM_GEN_SEQS = [7]  # Arbitrary values for testing
NUM_PREFILL_SEQS = [3]  # Arbitrary values for testing
NUM_HEADS = [(40, 40), (64, 8)]  # Arbitrary values for testing

# FlashAttention forward only supports head dimension at most 128
# https://github.com/ROCmSoftwarePlatform/flash-attention/blob/3d2b6f5d037782cc2c906909a46fb7e2e1b48b25/csrc/flash_attn_rocm/flash_api.cpp#L62
HEAD_SIZES = [64, 80, 96, 112, 128, 256
              ] if not is_hip() else [64, 80, 96, 112, 128]

BLOCK_SIZES = [16, 32]
USE_ALIBI = [False, True]
KV_CACHE_DTYPE = ["auto", "fp8_e5m2"]
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.float()
    attn_weights = torch.softmax(attn_weights, dim=-1)
    out = torch.einsum("hqk,khd->qhd", attn_weights.to(value.dtype), value)
    return out, attn_weights


def ref_single_query_cached_kv_attention(
    output: torch.Tensor,
    kv_metric_output: torch.Tensor,
    query: torch.Tensor,
    num_queries_per_kv: int,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    t1_block_tables: torch.Tensor,
    t2_block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    seq_context_lens: torch.Tensor,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
    atol,
    rtol,
) -> None:
    num_query_heads = query.shape[1]
    num_kv_heads = num_query_heads // num_queries_per_kv
    head_size = value_cache.shape[1]
    block_size = value_cache.shape[2]
    num_seqs = query.shape[0]

    t1_block_tables = t1_block_tables.cpu()
    t2_block_tables = t2_block_tables.cpu()
    context_lens = context_lens.cpu()
    success_heads = []
    for i in range(num_seqs):
        for h in range(num_kv_heads):
            context_len = int(context_lens[i,h])

            keys = []
            values = []
            t2_blk_numbers = []
            blk_offsets = []
            for j in range(context_len):
                t1_block_number = t1_block_tables[i,j//block_size].item()
                t2_block_number = t2_block_tables[t1_block_number,h].item()
                block_offset = j % block_size

                t2_blk_numbers.append(t2_block_number)
                blk_offsets.append(block_offset)

                k = key_cache[t2_block_number, :, block_offset, :].unsqueeze(0)
                k = k.reshape(1, head_size)
                keys.append(k)

                v = value_cache[t2_block_number, :, block_offset].unsqueeze(0)
                values.append(v)
            keys = torch.stack(keys, dim=0)
            values = torch.stack(values, dim=0)

            t2_blk_numbers = torch.tensor(t2_blk_numbers, device=keys.device, dtype=torch.int64)
            blk_offsets = torch.tensor(blk_offsets, device=keys.device, dtype=torch.int64)

            alibi_bias = None
            if alibi_slopes is not None:
                # Create the ALiBi bias used in the paged attention kernel.
                position_ids = torch.arange(context_len).int()
                alibi_bias = (position_ids - context_len + 1).float()
                alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(
                    1, 1, -1)
            # current kv head, query head for this kv head, we want the
            attn_weights = []
            for qh in range(h * num_queries_per_kv, (h + 1) * num_queries_per_kv):
                # For MQA/GQA handle all query heads that use this kv head
                q = query[i,qh].unsqueeze(0).unsqueeze(0)

                # attention for the current kv head and current query head
                out, attn = ref_masked_attention(q, keys, values, scale, alibi_bias)
                attn_weights.append(attn)
                out = out.view(1, head_size)
                assert torch.allclose(output[i,qh], out[0], atol=atol, rtol=rtol), f"KV head/Q head: {h}/{qh}, Successful: {success_heads}"
                success_heads.append((h, qh))
            
            attn_weights = torch.concat(attn_weights, dim=0)
            print(torch.where(kv_metric_output.flatten() > 0))
            print(torch.where(kv_metric_output.flatten() == 0))
            print("KV_METRIC_OUT:\n" + repr(kv_metric_output[:,0,0]))
            print((kv_metric_output > 0).sum())
            # print(block_table[0])

            # for j in range(min(6, context_len)):
            #     t1_block_number = t1_block_tables[i,j//block_size].item()
            #     t2_block_number = t2_block_tables[t1_block_number,h].item()
            #     block_offset = j % block_size
            #     assert torch.allclose(kv_metric_output[t2_block_number, block_offset], attn_weights[:,0,j]), f"j: {j}, h: {h}, kv_metric_out: {kv_metric_output[t1_block_number, block_offset]}, gt: {attn_weights[:,0,j]}"

            # succeed_metrics = kv_metric_output[(blk_numbers, blk_offsets)].T[torch.isclose(kv_metric_output[(blk_numbers, blk_offsets)].T, attn_weights[:,0,:])]
            # fail_metrics = kv_metric_output[(blk_numbers, blk_offsets)].T[~torch.isclose(kv_metric_output[(blk_numbers, blk_offsets)].T, attn_weights[:,0,:])]
            # print(f"Succeeding kv metrics all equal? {torch.allclose(succeed_metrics, succeed_metrics[0])}")
            # print(f"success metrics: {succeed_metrics[:5]}, diff: {(succeed_metrics[0] - succeed_metrics).abs().max().item()}")
            # print(f"fail metrics: {fail_metrics[:5]}, diff: {(succeed_metrics[0] - fail_metrics).abs().min().item()}")
            # print(f"SUM CLOSE: {torch.isclose(kv_metric_output[(blk_numbers, blk_offsets)].T, attn_weights[:,0,:]).sum()}")
            # print(f"WHERE CLOSE: {torch.where(torch.isclose(kv_metric_output[(blk_numbers, blk_offsets)].T, attn_weights[:,0,:]))}")
            # print(f"WHERE ~CLOSE: {torch.where(~torch.isclose(kv_metric_output[(blk_numbers, blk_offsets)].T, attn_weights[:,0,:]))}")
            assert torch.allclose(kv_metric_output[(t2_blk_numbers, blk_offsets)].T, attn_weights[:,0,:])


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("num_seqs", NUM_GEN_SEQS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
# @pytest.mark.parametrize("use_alibi", USE_ALIBI)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_kvcompress_paged_attention(
    kv_cache_factory,
    version: str,
    num_seqs: int,
    num_heads: Tuple[int, int],
    head_size: int,
    # use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    seed: int,
    device: str,
) -> None:
    # device = 'cuda:0'
    # seed = 1
    # kv_cache_dtype = KV_CACHE_DTYPE[0]
    # dtype = DTYPES[0]
    # block_size = BLOCK_SIZES[0]
    use_alibi = False
    # head_size = HEAD_SIZES[0]
    # num_heads = NUM_HEADS[1]
    # num_seqs = NUM_GEN_SEQS[0]
    # version = 'v2'
    # num_seqs = 2
    # num_heads = 2, 2
    # print(MAX_SEQ_LEN)

    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    query = torch.empty(num_seqs, num_query_heads, head_size, dtype=dtype)
    query.uniform_(-scale, scale)

    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads, dtype=torch.float)

    context_lens = [
        [random.randint(1, MAX_SEQ_LEN) for _ in range(num_kv_heads)]
        for _ in range(num_seqs)
    ]
    context_lens[-1][-1] = MAX_SEQ_LEN
    max_context_len = MAX_SEQ_LEN
    context_lens = torch.tensor(context_lens, dtype=torch.int)
    #context_lens[:] = MAX_SEQ_LEN
    print(context_lens)

    # get max context length for within each sequence
    seq_context_lens = context_lens.max(dim=1).values

    # Create T1/T2 block tables.
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    total_t1_blocks = ((seq_context_lens + block_size - 1) // block_size).sum().item()
    total_t2_blocks = ((context_lens + block_size - 1) // block_size).sum().item()
    t1_block_tables = []
    t2_block_tables = torch.zeros((total_t1_blocks, num_kv_heads), dtype=torch.int)
    print(t2_block_tables)
    print(((context_lens + block_size - 1) // block_size))
    # Track running T1/T2 block indices as we add blocks to the tables
    t1_block_idx = 0
    t2_block_idx = 0
    all_t1_block_nums = np.random.choice(total_t1_blocks, total_t1_blocks, replace=False)
    assert all_t1_block_nums.max().item() < total_t1_blocks
    all_t2_block_nums = np.random.choice(total_t2_blocks, total_t2_blocks, replace=False)
    assert all_t2_block_nums.max().item() < total_t2_blocks
    for i in range(num_seqs):
        # Create T1 block table entry for this sequence
        seq_ctx_len = seq_context_lens[i].item()
        num_seq_blocks = (seq_ctx_len + (block_size - 1)) // block_size
        assert num_seq_blocks <= max_num_blocks_per_seq
        seq_indices = torch.tensor(all_t1_block_nums[t1_block_idx:t1_block_idx+num_seq_blocks])
        t1_block_tables.append(
            torch.concat([
                seq_indices, torch.zeros(max_num_blocks_per_seq - num_seq_blocks)
            ]).type(torch.int)
        )

        # Create and insert T2 block table entries for this sequence
        for j in range(num_kv_heads):
            head_ctx_len = context_lens[i,j].item()
            num_head_blocks = (head_ctx_len + (block_size - 1)) // block_size
            head_seq_indices = torch.tensor(
                all_t2_block_nums[t2_block_idx:t2_block_idx+num_head_blocks], dtype=torch.int
            )
            assert head_seq_indices.max().item() < total_t2_blocks
            t2_block_tables[:,j] = t2_block_tables[:,j].scatter(
                dim=0,
                index=seq_indices[:num_head_blocks],
                src=head_seq_indices
            )
            assert t2_block_tables.max().item() < total_t2_blocks, j

            # Increment T2 block index
            t2_block_idx += num_head_blocks

        # Increment T1 block index
        t1_block_idx += num_seq_blocks

    t1_block_tables = torch.stack(t1_block_tables, dim=0)

    print(t1_block_tables)
    print(t2_block_tables)

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(total_t2_blocks, block_size, 1,
                                                1, head_size,
                                                kv_cache_dtype, dtype, seed,
                                                device)
    key_cache, value_cache = key_caches[0], value_caches[0]
    key_cache = key_cache.squeeze(1)
    value_cache = value_cache.squeeze(1)

    # print(f"block_tables: {block_tables.shape}, context_lens: {context_lens.shape}, key_cache: {key_cache.shape}, value_cache: {value_cache.shape}")

    # Call the paged attention kernel.
    output = torch.empty_like(query)
    kv_metric_output = torch.empty((total_t2_blocks, block_size, num_queries_per_kv), dtype=torch.float32, device=key_cache.device)
    
    print('SHAPES:')
    print(f'output={output.shape}')
    print(f'kv_metric_output={kv_metric_output.shape}')
    print(f'query={query.shape}')
    print(f'key_cache={key_cache.shape}')
    print(f'value_cache={value_cache.shape}')
    print(f't1_block_tables={t1_block_tables.shape}')
    print(f't2_block_tables={t2_block_tables.shape}')
    print(f'context_lens={context_lens.shape}')
    assert t1_block_tables.max().item() < total_t1_blocks
    assert t2_block_tables.max().item() < total_t2_blocks
    if version == "v1":
        ops.kvcompress_t2_paged_attention_v1(
            output,
            kv_metric_output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            t1_block_tables,
            t2_block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes,
            kv_cache_dtype,
        )
    elif version == "v2":
        num_partitions = ((max_context_len + PARTITION_SIZE - 1) //
                          PARTITION_SIZE)
        assert PARTITION_SIZE % block_size == 0
        num_seqs, num_heads, head_size = output.shape
        tmp_output = torch.empty(
            size=(num_seqs, num_heads, num_partitions, head_size),
            dtype=output.dtype,
        )
        tmp_kv_metric_output = torch.empty_like(kv_metric_output)
        exp_sums = torch.empty(
            size=(num_seqs, num_heads, num_partitions),
            dtype=torch.float32,
        )
        max_logits = torch.empty_like(exp_sums)
        ops.kvcompress_t2_paged_attention_v2(
            output,
            kv_metric_output,
            exp_sums,
            max_logits,
            tmp_output,
            tmp_kv_metric_output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            t1_block_tables,
            t2_block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes,
            kv_cache_dtype,
        )
    else:
        raise AssertionError(f"Unknown version: {version}")

    # Run the reference implementation.
    if kv_cache_dtype == "fp8_e5m2":
        # Convert cache data back to dtype.
        x = 16 // torch.tensor([], dtype=dtype).element_size()
        key_cache_shape = (total_t2_blocks, head_size // x, block_size, x)
        dequantized_key_cache = torch.empty(size=key_cache_shape,
                                            dtype=dtype,
                                            device=device)
        cache_ops.convert_fp8_e5m2(key_cache, dequantized_key_cache)
        key_cache = dequantized_key_cache

        value_cache_shape = value_cache.shape
        dequantized_value_cache = torch.empty(size=value_cache_shape,
                                              dtype=dtype,
                                              device=device)
        cache_ops.convert_fp8_e5m2(value_cache, dequantized_value_cache)
        value_cache = dequantized_value_cache

    # NOTE(woosuk): Due to the kernel-level differences in the two
    # implementations, there is a small numerical difference in the two
    # outputs. Thus, we use a relaxed tolerance for the test.
    atol = get_default_atol(output) if is_hip() else 1e-3
    rtol = get_default_rtol(output) if is_hip() else 1e-5

    # NOTE(zhaoyang): FP8 KV Cache will introduce quantization error,
    # so we use a relaxed tolerance for the test.
    if kv_cache_dtype == "fp8_e5m2":
        atol, rtol = 1e-2, 1e-5

    ref_single_query_cached_kv_attention(
        output,
        kv_metric_output,
        query,
        num_queries_per_kv,
        key_cache,
        value_cache,
        t1_block_tables,
        t2_block_tables,
        context_lens,
        seq_context_lens,
        scale,
        alibi_slopes,
        atol,
        rtol,
    )
