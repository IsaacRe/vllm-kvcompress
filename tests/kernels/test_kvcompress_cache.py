import random
from typing import List, Optional, Tuple
import time

import pytest
import torch

from vllm._C import ops
from vllm.utils import get_max_shared_memory_bytes
from vllm.utils import is_hip
from allclose_default import get_default_atol, get_default_rtol

FLOAT32_BYTES = torch.finfo(torch.float).bits // 8
# This will change depending on the compute capability.
# - 512 as a buffer
MAX_SEQ_LEN = get_max_shared_memory_bytes() // FLOAT32_BYTES - 512
# There may not be enough gpu memory due to large NUM_BLOCKS.
# Reduce NUM_BLOCKS when it happens.
NUM_BLOCKS = 321  # Arbitrary values for testing
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


@pytest.mark.parametrize("num_seqs", NUM_GEN_SEQS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_kvcompress_schedule_evictions(
    num_seqs: int,
    num_heads: int,
    block_size: int,
    seed: int,
    device: str,
):
    # nonstexpr int num_seqs = 2;
    # constexpr int num_layers = 4;
    # constexpr int num_kv_heads = 4;
    # const int total_kv_heads = num_layers * num_kv_heads;
    # constexpr int max_evicted_blocks = 6;
    # constexpr int blocks_per_head = 8;
    # constexpr int block_size = 8;

    num_seqs = NUM_GEN_SEQS[0]
    NUM_LAYERS = 4
    num_heads = NUM_HEADS[0]
    NUM_EVICTED = 6
    block_size = BLOCK_SIZES[0]
    NUM_BLOCKS = 8

    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    metrics_by_layer = [
        torch.rand(NUM_BLOCKS * num_seqs * num_heads * block_size, device=device)
        for _ in range(NUM_LAYERS)
    ]
    seq_idxs_by_layer = [
        torch.arange(num_seqs, device=device, dtype=torch.int) \
            .repeat_interleave(NUM_BLOCKS * num_heads)
        for _ in range(NUM_LAYERS)
    ]
    head_idxs_by_layer = [
        torch.arange(num_heads, device=device, dtype=torch.int) \
            .repeat_interleave(NUM_BLOCKS).repeat(num_seqs)
        for _ in range(NUM_LAYERS)
    ]

    all_metrics = torch.cat(metrics_by_layer)
    all_seq_idxs = torch.cat(seq_idxs_by_layer).repeat_interleave(block_size)
    head_by_block = torch.cat(head_idxs_by_layer)
    layer_by_block = torch.tensor(
        [[l] * m.size(0) for l, m in enumerate(seq_idxs_by_layer)],
        device=device,
        dtype=torch.int,
    ).flatten()
    sorted_metrics_idx = torch.sort(all_metrics).indices  # sort by metric
    sorted_metrics_idx = sorted_metrics_idx.gather(       # sort by layer
        dim=0,
        index=torch.sort(
            all_seq_idxs.gather(dim=0, index=sorted_metrics_idx)
        ).indices,
    )
    seq_block_offsets = torch.tensor(
        [
            torch.where(all_seq_idxs.gather(dim=0, index=sorted_metrics_idx) == i)[0][0] // block_size
            for i in range(num_seqs)
        ],
        dtype=torch.int,
        device=device,
    )
    sorted_metrics_idx = sorted_metrics_idx.type(torch.int)
    num_evicted = torch.tensor(
        [NUM_EVICTED for _ in range(num_seqs)],
        device=device,
        dtype=torch.int,
    )

    out_idx = torch.empty(
        num_seqs,
        NUM_LAYERS,
        num_heads,
        NUM_EVICTED,
        block_size,
        dtype=torch.int,
        device=device,
    )
    out_cnt = torch.empty(
        num_seqs, NUM_LAYERS, num_heads, NUM_LAYERS * num_heads, dtype=torch.int, device=device
    )

    ops.kvcompress_schedule_evictions(
        out_idx,
        out_cnt,
        sorted_metrics_idx.type(torch.int),
        seq_block_offsets,
        layer_by_block,
        head_by_block,
        num_evicted,
        block_size,
    )

    num_heads = NUM_HEADS[1]

    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    metrics_by_layer = [
        torch.rand(NUM_BLOCKS * num_seqs * num_heads * block_size, device=device)
        for _ in range(NUM_LAYERS)
    ]
    seq_idxs_by_layer = [
        torch.arange(num_seqs, device=device, dtype=torch.int) \
            .repeat_interleave(NUM_BLOCKS * num_heads)
        for _ in range(NUM_LAYERS)
    ]
    head_idxs_by_layer = [
        torch.arange(num_heads, device=device, dtype=torch.int) \
            .repeat_interleave(NUM_BLOCKS).repeat(num_seqs)
        for _ in range(NUM_LAYERS)
    ]

    all_metrics = torch.cat(metrics_by_layer)
    all_seq_idxs = torch.cat(seq_idxs_by_layer).repeat_interleave(block_size)
    head_by_block = torch.cat(head_idxs_by_layer)
    layer_by_block = torch.tensor(
        [[l] * m.size(0) for l, m in enumerate(seq_idxs_by_layer)],
        device=device,
        dtype=torch.int,
    ).flatten()
    sorted_metrics_idx = torch.sort(all_metrics).indices  # sort by metric
    sorted_metrics_idx = sorted_metrics_idx.gather(       # sort by layer
        dim=0,
        index=torch.sort(
            all_seq_idxs.gather(dim=0, index=sorted_metrics_idx)
        ).indices,
    )
    seq_block_offsets = torch.tensor(
        [
            torch.where(all_seq_idxs.gather(dim=0, index=sorted_metrics_idx) == i)[0][0] // block_size
            for i in range(num_seqs)
        ],
        dtype=torch.int,
        device=device,
    )
    sorted_metrics_idx = sorted_metrics_idx.type(torch.int)
    num_evicted = torch.tensor(
        [NUM_EVICTED for _ in range(num_seqs)],
        device=device,
        dtype=torch.int,
    )

    out_idx = torch.empty(
        num_seqs,
        NUM_LAYERS,
        num_heads,
        NUM_EVICTED,
        block_size,
        dtype=torch.int,
        device=device,
    )
    out_cnt = torch.empty(
        num_seqs, NUM_LAYERS, num_heads, NUM_LAYERS * num_heads, dtype=torch.int, device=device
    )

    ops.kvcompress_schedule_evictions(
        out_idx,
        out_cnt,
        sorted_metrics_idx.type(torch.int),
        seq_block_offsets,
        layer_by_block,
        head_by_block,
        num_evicted,
        block_size,
    )

    torch.ones(1).to(0)


def kvcompress_schedule_evictions(
    num_seqs, num_heads, block_size
):
    seed = 1
    device = "cuda"
    kv_metrics_factory = create_kvc_metrics_with_random
    metrics_by_layer, seq_idxs_by_layer, head_idxs_by_layer = kv_metrics_factory(
        batch_size=num_seqs,
        num_blocks=10,
        block_size=block_size,
        num_layers=10,
        num_heads=num_heads,
        seed=seed,
        device=device,
    )

    all_metrics = torch.cat(metrics_by_layer)
    all_seq_idxs = torch.cat(seq_idxs_by_layer).repeat_interleave(block_size)
    head_by_block = torch.cat(head_idxs_by_layer)
    layer_by_block = torch.tensor(
        [[l] * m.size(0) for l, m in enumerate(seq_idxs_by_layer)],
        device=device,
        dtype=torch.int,
    ).flatten()
    sorted_metrics_idx_ = torch.sort(all_metrics).indices  # sort by metric
    sorted_seq_idx = torch.sort(
        all_seq_idxs.gather(dim=0, index=sorted_metrics_idx_)
    ).indices
    sorted_metrics_idx = sorted_metrics_idx_.gather(       # sort by layer
        dim=0,
        index=sorted_seq_idx,
    )

    seq_block_offsets = torch.tensor(
        [
            torch.where(all_seq_idxs.gather(dim=0, index=sorted_metrics_idx) == i)[0][0]
            for i in range(num_seqs)
        ]
    )

    return all_metrics, all_seq_idxs, head_by_block, layer_by_block

    out_idx = torch.empty(
        num_seqs,
        NUM_LAYERS,
        num_heads,
        NUM_BLOCKS,
        block_size,
        dtype=torch.int,
        device=device,
    )
    out_cnt = torch.empty(
        num_seqs, NUM_LAYERS, num_heads, dtype=torch.int, device=device
    )
