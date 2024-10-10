import random
from typing import Tuple

import pytest
import torch
import numpy as np

from vllm import _custom_ops as ops
from vllm.utils import is_hip

COPYING_DIRECTION = [('cuda', 'cpu'), ('cuda', 'cuda'), ('cpu', 'cuda')]
DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [42]  # Arbitrary values for testing
NUM_LAYERS = [2]  # Arbitrary values for testing
LAYER_INDEX = [1]
NUM_HEADS = [8]  # Arbitrary values for testing
HEAD_SIZES = [64, 80, 96, 112, 128, 256]
BLOCK_SIZES = [8, 16, 32]

# Arbitrary values for testing
# don't make it too large. e.g. [1024, 36000] will OOM
NUM_BLOCKS = [1024, 10000]

NUM_MAPPINGS = [256]  # Arbitrary values for testing
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]
KV_CACHE_DTYPE = ["auto", "fp8"]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_layers", NUM_LAYERS)
@pytest.mark.parametrize("layer_index", LAYER_INDEX)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
@torch.inference_mode()
def test_reshape_and_cache(
    kv_cache_factory,
    num_tokens: int,
    num_layers: int,
    layer_index: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
    kv_cache_dtype: str,
) -> None:
    if not is_hip() and kv_cache_dtype == "fp8":
        pytest.skip()  # This test is not tuned for e5m2 cuda precision
    random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    # Create a random slot mapping.
    num_slots = block_size * num_blocks
    slot_mapping = np.random.choice(num_slots,
                                    (num_layers, num_tokens, num_heads),
                                    replace=False)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.long)
    slot_mapping = slot_mapping[layer_index]

    qkv = torch.randn(num_tokens, 3, num_heads, head_size, dtype=dtype)
    _, key, value = qkv.unbind(dim=1)

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(num_blocks, block_size, 1,
                                                1, head_size,
                                                kv_cache_dtype, dtype, seed,
                                                device)
    key_cache, value_cache = key_caches[0].squeeze(1), value_caches[0].squeeze(1)

    # Create KV metrics
    kv_metrics = torch.randn(num_blocks, block_size,
                             dtype=torch.float32, device=device)
    kv_metric_head_bias = torch.randn(num_layers, num_heads,
                                      dtype=torch.float32, device=device)
    kv_metric_head_bias = kv_metric_head_bias[layer_index]  # head bias for single layer

    # Clone the KV caches.
    if kv_cache_dtype == "fp8":
        cloned_key_cache = torch.empty_like(key_cache, dtype=torch.float16)
        ops.convert_fp8(key_cache, cloned_key_cache)
        cloned_value_cache = torch.empty_like(value_cache, dtype=torch.float16)
        ops.convert_fp8(value_cache, cloned_value_cache)
    else:
        cloned_key_cache = key_cache.clone()
        cloned_value_cache = value_cache.clone()
    cloned_kv_metrics = kv_metrics.clone()

    # Using default kv_scale
    kv_scale = 1.0

    # Call the reshape_and_cache kernel.
    ops.reshape_and_cache_kvc(key, value, key_cache, value_cache, kv_metrics,
                              slot_mapping, kv_metric_head_bias, kv_cache_dtype, kv_scale)

    if kv_cache_dtype == "fp8":
        result_key_cache = torch.empty_like(key_cache, dtype=torch.float16)
        ops.convert_fp8(key_cache, result_key_cache)
        result_value_cache = torch.empty_like(value_cache, dtype=torch.float16)
        ops.convert_fp8(value_cache, result_value_cache)

    # Run the reference implementation.
    reshaped_key = key.reshape(num_tokens, num_heads, *key_cache[0, :, 0, :].shape)
    block_indicies = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_indicies = block_indicies.cpu()
    block_offsets = slot_mapping % block_size
    block_offsets = block_offsets.cpu()
    for i in range(num_tokens):
        for h in range(num_heads):
            block_idx = block_indicies[i,h]
            block_offset = block_offsets[i,h]
            cloned_key_cache[block_idx, :, block_offset, :] = reshaped_key[i,h]
            cloned_value_cache[block_idx, :, block_offset] = value[i,h]
            cloned_kv_metrics[block_idx, block_offset] = kv_metric_head_bias[h]

    if kv_cache_dtype == "fp8":
        assert torch.allclose(result_key_cache,
                              cloned_key_cache,
                              atol=0.001,
                              rtol=0.1)
        assert torch.allclose(result_value_cache,
                              cloned_value_cache,
                              atol=0.001,
                              rtol=0.1)
    else:
        assert torch.allclose(key_cache, cloned_key_cache)
        assert torch.allclose(value_cache, cloned_value_cache)

    assert torch.allclose(kv_metrics, cloned_kv_metrics)
