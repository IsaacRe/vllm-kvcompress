"""CacheEngine class for managing the KV cache."""
from typing import Dict, List, Optional

import torch

from vllm._custom_ops import execute_cache_moves as _execute_cache_moves
from vllm.attention import get_attn_backend
from vllm.attention.ops.paged_attn import KVCAttention
from vllm.core.kv_cache import KVCache, UnifiedKVCache
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, KVCompressConfig
from vllm.kvcompress.scheduler import CacheMoves
from vllm.kvcompress.metrics import CompressionMetrics
from vllm.logger import init_logger
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, is_pin_memory_available

logger = init_logger(__name__)


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        # When using KV-Compress each cache block holds a single head's KVs
        self.num_heads = (
            1 if cache_config.enable_kvcompress else
            model_config.get_num_kv_heads(parallel_config)
        )

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Get attention backend.
        self.attn_backend = get_attn_backend(model_config.dtype)

        # Initialize the cache. KV-Compress uses a unified KV cache where
        # all KVs for all layers are stored in contiguous memory.
        self.gpu_cache = (
            self._allocate_unified_kv_cache(self.num_gpu_blocks, "cuda")
            if cache_config.enable_kvcompress else
            self._allocate_kv_cache(self.num_gpu_blocks, "cuda")
        )
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> KVCache:
        """Allocates KV cache on the specified device."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_heads, self.head_size)
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []
        for _ in range(self.num_layers):
            kv_cache.append(
                torch.empty(kv_cache_shape,
                            dtype=self.dtype,
                            pin_memory=pin_memory,
                            device=device))
        return KVCache(kv_cache)
    
    def _allocate_unified_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> UnifiedKVCache:
        """Allocates unified KV cache tensor for all layers in contiguous
        memory.
        """
        assert device != "cpu", "CPU inference not supported with KV-Compress"
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_heads, self.head_size)
        print(f"CUDA Mem: {torch.cuda.memory_allocated(0) * 1e-9}")
        kv_cache = torch.empty(kv_cache_shape,
                               dtype=self.dtype,
                               pin_memory=False,
                               device=device)
        return UnifiedKVCache(kv_cache)

    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        for i in range(self.num_layers):
            self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                          src_to_dst)

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        # KV-Compress does not yet support swap-preemption
        assert not self.cache_config.enable_kvcompress
        for i in range(self.num_layers):
            self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                          src_to_dst)

    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)

    def execute_cache_moves(self, cache_moves: CacheMoves, kv_metrics: CompressionMetrics) -> None:
        k_cache, v_cache = KVCAttention.split_kv_cache(self.gpu_cache, self.head_size)
        _execute_cache_moves(
            k_cache=k_cache,
            v_cache=v_cache,
            kv_metrics=kv_metrics.metrics,
            cache_moves_indices=cache_moves.index,
            cache_moves_count=cache_moves.count,
            blocks_per_head=1,
            threads_per_head=1,
        )

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        kvcompress_config: Optional[KVCompressConfig],
    ) -> int:
        head_size = model_config.get_head_size()
        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        dtype_size = _get_dtype_size(dtype)

        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = cache_config.block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)

        if cache_config.enable_kvcompress:
            block_size = kvcompress_config.get_cache_block_size(
                cache_config.block_size, head_size, dtype_size)
            return block_size
        return dtype_size * total


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
