from abc import ABC
from typing import List, Any
import torch


class KVCacheBase(ABC):

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.get_layer(i)
    
    def __len__(self) -> int:
        pass

    def get_layer(self, layer_index: int) -> torch.Tensor:
        """Return the KV cache for a particular layer"""
        pass


class KVCache(KVCacheBase):
    """Standard multi-layer KV cache implementation."""

    def __init__(self, layer_caches: List[torch.Tensor]) -> None:
        self.layer_caches = layer_caches

    def __len__(self) -> int:
        return len(self.layer_caches)
    
    def get_layer(self, layer_index: int) -> torch.Tensor:
        return self.layer_caches[layer_index]
    

class UnifiedKVCache(KVCacheBase):
    """KV cache that represents cache of all layers in unified
    memory. Used when running with KV-Compress.
    """

    def __init__(self, unified_cache: torch.Tensor) -> None:
        self.unified_cache = unified_cache

    def __len__(self) -> int:
        raise ValueError("len not defined for UnifiedKVCache")

    def __getitem__(self, i: Any) -> torch.Tensor:
        return self.unified_cache[i]

    def get_layer(self, layer_index: int) -> torch.Tensor:
        return self.unified_cache
