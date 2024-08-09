from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any, Dict, Generic, List, Optional, Tuple, Type, TypeVar
from vllm.kvcompress.metrics import CompressionMetrics

import torch


class AttentionBackend(ABC):
    """Abstract class for attention backends."""

    @staticmethod
    @abstractmethod
    def get_impl_cls() -> Type["AttentionImpl"]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def make_metadata(*args, **kwargs) -> "AttentionMetadata":
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: Dict[int, int],
    ) -> None:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: Dict[int, List[int]],
    ) -> None:
        raise NotImplementedError


@dataclass
class AttentionMetadataPerStage:
    """Attention metadata for a specific stage. I.e., prefill or decode."""

    def asdict_zerocopy(self) -> Dict[str, Any]:
        """Similar to dataclasses.asdict, but avoids deepcopying."""
        # Note that if we add dataclasses as fields, they will need
        # similar handling.
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }


T = TypeVar("T", bound=AttentionMetadataPerStage)


@dataclass
class AttentionMetadata(Generic[T]):
    """Attention metadata for prefill and decode batched together."""
    # Total number of prefill requests.
    num_prefills: int
    # Number of prefill tokens.
    num_prefill_tokens: int
    # Number of decode tokens. Note that it is equivalent to the number of
    # decode requests.
    num_decode_tokens: int
    # The attention metadata for prefill requests in a batch.
    # None if there's no prefill requests in a batch.
    prefill_metadata: Optional[T]
    # The attention metadata for decode requests in a batch.
    # None if there's no decode requests in a batch.
    decode_metadata: Optional[T]
    # (num_tokens,). The indices of the token slots that input tokens will be
    # stored into. E.g., if `slot_mapping` is [35, 2, 17] and the block size
    # is 16, the three tokens are stored in the 3rd slot in block 2, 2nd slot
    # in block 0, and 1st slot in block 1, respectively.
    slot_mapping: torch.Tensor
    # The kv cache's data type.
    kv_cache_dtype: str
    # Running metrics for KV cache compression. Must be recorded during
    # paged attention kernel.
    kv_metrics: Optional[CompressionMetrics] = None
    # Minimum distance between a key and query for the query's attention to
    # the key to be aggregated into the key's metric.
    kv_metric_buffer_len: Optional[torch.Tensor] = None
    # Used to determine whether to aggregate metrics for each KV during decoding
    token_positions: Optional[torch.Tensor] = None
    # Last N prefill queries used to initialize KV metrics
    prefill_kv_metric_window_size: int = 32
    # Max number of queries to collect KV metrics for at a time
    prefill_kv_metric_block_size: int = 4096
    # If true, evict based on L2 norm of attention
    kv_metric_use_l2: bool = True
    # If true, evict based on norm of average attention
    kv_metric_use_average: bool = False
    # If true, use maxpool over KV metrics along sequence dimension
    kv_metric_use_maxpool: bool = True

    def __post_init__(self):
        # If layer-specific metadata is required during attention, layer_index
        # should be set before each call to the attention backend.
        self.layer_index = None
        if self.num_prefill_tokens > 0:
            assert self.num_prefills > 0
            assert self.prefill_metadata is not None
        if self.num_decode_tokens > 0:
            assert self.decode_metadata is not None

    def set_layer(self, layer_index: int) -> "AttentionMetadata":
        """Record the active layer and return."""
        self.layer_index = layer_index
        return self


class AttentionImpl(ABC):

    @abstractmethod
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        kv_scale: float,
    ) -> torch.Tensor:
        raise NotImplementedError
