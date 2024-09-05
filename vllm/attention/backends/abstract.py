from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, fields
from enum import Enum, auto
from typing import (TYPE_CHECKING, Any, Dict, Generic, List, Optional, Set,
                    Tuple, Type, TypeVar)
from vllm.kvcompress.metrics import CompressionMetrics
from vllm.kvcompress.block import BlockState


import torch

if TYPE_CHECKING:
    from vllm.worker.model_runner_base import (ModelRunnerBase,
                                               ModelRunnerInputBase,
                                               ModelRunnerInputBuilderBase)


class AttentionType(Enum):
    DECODER = auto()  # Decoder attention between previous layer Q/K/V
    ENCODER = auto()  # Encoder attention between previous layer Q/K/V
    ENCODER_DECODER = auto()  # Attention between dec. Q and enc. K/V


class AttentionBackend(ABC):
    """Abstract class for attention backends."""

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_impl_cls() -> Type["AttentionImpl"]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_state_cls() -> Type["AttentionState"]:
        raise NotImplementedError

    @classmethod
    def make_metadata(cls, *args, **kwargs) -> "AttentionMetadata":
        return cls.get_metadata_cls()(*args, **kwargs)

    @staticmethod
    @abstractmethod
    def get_builder_cls() -> Type["AttentionMetadataBuilder"]:
        raise NotImplementedError

    @classmethod
    def make_metadata_builder(cls, *args,
                              **kwargs) -> "AttentionMetadataBuilder":
        return cls.get_builder_cls()(*args, **kwargs)

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
        src_to_dst: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    def advance_step(self, num_seqs: int, num_queries: int):
        raise NotImplementedError


@dataclass
class AttentionMetadata:
    """Attention metadata for prefill and decode batched together."""
    # Total number of prefill requests.
    num_prefills: int
    # Number of prefill tokens.
    num_prefill_tokens: int
    # Number of decode tokens. Note that it is equivalent to the number of
    # decode requests.
    num_decode_tokens: int
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
    # Use modified flash_attn implementation that returns attention values for
    # KV metric initialization without requiring an additional call to
    # _naive_kvc_attention.
    enable_flash_kvc: bool = False

    def __post_init__(self):
        # If layer-specific metadata is required during attention, layer_index
        # should be set before each call to the attention backend.
        self.layer_index = None
        if self.num_prefill_tokens > 0:
            assert self.num_prefills > 0
            assert self.prefill_metadata is not None
        if self.num_decode_tokens > 0:
            assert self.decode_metadata is not None

    @property
    @abstractmethod
    def prefill_metadata(self) -> Optional["AttentionMetadata"]:
        """Return the attention metadata that's required to run prefill
        attention."""
        pass

    @property
    @abstractmethod
    def decode_metadata(self) -> Optional["AttentionMetadata"]:
        """Return the attention metadata that's required to run decode
        attention."""
        pass

    def asdict_zerocopy(self,
                        skip_fields: Optional[Set[str]] = None
                        ) -> Dict[str, Any]:
        """Similar to dataclasses.asdict, but avoids deepcopying."""
        if skip_fields is None:
            skip_fields = set()
        # Note that if we add dataclasses as fields, they will need
        # similar handling.
        return {
            field.name: getattr(self, field.name)
            for field in fields(self) if field.name not in skip_fields
        }

    def set_layer(self, layer_index: int) -> "AttentionMetadata":
        """Record the active layer and return."""
        self.layer_index = layer_index
        return self


T = TypeVar("T", bound=AttentionMetadata)


class AttentionState(ABC, Generic[T]):
    """Holds attention backend-specific objects reused during the
    lifetime of the model runner."""

    @abstractmethod
    def __init__(self, runner: "ModelRunnerBase"):
        ...

    @abstractmethod
    @contextmanager
    def graph_capture(self, max_batch_size: int):
        """Context manager used when capturing CUDA graphs."""
        yield

    @abstractmethod
    def graph_clone(self, batch_size: int) -> "AttentionState[T]":
        """Clone attention state to save in CUDA graph metadata."""
        ...

    @abstractmethod
    def graph_capture_get_metadata_for_batch(self, batch_size: int) -> T:
        """Get attention metadata for CUDA graph capture of batch_size."""
        ...

    @abstractmethod
    def get_graph_input_buffers(self, attn_metadata: T) -> Dict[str, Any]:
        """Get attention-specific input buffers for CUDA graph capture."""
        ...

    @abstractmethod
    def prepare_graph_input_buffers(self, input_buffers: Dict[str, Any],
                                    attn_metadata: T) -> None:
        """In-place modify input buffers dict for CUDA graph replay."""
        ...

    @abstractmethod
    def begin_forward(self, model_input: "ModelRunnerInputBase") -> None:
        """Prepare state for forward pass."""
        ...


class AttentionMetadataBuilder(ABC, Generic[T]):
    """Abstract class for attention metadata builders."""

    @abstractmethod
    def __init__(self, input_builder: "ModelRunnerInputBuilderBase") -> None:
        raise NotImplementedError

    @abstractmethod
    def build(self, seq_lens: List[int], query_lens: List[int],
              cuda_graph_pad_size: int, batch_size: int,
              block_state: Optional[BlockState]) -> T:
        """Build attention metadata with on-device tensors."""
        raise NotImplementedError


class AttentionImpl(ABC, Generic[T]):

    @abstractmethod
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
        kv_cache_dtype: str = "auto",
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: T,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> torch.Tensor:
        raise NotImplementedError
