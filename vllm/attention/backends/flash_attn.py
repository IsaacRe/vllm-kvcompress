"""Attention layer with FlashAttention."""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata,
                                              AttentionMetadataBuilder,
                                              AttentionType)
from vllm.attention.backends.utils import (PAD_SLOT_ID, CommonAttentionState,
                                           compute_slot_mapping,
                                           compute_slot_mapping_start_idx,
                                           is_block_tables_empty)
from vllm.utils import async_tensor_h2d, make_tensor_with_pad
from vllm.attention.ops.paged_attn import (PagedAttention, KVCAttention,
                                           PagedAttentionMetadata)
from vllm.kvcompress.block import BlockState
from vllm.debug import CHECKPOINTER
from vllm.benchmark import BENCHMARKER
from vllm.kvcompress.metrics import CompressionMetrics
from vllm.kvcompress.state import KVCompressState

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUBuilder

from vllm_flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
from vllm_flash_attn import flash_attn_with_kvcache as _flash_attn_with_kvcache
try:
    from flash_attn_kvc import (
        flash_attn_varlen_func as flash_attn_kvc_varlen_func,
    )
    FLASH_KVC_ENABLED = True
except Exception:
    FLASH_KVC_ENABLED = False


@torch.library.custom_op("vllm::flash_attn_varlen_func", mutates_args=[])
def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Optional[List[int]] = None,
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # custom op does not support tuple input
    real_window_size: Tuple[int, int]
    if window_size is None:
        real_window_size = (-1, -1)
    else:
        assert len(window_size) == 2
        real_window_size = (window_size[0], window_size[1])
    return _flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=real_window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        block_table=block_table,
    )


@flash_attn_varlen_func.register_fake  # type: ignore
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Optional[List[int]] = None,
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.custom_op("vllm::flash_attn_with_kvcache", mutates_args=[])
def flash_attn_with_kvcache(
    decode_query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_seqlens: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    alibi_slopes: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
) -> torch.Tensor:
    return _flash_attn_with_kvcache(
        decode_query,
        key_cache,
        value_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        softmax_scale=softmax_scale,
        causal=causal,
        alibi_slopes=alibi_slopes,
        softcap=softcap,
    )


@flash_attn_with_kvcache.register_fake  # type: ignore
def _(
    decode_query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_seqlens: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    alibi_slopes: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
) -> torch.Tensor:
    return torch.empty_like(decode_query)


class FlashAttentionBackend(AttentionBackend):

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "flash-attn"

    @staticmethod
    def get_impl_cls() -> Type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return FlashAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type["FlashAttentionMetadataBuilder"]:
        return FlashAttentionMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache = src_kv_cache[0]
        dst_key_cache = dst_kv_cache[0]
        ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)

        src_value_cache = src_kv_cache[1]
        dst_value_cache = dst_kv_cache[1]
        ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        key_caches = [kv_cache[0] for kv_cache in kv_caches]
        value_caches = [kv_cache[1] for kv_cache in kv_caches]
        ops.copy_blocks(key_caches, value_caches, src_to_dists)


@dataclass
class FlashAttentionMetadata(AttentionMetadata):
    """Metadata for FlashAttentionBackend.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """
    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]]
    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor]

    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    # Maximum query length in the batch. None for decoding.
    max_query_len: Optional[int]
    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int
    # Maximum sequence length among decode batch. 0 if there are prefill
    # requests only.
    max_decode_seq_len: int
    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    query_start_loc: Optional[torch.Tensor]
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor]
    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor]

    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    # E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    # in the kv cache. Each block can contain up to block_size tokens.
    # 2nd dimensions are padded up to max_blocks_per_seq if it is cuda-graph
    # captured.
    block_tables: Optional[torch.Tensor]

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool

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

    _cached_prefill_metadata: Optional["FlashAttentionMetadata"] = None
    _cached_decode_metadata: Optional["FlashAttentionMetadata"] = None

    @property
    def prefill_metadata(self) -> Optional["FlashAttentionMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            return self._cached_prefill_metadata

        assert self.seq_lens is not None
        assert self.seq_lens_tensor is not None
        assert self.query_start_loc is not None
        assert self.context_lens_tensor is not None
        assert self.block_tables is not None
        assert self.seq_start_loc is not None

        self._cached_prefill_metadata = FlashAttentionMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=self.slot_mapping[:self.num_prefill_tokens],
            kv_cache_dtype=self.kv_cache_dtype,
            seq_lens=self.seq_lens[:self.num_prefills],
            seq_lens_tensor=self.seq_lens_tensor[:self.num_prefills],
            max_query_len=self.max_query_len,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_seq_len=0,
            query_start_loc=self.query_start_loc[:self.num_prefills + 1],
            seq_start_loc=self.seq_start_loc[:self.num_prefills + 1],
            context_lens_tensor=self.context_lens_tensor[:self.num_prefills],
            block_tables=self.block_tables[:self.num_prefills],
            use_cuda_graph=False,
            kv_metrics=self.kv_metrics,
            kv_metric_buffer_len=None if self.kv_metric_buffer_len is None else
                                 self.kv_metric_buffer_len[:self.num_prefills],
            token_positions=None if self.token_positions is None else
                            self.token_positions[:self.num_prefill_tokens],
            prefill_kv_metric_window_size=self.prefill_kv_metric_window_size,
            prefill_kv_metric_block_size=self.prefill_kv_metric_block_size,
            kv_metric_use_l2=self.kv_metric_use_l2,
            kv_metric_use_average=self.kv_metric_use_average,
            kv_metric_use_maxpool=self.kv_metric_use_maxpool,
        )
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["FlashAttentionMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            return self._cached_decode_metadata
        assert self.block_tables is not None
        assert self.seq_lens_tensor is not None

        # If using KV-Compress
        if self.kv_metrics:
            context_lens_tensor = self.context_lens_tensor[:,self.num_prefills:]
            block_tables = self.block_tables[:,self.num_prefills:]
        else:
            context_lens_tensor = self.context_lens_tensor[self.num_prefills:]
            block_tables = self.block_tables[self.num_prefills:]

        self._cached_decode_metadata = FlashAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=self.slot_mapping[self.num_prefill_tokens:],
            kv_cache_dtype=self.kv_cache_dtype,
            seq_lens=None,
            seq_lens_tensor=self.seq_lens_tensor[self.num_prefills:],
            max_query_len=None,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            query_start_loc=None,
            seq_start_loc=None,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=self.use_cuda_graph,
            kv_metrics=self.kv_metrics,
            kv_metric_buffer_len=None if self.kv_metric_buffer_len is None else
                                 self.kv_metric_buffer_len[self.num_prefills:],
            token_positions=None if self.token_positions is None else
                            self.token_positions[self.num_prefill_tokens:],
            prefill_kv_metric_window_size=self.prefill_kv_metric_window_size,
            prefill_kv_metric_block_size=self.prefill_kv_metric_block_size,
            kv_metric_use_l2=self.kv_metric_use_l2,
            kv_metric_use_average=self.kv_metric_use_average,
            kv_metric_use_maxpool=self.kv_metric_use_maxpool,
        )
        return self._cached_decode_metadata

    def advance_step(self, num_seqs: int, num_queries: int):
        """
        Update metadata in-place to advance one decode step.
        """
        # GPU in-place update is currently called separately through
        # custom_ops.advance_step(). See draft_model_runner. TODO(will): Move
        # this logic to the backend.

        # When using cudagraph, the num_seqs is padded to the next captured
        # batch sized, but num_queries tracks the actual number of requests in
        # the batch. For --enforce-eager mode, num_seqs == num_queries
        if num_seqs != num_queries:
            assert num_seqs > num_queries
            assert self.use_cuda_graph

        assert self.num_prefills == 0
        assert self.num_prefill_tokens == 0
        assert self.num_decode_tokens == num_seqs
        assert self.slot_mapping.shape == (num_seqs, )

        assert self.seq_lens is not None
        assert len(self.seq_lens) == num_seqs
        assert self.seq_lens_tensor is not None
        assert self.seq_lens_tensor.shape == (num_seqs, )
        assert self.max_query_len == 1
        assert self.max_prefill_seq_len == 0
        assert self.max_decode_seq_len == max(self.seq_lens)

        assert self.query_start_loc is not None
        assert self.query_start_loc.shape == (num_queries + 1, )
        assert self.seq_start_loc is not None
        assert self.seq_start_loc.shape == (num_seqs + 1, )

        assert self.context_lens_tensor is not None
        assert self.context_lens_tensor.shape == (num_queries, )

        assert self.block_tables is not None
        assert self.block_tables.shape[0] == num_seqs

        # Update query lengths. Note that we update only queries and not seqs,
        # since tensors may be padded due to captured cuda graph batch size
        for i in range(num_queries):
            self.seq_lens[i] += 1
        self.max_decode_seq_len = max(self.seq_lens)


class FlashAttentionMetadataBuilder(
        AttentionMetadataBuilder[FlashAttentionMetadata]):

    def __init__(self, input_builder: "ModelInputForGPUBuilder"):
        self.slot_mapping: List[int] = []
        self.token_positions: List[int] = []
        self.prefill_seq_lens: List[int] = []
        self.context_lens: List[int] = []
        self.block_tables: List[List[int]] = []
        self.curr_seq_lens: List[int] = []

        # KV-Compress
        self.prefill_block_state_indices: List[int] = []
        self.decode_block_state_indices: List[int] = []
        self.kv_metric_buffer_lens: List[int] = []

        self.num_prefills = 0
        self.num_prefill_tokens = 0
        self.num_decode_tokens = 0
        self.has_prefix_cache_hit = False

        self.input_builder = input_builder
        self.runner = input_builder.runner
        self.sliding_window = input_builder.sliding_window
        self.block_size = input_builder.block_size
        self.use_v2_block_manager = (
            input_builder.scheduler_config.use_v2_block_manager)
        self.use_kvcompress = bool(input_builder.kvcompress_config)

    def _add_seq_group_kvcompress(
            self, inter_data: "ModelInputForGPUBuilder.InterDataForSeqGroup"):
        """Add a sequence group to the metadata. With KV-Compress we only need
        to handle num_prefills, num_prefill_tokens, prefill_seq_lens and
        curr_seq_lens here.
        """
        is_prompt = inter_data.is_prompt

        for (token_len, seq_len, curr_seq_len, query_len, context_len) in zip(
                 [len(t) for t in inter_data.input_tokens],
                 inter_data.orig_seq_lens, inter_data.seq_lens,
                 inter_data.query_lens, inter_data.context_lens):
            self.kv_metric_buffer_lens.append(inter_data.kv_metric_buffer_lens[0])

            # Used during prompt itertations
            self.block_tables.append([])
            self.context_lens.append(context_len)

            if is_prompt:
                self.prefill_block_state_indices.append(
                    inter_data.block_state_index_mapping[0])
                self.num_prefills += 1
                self.num_prefill_tokens += token_len
                self.prefill_seq_lens.append(seq_len)
                self.token_positions.extend(list(range(seq_len)))
            else:
                self.decode_block_state_indices.append(
                    inter_data.block_state_index_mapping[0])
                assert query_len == 1, (
                    "seq_len: {}, context_len: {}, query_len: {}".format(
                        seq_len, context_len, query_len))
                self.num_decode_tokens += query_len
                self.curr_seq_lens.append(curr_seq_len)
                self.token_positions.append(seq_len - 1)

    def _add_seq_group(
            self, inter_data: "ModelInputForGPUBuilder.InterDataForSeqGroup",
            chunked_prefill_enabled: bool, prefix_cache_hit: bool):
        """Add a sequence group to the metadata. Specifically update/append
        1. context length.
        2. block table.
        3. slot mapping.
        """
        is_prompt = inter_data.is_prompt
        block_tables = inter_data.block_tables

        for (seq_id, token_len, seq_len, curr_seq_len, query_len, context_len,
             curr_sliding_window_block) in zip(
                 inter_data.seq_ids, [len(t) for t in inter_data.input_tokens],
                 inter_data.orig_seq_lens, inter_data.seq_lens,
                 inter_data.query_lens, inter_data.context_lens,
                 inter_data.curr_sliding_window_blocks):
            self.context_lens.append(context_len)

            if is_prompt:
                self.num_prefills += 1
                self.num_prefill_tokens += token_len
                self.prefill_seq_lens.append(seq_len)
            else:
                assert query_len == 1, (
                    "seq_len: {}, context_len: {}, query_len: {}".format(
                        seq_len, context_len, query_len))
                self.num_decode_tokens += query_len
                self.curr_seq_lens.append(curr_seq_len)

            # Compute block table.
            # TODO(sang): Combine chunked prefill and prefix caching by
            # only allowing multiple of block_size chunk size.
            # NOTE: This only works for oooooooxxx style attention.
            block_table = []
            if prefix_cache_hit:
                # NOTE(woosuk): For flash-attn, the block table should
                # include the entries for the incoming prefill tokens.
                block_table = block_tables[seq_id]
            elif ((chunked_prefill_enabled or not is_prompt)
                  and block_tables is not None):
                if curr_sliding_window_block == 0:
                    block_table = block_tables[seq_id]
                else:
                    block_table = block_tables[seq_id][
                        -curr_sliding_window_block:]
            self.block_tables.append(block_table)

            # Compute slot mapping.
            is_profile_run = is_block_tables_empty(block_tables)
            start_idx = compute_slot_mapping_start_idx(
                is_prompt, query_len, context_len, self.sliding_window,
                self.use_v2_block_manager)
            compute_slot_mapping(is_profile_run, self.slot_mapping, seq_id,
                                 seq_len, context_len, start_idx,
                                 self.block_size, inter_data.block_tables)

    def build(self, seq_lens: List[int], query_lens: List[int],
              cuda_graph_pad_size: int, batch_size: int):
        """Build attention metadata with on-device tensors.

        Args:
            seq_lens: The maybe padded sequence lengths of the input sequences.
            query_lens: The query lengths of the input sequences.
            cuda_graph_pad_size: The padding size for cuda graph.
                                 -1 if cuda graph is not used.
            batch_size: The maybe padded batch size.
        """
        device = self.runner.device
        use_captured_graph = cuda_graph_pad_size != -1
        use_kvcompress = self.runner.kvc_state is not None
        block_state = self.runner.kvc_state.block_state if use_kvcompress else None
        is_profile_run = is_block_tables_empty(
            self.input_builder.inter_data_list[0].block_tables)

        if use_kvcompress:
            for inter_data in self.input_builder.inter_data_list:
                # Neither chunked prefill nor prefix caching
                # currently supported with KV-Compress
                self._add_seq_group_kvcompress(inter_data)
        else:
            prefix_cache_hit = any([
                inter_data.prefix_cache_hit
                for inter_data in self.input_builder.inter_data_list
            ])
            for inter_data in self.input_builder.inter_data_list:
                self._add_seq_group(inter_data,
                                    self.input_builder.chunked_prefill_enabled,
                                    prefix_cache_hit)

        max_query_len = max(query_lens)
        max_prefill_seq_len = max(self.prefill_seq_lens, default=0)
        max_decode_seq_len = max(self.curr_seq_lens, default=0)
        num_decode_tokens = self.num_decode_tokens

        if use_kvcompress:
            # NOTE: KV-Compress handles context_lens, slot_mapping and block_tables
            # separately during decoding, passing on-device tensors directly from
            # block manager.
            assert not (self.prefill_block_state_indices
                        and self.decode_block_state_indices)
            assert self.prefill_block_state_indices or self.decode_block_state_indices
            assert self.prefill_block_state_indices or not is_profile_run
            if self.prefill_block_state_indices:
                # use original context_lens and block_tables
                block_tables = make_tensor_with_pad(
                    self.block_tables,
                    pad=0,
                    dtype=torch.int,
                    device=device,
                )
                context_lens_tensor = async_tensor_h2d(self.context_lens, torch.int,
                                                       device, self.runner.pin_memory)
                if is_profile_run:
                    slot_mapping_tensor = torch.ones(
                        (
                            self.runner.kvcompress_config.num_layers,
                            self.num_prefill_tokens,
                            self.runner.kvcompress_config.num_kv_heads,
                        ),
                        dtype=torch.int,
                        device=device
                    ) * PAD_SLOT_ID
                else:
                    slot_mapping_tensor = torch.cat(
                        [block_state.get_block_state_seq_view(i)
                                    .get_prefill_slot_mapping()
                            for i in self.prefill_block_state_indices],
                        dim=1,
                    )
            elif self.decode_block_state_indices:
                decode_block_state_view = block_state.get_block_state_batch_view(
                    self.decode_block_state_indices)
                block_tables = decode_block_state_view.get_block_tables()
                context_lens_tensor = decode_block_state_view.get_context_lens()
                slot_mapping_tensor = decode_block_state_view.get_decode_slot_mapping()
        elif use_captured_graph:
            self.slot_mapping.extend([PAD_SLOT_ID] * cuda_graph_pad_size)
            self.block_tables.extend([] * cuda_graph_pad_size)
            num_decode_tokens = batch_size

            # The shape of graph_block_tables is
            # [max batch size, max context len // block size].
            input_block_tables = self.runner.graph_block_tables[:batch_size]
            for i, block_table in enumerate(self.block_tables):
                if block_table:
                    input_block_tables[i, :len(block_table)] = block_table
            block_tables = torch.from_numpy(input_block_tables).to(
                device=device, non_blocking=True)
        else:
            block_tables = make_tensor_with_pad(
                self.block_tables,
                pad=0,
                dtype=torch.int,
                device=device,
            )
        assert max_query_len > 0, ("query_lens: {}".format(query_lens))

        assert device is not None

        kv_metric_buffer_len = token_positions = None
        if use_kvcompress:
            kv_metric_buffer_len = async_tensor_h2d(self.kv_metric_buffer_lens,
                                                    torch.int, device,
                                                    self.runner.pin_memory)
            token_positions = async_tensor_h2d(self.token_positions,
                                               torch.int, device,
                                               self.runner.pin_memory)
        else:
            # NOTE: KV-Compress handles context_lens, slot_mapping and block_tables
            # separately, passing on-device tensors directly from block manager.
            context_lens_tensor = async_tensor_h2d(self.context_lens, torch.int,
                                                device, self.runner.pin_memory)
            slot_mapping_tensor = async_tensor_h2d(self.slot_mapping, torch.long,
                                                device, self.runner.pin_memory)

        seq_lens_tensor = async_tensor_h2d(seq_lens, torch.int, device,
                                           self.runner.pin_memory)
        query_lens_tensor = async_tensor_h2d(query_lens, torch.long, device,
                                             self.runner.pin_memory)
        query_start_loc = torch.zeros(query_lens_tensor.shape[0] + 1,
                                      dtype=torch.int32,
                                      device=device)
        seq_start_loc = torch.zeros(seq_lens_tensor.shape[0] + 1,
                                    dtype=torch.int32,
                                    device=device)
        torch.cumsum(seq_lens_tensor,
                     dim=0,
                     dtype=seq_start_loc.dtype,
                     out=seq_start_loc[1:])
        torch.cumsum(query_lens_tensor,
                     dim=0,
                     dtype=query_start_loc.dtype,
                     out=query_start_loc[1:])

        return FlashAttentionMetadata(
            num_prefills=self.num_prefills,
            slot_mapping=slot_mapping_tensor,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            kv_cache_dtype=self.runner.kv_cache_dtype,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=max_query_len,
            max_prefill_seq_len=max_prefill_seq_len,
            max_decode_seq_len=max_decode_seq_len,
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=use_captured_graph,
            kv_metrics=None if is_profile_run or self.runner.kvc_state is None else
                       self.runner.kvc_state.kv_metrics,
            kv_metric_buffer_len=kv_metric_buffer_len,
            token_positions=token_positions,
            prefill_kv_metric_window_size=
                self.runner.kvcompress_config.prefill_metric_collection_window_size
                if self.runner.kvcompress_config else None,
            prefill_kv_metric_block_size=
                self.runner.kvcompress_config.prefill_metric_collection_block_size
                if self.runner.kvcompress_config else None,
            kv_metric_use_l2=self.runner.kvcompress_config.kv_metric_use_l2
                             if self.runner.kvcompress_config else None,
            kv_metric_use_average=self.runner.kvcompress_config.kv_metric_use_average
                                  if self.runner.kvcompress_config else None,
            kv_metric_use_maxpool=self.runner.kvcompress_config.kv_metric_use_maxpool
                                  if self.runner.kvcompress_config else None,
            enable_flash_kvc=self.runner.kvcompress_config.enable_flash_kvc
                             if self.runner.kvcompress_config else None,
        )


class FlashAttentionImpl(AttentionImpl):
    """
    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prefill_tokens ----------------->|
    |<--prefill_0-->|<--prefill_1-->|...|<--prefill_N-1--->|

    Otherwise, the layout is as follows:
    |<----------------- num_decode_tokens ------------------>|
    |<--decode_0-->|..........|<--decode_M-1-->|<--padding-->|

    Generation tokens can contain padding when cuda-graph is used.
    Currently, prompt tokens don't contain any padding.

    The prompts might have different lengths, while the generation tokens
    always have length 1.

    If chunked prefill is enabled, prefill tokens and decode tokens can be
    batched together in a flattened 1D query.

    |<----- num_prefill_tokens ---->|<------- num_decode_tokens --------->|
    |<-prefill_0->|...|<-prefill_N-1->|<--decode_0-->|...|<--decode_M-1-->|

    Currently, cuda graph is disabled for chunked prefill, meaning there's no
    padding between prefill and decode tokens.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "FlashAttention does not support block-sparse attention.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = ((sliding_window, sliding_window)
                               if sliding_window is not None else (-1, -1))
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if sliding_window is not None:
            # NOTE(woosuk): flash-attn's sliding window does not work with
            # paged KV cache.
            raise ValueError(
                "Sliding window is not supported in FlashAttention.")

        support_head_sizes = FlashAttentionBackend.get_supported_head_sizes()
        if head_size not in support_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by FlashAttention. "
                f"Supported head sizes are: {support_head_sizes}.")

    def repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
        tokens, n_kv_heads, head_dim = x.shape
        return (x[:, :,
                  None, :].expand(tokens, n_kv_heads, n_rep,
                                  head_dim).reshape(tokens, n_kv_heads * n_rep,
                                                    head_dim))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "FlashAttentionImpl")

        # NOTE(woosuk): FlashAttention does not support FP8 KV cache.
        assert k_scale == 1.0 and v_scale == 1.0, (
            "key/v_scale is not supported in FlashAttention.")

        num_tokens, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        layer_index = attn_metadata.layer_index

        CHECKPOINTER.condition(checkpoint_layer=layer_index)

        kv_metrics = attn_metadata.kv_metrics
        kvcompress_enabled = bool(kv_metrics)
        kv_metric_buffer_len = attn_metadata.kv_metric_buffer_len
        kv_metric_use_l2 = attn_metadata.kv_metric_use_l2
        kv_metric_use_average = attn_metadata.kv_metric_use_average
        kv_metric_use_maxpool = attn_metadata.kv_metric_use_maxpool
        prefill_observed_queries = attn_metadata.prefill_kv_metric_window_size
        prefill_max_observed_block_size = attn_metadata.prefill_kv_metric_block_size

        CHECKPOINTER.checkpoint('flash_attn__query', query)
        CHECKPOINTER.checkpoint('flash_attn__key', key)
        CHECKPOINTER.checkpoint('flash_attn__value', value)

        if kv_cache is not None:
            # Reshape the input keys and values and store them in the cache.
            # If kv_cache is not provided, the new key and value tensors are
            # not cached. This happens during the initial memory profiling run.
            if kvcompress_enabled:
                assert k_scale == v_scale
                key_cache, value_cache = KVCAttention.split_kv_cache(
                    kv_cache, self.head_size)
                assert kv_metrics
                assert layer_index is not None
                # Extract layer-dependent metadata
                # TODO remove line below - we now initialize new KV's to zero
                # and apply bias during the compression
                kv_metric_head_bias = torch.zeros(
                    self.num_kv_heads,
                    dtype=torch.float,
                    device=key.device,
                )  #attn_metadata.kv_metric_head_bias[layer_index]
                slot_mapping = attn_metadata.slot_mapping[layer_index]

                CHECKPOINTER.checkpoint('flash_attn__slot_mapping', slot_mapping)

                KVCAttention.write_to_paged_cache(key, value, key_cache,
                                                  value_cache, kv_metrics.metrics,
                                                  slot_mapping,
                                                  torch.zeros_like(kv_metric_head_bias),
                                                  attn_metadata.kv_cache_dtype,
                                                  k_scale, v_scale)

                CHECKPOINTER.checkpoint('flash_attn__kv_metrics', kv_metrics.metrics)

                if kv_metrics.random:
                    # if running random-eviction baseline, randomize the metrics that
                    # were just inserted
                    kv_metrics.randomize_metric_slots(slot_mapping)
            else:
                key_cache = kv_cache[0]
                value_cache = kv_cache[1]
                ops.reshape_and_cache_flash(
                    key,
                    value,
                    key_cache,
                    value_cache,
                    attn_metadata.slot_mapping.flatten(),
                    self.kv_cache_dtype,
                    k_scale,
                    v_scale,
                )

            CHECKPOINTER.checkpoint('flash_attn__key_cache', key_cache)
            CHECKPOINTER.checkpoint('flash_attn__value_cache', value_cache)

        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        assert key.shape[0] == num_prefill_tokens + num_decode_tokens
        assert value.shape[0] == num_prefill_tokens + num_decode_tokens

        output = torch.empty_like(query)
        # Query for decode. KV is not needed because it is already cached.
        decode_query = query[num_prefill_tokens:]
        # QKV for prefill.
        query = query[:num_prefill_tokens]
        key = key[:num_prefill_tokens]
        value = value[:num_prefill_tokens]

        assert query.shape[0] == num_prefill_tokens
        assert decode_query.shape[0] == num_decode_tokens

        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            if kvcompress_enabled:
                BENCHMARKER.start_range("prefill_attn_kvc")

                # Extract layer-dependent metadata
                slot_mapping = attn_metadata.slot_mapping[layer_index]
                slot_mapping = slot_mapping[:num_prefill_tokens]

                CHECKPOINTER.checkpoint('flash_attn__prefill_slot_mapping', slot_mapping)

                if attn_metadata.enable_flash_kvc and FLASH_KVC_ENABLED:
                    BENCHMARKER.start_range("flash_kvc_prefill")
                    assert self.alibi_slopes is None, (
                        "use of alibi with KVC is unsupported")
                    assert self.sliding_window == (-1, -1), (
                        "use of sliding window with KVC is unsupported")
                    assert prefill_observed_queries <= 64, (
                        "query range greater than query block size is not supported")
                    assert not kv_metric_use_average, (
                        "use_average not supported with flash-attn-kvc")
                    BENCHMARKER.start_range("flash_attn_kvc_varlen_func")
                    out, sm_lse, kvc_S = flash_attn_kvc_varlen_func(
                        q=query,
                        k=key,
                        v=value,
                        cu_seqlens_q=prefill_meta.seq_start_loc,
                        cu_seqlens_k=prefill_meta.seq_start_loc,
                        max_seqlen_q=prefill_meta.max_prefill_seq_len,
                        max_seqlen_k=prefill_meta.max_prefill_seq_len,
                        softmax_scale=self.scale,
                        causal=True,
                        window_size=self.sliding_window,
                        alibi_slopes=self.alibi_slopes,
                        key_attn_agg_window=prefill_observed_queries,
                    )
                    BENCHMARKER.end_range("flash_attn_kvc_varlen_func")
                    BENCHMARKER.start_range("convert_kvc_S_to_attn")
                    suffix_attn = convert_kvc_S_to_attn(
                        kvc_S,
                        sm_lse,
                        prefill_observed_queries,
                        prefill_meta.seq_start_loc,
                        self.scale,
                        kv_metric_buffer_len,
                    )
                    BENCHMARKER.end_range("convert_kvc_S_to_attn")
                    kv_metric_out = _collect_kv_prefill_metrics(
                        suffix_attn,
                        kv_metric_use_l2,
                        kv_metric_use_maxpool
                    )
                    BENCHMARKER.end_range("flash_kvc_prefill")
                else:
                    BENCHMARKER.start_range("naive_kvc_prefill")
                    BENCHMARKER.start_range("flash_attn_varlen_func")
                    out = flash_attn_varlen_func(
                        q=query,
                        k=key,
                        v=value,
                        cu_seqlens_q=prefill_meta.seq_start_loc,
                        cu_seqlens_k=prefill_meta.seq_start_loc,
                        max_seqlen_q=prefill_meta.max_prefill_seq_len,
                        max_seqlen_k=prefill_meta.max_prefill_seq_len,
                        softmax_scale=self.scale,
                        causal=True,
                        window_size=self.sliding_window,
                        alibi_slopes=self.alibi_slopes,
                    )
                    BENCHMARKER.end_range("flash_attn_varlen_func")
                    BENCHMARKER.start_range("repeat_kv")
                    if self.num_kv_heads != self.num_heads:
                        # Interleave for MQA workaround.
                        key = self.repeat_kv(key, self.num_queries_per_kv)
                        value = self.repeat_kv(value, self.num_queries_per_kv)
                    BENCHMARKER.end_range("repeat_kv")
                    _, kv_metric_out = _naive_kvc_attention(
                        query,
                        key,
                        value,
                        prefill_meta.seq_lens,
                        self.scale,
                        kv_metric_buffer_len,
                        n_observed=prefill_observed_queries,
                        max_observed_block_size=prefill_max_observed_block_size,
                        use_l2=kv_metric_use_l2,
                        use_average=kv_metric_use_average,
                        use_maxpool=kv_metric_use_maxpool,
                    )
                    BENCHMARKER.end_range("naive_kvc_prefill")

                CHECKPOINTER.checkpoint('flash_attn__prefill_out', out)
                CHECKPOINTER.checkpoint('flash_attn__prefill_kv_metric_out', kv_metric_out)

                kv_metrics.aggregate_prefill(
                    kv_metric_out,
                    slot_mapping,
                )

                CHECKPOINTER.checkpoint('flash_attn__prefill_kv_metrics_agg', kv_metrics.metrics)

                assert output[:num_prefill_tokens].shape == out.shape
                output[:num_prefill_tokens] = out

                BENCHMARKER.end_range("prefill_attn_kvc")
            elif (kv_cache is None or prefill_meta.block_tables is None
                    or prefill_meta.block_tables.numel() == 0):
                # normal attention
                # When block_tables are not filled, it means q and k are the
                # prompt, and they have the same length.
                BENCHMARKER.start_range("flash_attn_varlen_func")
                out = torch.ops.vllm.flash_attn_varlen_func(
                    q=query,
                    k=key,
                    v=value,
                    cu_seqlens_q=prefill_meta.seq_start_loc,
                    cu_seqlens_k=prefill_meta.seq_start_loc,
                    max_seqlen_q=prefill_meta.max_prefill_seq_len,
                    max_seqlen_k=prefill_meta.max_prefill_seq_len,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=self.sliding_window,
                    alibi_slopes=self.alibi_slopes,
                    softcap=self.logits_soft_cap,
                )
                BENCHMARKER.end_range("flash_attn_varlen_func")
                assert output[:num_prefill_tokens].shape == out.shape
                output[:num_prefill_tokens] = out
            else:
                # prefix-enabled attention
                assert prefill_meta.seq_lens is not None
                max_seq_len = max(prefill_meta.seq_lens)
                output[:
                       num_prefill_tokens] = torch.ops.vllm.flash_attn_varlen_func(  # noqa
                           q=query,
                           k=key_cache,
                           v=value_cache,
                           cu_seqlens_q=prefill_meta.query_start_loc,
                           max_seqlen_q=prefill_meta.max_query_len,
                           cu_seqlens_k=prefill_meta.seq_start_loc,
                           max_seqlen_k=max_seq_len,
                           softmax_scale=self.scale,
                           causal=True,
                           alibi_slopes=self.alibi_slopes,
                           block_table=prefill_meta.block_tables,
                           softcap=self.logits_soft_cap,
                       )

        if decode_meta := attn_metadata.decode_metadata:
            # Decoding run.
            if kvcompress_enabled:
                # Extract layer-dependent metadata
                block_tables = decode_meta.block_tables[layer_index]
                context_lens = decode_meta.context_lens_tensor[layer_index]
                decode_positions = attn_metadata.token_positions[num_prefill_tokens:]
                output[num_prefill_tokens:] = KVCAttention.forward_decode(
                    decode_query,
                    key_cache,
                    value_cache,
                    block_tables,
                    context_lens,
                    kv_metrics.token_positions,
                    decode_positions,
                    kv_metric_buffer_len,
                    decode_meta.max_decode_seq_len,
                    attn_metadata.kv_cache_dtype,
                    self.num_kv_heads,
                    self.scale,
                    self.alibi_slopes,
                    k_scale,
                    v_scale,
                    kv_metrics.temp_metrics,
                    kv_metrics.temp_v2_metrics,
                    kv_metrics.record_decoding_metrics,
                )

                CHECKPOINTER.checkpoint('flash_attn__decode_block_tables', block_tables)
                CHECKPOINTER.checkpoint('flash_attn__decode_context_lens', context_lens)
                CHECKPOINTER.checkpoint('flash_attn__decode_kv_positions', kv_metrics.token_positions)
                CHECKPOINTER.checkpoint('flash_attn__decode_q_positions', decode_positions)
                CHECKPOINTER.checkpoint('flash_attn__decode_temp_kv_metrics', kv_metrics.temp_metrics)

            else:
                output[
                    num_prefill_tokens:] = torch.ops.vllm.flash_attn_with_kvcache(
                        decode_query.unsqueeze(1),
                        key_cache,
                        value_cache,
                        block_table=decode_meta.block_tables,
                        cache_seqlens=decode_meta.seq_lens_tensor,
                        softmax_scale=self.scale,
                        causal=True,
                        alibi_slopes=self.alibi_slopes,
                        softcap=self.logits_soft_cap,
                    ).squeeze(1)

        CHECKPOINTER.checkpoint('flash_attn__output', output)
        CHECKPOINTER.end_condition()

        # Reshape the output tensor.
        return output.view(num_tokens, hidden_size)


# For KV-Compress we require aggregated attention allocated to each key
@BENCHMARKER.wrap()
def _naive_kvc_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    prompt_lens: List[int],
    scale: float,
    kv_metric_buffer_len: torch.Tensor,
    n_observed: int = 32,
    max_observed_block_size: int = 4096,
    use_l2: bool = True,
    use_average: bool = False,
    use_maxpool: bool = True,
) -> torch.Tensor:
    # output = torch.empty_like(query)
    seq_len, num_heads, _ = key.shape
    kv_metric_output = torch.empty(
        (seq_len, num_heads),
        dtype=torch.float,
        device=key.device,
    )
    start = 0
    for i, prompt_len in enumerate(prompt_lens):
        end = start + prompt_len
        start_trunc = end - min(prompt_len, n_observed)
        kv_metric_output[start:end].fill_(0)
        for l in range(start_trunc, end, max_observed_block_size):
            _, kv_metrics = _naive_kvc_masked_attention(
                query[l:min(l+max_observed_block_size, end)],
                key[start:end],
                value[start:end],
                scale,
                kv_metric_buffer_len[i],
                use_l2,
                use_average,
                use_maxpool,
                q_offset=l-start,
            )
            # TODO(woosuk): Unnecessary copy. Optimize.
            # output[start:end].copy_(out)
            kv_metric_output[start:end] += kv_metrics.T
        start += prompt_len

    return None, kv_metric_output


def _naive_kvc_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    kv_metric_buffer_len: torch.Tensor,
    use_l2: bool,
    use_average: bool,
    use_maxpool: bool,
    q_offset: int = 0,
) -> torch.Tensor:
    n_observed, num_heads, head_dim = query.shape
    seq_len, num_heads, head_dim = key.shape
    ones = torch.ones(n_observed,
                      seq_len,
                      dtype=query.dtype,
                      device=query.device)
    attn_mask = torch.triu(ones, diagonal=q_offset + 1)
    attn_mask = attn_mask * torch.finfo(query.dtype).min
    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    attn_weights = attn_weights + attn_mask.float()
    attn_weights = torch.softmax(attn_weights, dim=-1)
    if use_l2:
        attn_weights = attn_weights ** 2
    # out = torch.einsum("hqk,khd->qhd", attn_weights.to(value.dtype), value)
    kv_metric_mask = torch.tril(ones, diagonal=q_offset - kv_metric_buffer_len)
    # sum L2 of attention over queries
    kv_metrics = (attn_weights * kv_metric_mask).sum(dim=-2)
    if use_average:
        # Need to rescale KV metrics by token position
        # since we will be normalizing by token position later
        kv_metrics *= (
            torch.arange(1, kv_metrics.size(-1) + 1,
                         device=kv_metrics.device,
                         dtype=torch.float)[None]
            / n_observed
        )
    if use_maxpool:
        kv_metrics = F.max_pool1d(
            kv_metrics,
            kernel_size=7,
            padding=7//2,
            stride=1,
        )
    return None, kv_metrics


@BENCHMARKER.wrap()
def _collect_kv_prefill_metrics(
    suffix_attn,
    use_l2: bool,
    use_maxpool: bool,
):
    if use_l2:
        suffix_attn = suffix_attn ** 2
    kv_metrics = suffix_attn.sum(dim=-1)
    if use_maxpool:
        kv_metrics = F.max_pool1d(
            kv_metrics.T,
            kernel_size=7,
            padding=7//2,
            stride=1,
        ).T
    return kv_metrics


def convert_kvc_S_to_attn(s, sm_lse, kagg_window, cu_seqlens, scale, buffer_lens):
    assert s.dtype == sm_lse.dtype == torch.float
    s_ = s * scale
    start = cu_seqlens[0]
    lse = []
    min_val = torch.finfo(torch.float).min
    for l, buf in zip(cu_seqlens[1:], buffer_lens):
        curr_l = l - start
        nq = min(kagg_window, curr_l)
        offset = curr_l - nq
        attn_mask = 0
        if buf > 0:
            ones = torch.ones_like(s_[start:l,:,-nq:])
            attn_mask = torch.triu(ones, diagonal=offset + 1 - buf)
            attn_mask = attn_mask * min_val
        lse = sm_lse[:,l-nq:l]
        s_[start:l,:,-nq:] += (attn_mask - lse[None])

        start = l

    s_ = s_.exp()
    s_[s == float('-inf')] = 0

    return s_
