"""Attention layer with Flash and PagedAttention.

NOTE(woosuk): At the moment, this file includes a lot of duplicated code from
XFormers backend. The duplicated code will be removed once we use flash-attn or
flashinfer for all the attention operations.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from einops import rearrange
from flash_attn import flash_attn_varlen_func

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata,
                                              AttentionMetadataPerStage)
from vllm.attention.ops.paged_attn import (PagedAttention, KVCAttention,
                                           PagedAttentionMetadata)
from vllm.debug import CHECKPOINTER
from vllm.benchmark import BENCHMARKER


class FlashAttentionBackend(AttentionBackend):

    @staticmethod
    def get_impl_cls() -> Type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def make_metadata(*args, **kwargs) -> "FlashAttentionMetadata":
        return FlashAttentionMetadata(*args, **kwargs)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return PagedAttention.get_kv_cache_shape(num_blocks, block_size,
                                                 num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: Dict[int, int],
    ) -> None:
        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: Dict[int, List[int]],
    ) -> None:
        PagedAttention.copy_blocks(kv_caches, src_to_dists)


@dataclass
class FlashAttentionMetadata(AttentionMetadataPerStage,
                             PagedAttentionMetadata):
    """Metadata for FlashAttentionBackend.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """
    # Currently, input sequences can only contain all prompts
    # or all decoding. True if all sequences are prompts.
    is_prompt: bool
    # (batch_size,). The prompt length per sequence. None if it is a decoding.
    prompt_lens: Optional[List[int]]
    # prompt_lens stored as a tensor.
    prompt_lens_tensor: Optional[torch.Tensor]

    # NOTE(sang): Definition of context_len, subquery_len, and seqlen.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seqlen ----------------------|
    #                                   |- subquery_len -|

    # WARNING(sang): context_len has different definition depending on if it is
    # prefill vs decoding. When it is prefill, it doesn't include new tokens.
    # When it is for decoding, it includes a new token.

    # Maximum subquery length in the batch.
    max_subquery_len: Optional[int]
    # Maximum prompt length in the batch.
    max_prompt_len: Optional[int]
    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    subquery_start_loc: Optional[torch.Tensor]
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor]

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool


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
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = ((sliding_window, sliding_window)
                               if sliding_window is not None else (-1, -1))
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        suppored_head_sizes = PagedAttention.get_supported_head_sizes()
        if head_size not in suppored_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {suppored_head_sizes}.")

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
        attn_metadata: AttentionMetadata[FlashAttentionMetadata],
        kv_scale: float,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention and PagedAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
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
                                                  kv_metric_head_bias,
                                                  attn_metadata.kv_cache_dtype,
                                                  kv_scale)

                CHECKPOINTER.checkpoint('flash_attn__kv_metrics', kv_metrics.metrics)

                if kv_metrics.random:
                    # if running random-eviction baseline, randomize the metrics that
                    # were just inserted
                    kv_metrics.randomize_metric_slots(slot_mapping)
            else:
                key_cache, value_cache = PagedAttention.split_kv_cache(
                    kv_cache, self.num_kv_heads, self.head_size)
                PagedAttention.write_to_paged_cache(key, value, key_cache,
                                                    value_cache,
                                                    attn_metadata.slot_mapping,
                                                    attn_metadata.kv_cache_dtype,
                                                    kv_scale)

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
                # Extract layer-dependent metadata
                slot_mapping = attn_metadata.slot_mapping[layer_index]
                slot_mapping = slot_mapping[:num_prefill_tokens]

                CHECKPOINTER.checkpoint('flash_attn__prefill_slot_mapping', slot_mapping)

                if self.num_kv_heads != self.num_heads:
                    # Interleave for MQA workaround.
                    key = self.repeat_kv(key, self.num_queries_per_kv)
                    value = self.repeat_kv(value, self.num_queries_per_kv)
                out = flash_attn_varlen_func(
                    q=query,
                    k=key,
                    v=value,
                    cu_seqlens_q=prefill_meta.seq_start_loc,
                    cu_seqlens_k=prefill_meta.seq_start_loc,
                    max_seqlen_q=prefill_meta.max_prompt_len,
                    max_seqlen_k=prefill_meta.max_prompt_len,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=self.sliding_window,
                    alibi_slopes=self.alibi_slopes,
                )
                _, kv_metric_out = _naive_kvc_attention(
                    query,
                    key,
                    value,
                    prefill_meta.prompt_lens,
                    self.scale,
                    kv_metric_buffer_len,
                    n_observed=prefill_observed_queries,
                    max_observed_block_size=prefill_max_observed_block_size,
                    use_l2=kv_metric_use_l2,
                    use_average=kv_metric_use_average,
                    use_maxpool=kv_metric_use_maxpool,
                )

                CHECKPOINTER.checkpoint('flash_attn__prefill_out', out)
                CHECKPOINTER.checkpoint('flash_attn__prefill_kv_metric_out', kv_metric_out)

                kv_metrics.aggregate_prefill(
                    kv_metric_out,
                    slot_mapping,
                )

                CHECKPOINTER.checkpoint('flash_attn__prefill_kv_metrics_agg', kv_metrics.metrics)

                assert output[:num_prefill_tokens].shape == out.shape
                output[:num_prefill_tokens] = out
            elif kv_cache is None or prefill_meta.block_tables.numel() == 0:
                # normal attention
                # When block_tables are not filled, it means q and k are the
                # prompt, and they have the same length.
                BENCHMARKER.start_range("flash_attn_varlen_func")
                out = flash_attn_varlen_func(
                    q=query,
                    k=key,
                    v=value,
                    cu_seqlens_q=prefill_meta.seq_start_loc,
                    cu_seqlens_k=prefill_meta.seq_start_loc,
                    max_seqlen_q=prefill_meta.max_prompt_len,
                    max_seqlen_k=prefill_meta.max_prompt_len,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=self.sliding_window,
                    alibi_slopes=self.alibi_slopes,
                )
                BENCHMARKER.end_range("flash_attn_varlen_func")
                assert output[:num_prefill_tokens].shape == out.shape
                output[:num_prefill_tokens] = out
            else:
                # prefix-enabled attention
                # TODO(Hai) this triton kernel has regression issue (broke) to
                # deal with different data types between KV and FP8 KV cache,
                # to be addressed separately.
                output[:num_prefill_tokens] = PagedAttention.forward_prefix(
                    query,
                    key,
                    value,
                    key_cache,
                    value_cache,
                    prefill_meta.block_tables,
                    prefill_meta.subquery_start_loc,
                    prefill_meta.prompt_lens_tensor,
                    prefill_meta.context_lens,
                    prefill_meta.max_subquery_len,
                    self.alibi_slopes,
                )
        if decode_meta := attn_metadata.decode_metadata:
            # Decoding run.
            if kvcompress_enabled:
                # Extract layer-dependent metadata
                block_tables = decode_meta.block_tables[layer_index]
                context_lens = decode_meta.context_lens[layer_index]
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
                    decode_meta.max_context_len,
                    attn_metadata.kv_cache_dtype,
                    self.num_kv_heads,
                    self.scale,
                    self.alibi_slopes,
                    kv_scale,
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
                output[num_prefill_tokens:] = PagedAttention.forward_decode(
                    decode_query,
                    key_cache,
                    value_cache,
                    decode_meta.block_tables,
                    decode_meta.context_lens,
                    decode_meta.max_context_len,
                    attn_metadata.kv_cache_dtype,
                    self.num_kv_heads,
                    self.scale,
                    self.alibi_slopes,
                    kv_scale,
                )

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
                query[l:l+max_observed_block_size],
                key[start:end],
                value[start:end],
                scale,
                kv_metric_buffer_len[i],
                use_l2,
                use_average,
                use_maxpool,
                q_offset=l,
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
