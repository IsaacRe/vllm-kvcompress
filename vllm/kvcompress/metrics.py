from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import torch
import numpy as np
from tqdm.auto import tqdm

from vllm.kvcompress.block import BlockMetadata
from vllm.debug import CHECKPOINTER
from vllm.benchmark import BENCHMARKER
from vllm._custom_ops import count_block_evictions

MAX_INT = 2147483000
_BIAS_KEY = "bias"
_POSITION_RANGE_KEY = "pos_bins"


def _load_kv_head_bias(path: str) -> "KVHeadBias":
    # currently just read from local filesystem
    ext = path.split('.')[-1]
    if ext == "safetensors":
        from safetensors import safe_open
        f = safe_open(path)
        return KVHeadBias(
            f.get_tensor(_BIAS_KEY).type(torch.float),
            f.get_tensor(_POSITION_RANGE_KEY).type(torch.int),
        )
    elif ext in ["pt", "bin"]:
        f = torch.load(path)
        return KVHeadBias(
            f.get_tensor(_BIAS_KEY).type(torch.float),
            f.get_tensor(_POSITION_RANGE_KEY).type(torch.int),
        )
    elif ext == "npz":
        import numpy as np
        f = np.load(path)
        return KVHeadBias(
            torch.tensor(f[_BIAS_KEY]).type(torch.float),
            torch.tensor(f[_POSITION_RANGE_KEY]).type(torch.int),
        )
    else:
        raise ValueError(f"Unsupported file format {ext}")


@dataclass
class KVHeadBias:
    bias: torch.Tensor  # [ num_layers, num_kv_heads, num_bins ]
    position_bins: torch.Tensor  # [ num_bins ]

    def to(self, device) -> "KVHeadBias":
        self.bias = self.bias.to(device)
        self.position_bins = self.position_bins.to(device)
        return self

    def get_bias_for_position(
        self,
        positions: torch.Tensor,
        layer_indices: torch.Tensor,
        head_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return head bias for input tensor of input token positions.
        Zero bias is returned for elements of `positions` with a value
        less than zero.

        Args:
            positions: [ num_blocks, block_size ] Tensor containing positions.
            layer_indices: [ num_blocks ] Tensor containing layer indices.
            head_indices: [ num_blocks ] Tensor containing KV head indices.
        """
        _, block_size = positions.shape
        # [ num_blocks, block_size, num_bins ]
        bin_indices = positions[...,None] >= self.position_bins[None,None]
        # [ num_blocks, block_size ]
        bin_indices = bin_indices.cumsum(dim=-1).max(dim=-1).values - 1
        # [ num_blocks * block_size ]
        bias = self.bias[(layer_indices.repeat_interleave(block_size),
                          head_indices.repeat_interleave(block_size),
                          bin_indices.flatten())]
        bias = bias.view(-1, block_size)  # [ num_blocks, block_size ]
        # Set bias for -1 positions to 0
        bias[positions < 0] = 0
        return bias


@dataclass
class SortedMetricOutputs:
    sorted_indices: torch.Tensor
    seq_block_offsets: torch.Tensor
    layer_by_block: torch.Tensor
    head_by_block: torch.Tensor
    logical_block_num_by_block: torch.Tensor
    token_positions: torch.Tensor


class CompressionMetrics:
    """Class to handle allocation and management of metrics tensor
    used to determine priority of KV eviction during cache compression.

    Lifetime of the metrics tensor:
    1a. When a new KV corresponding to a prefill token is added to cache
        the attention values of all queries against that KV during prefill
        are aggregated and used to initialize that KV's metrics slot.
    1b. When a new KV corresponding to a decode token is added to cache
        the corresponding metrics slot is set to zero.
    2. Whenever attention is computed against an existing KV during
        decoding that attention is aggregated and output into the
        temp_metrics tensor. After each model forward pass the temp_metrics
        values are aggregated into the metrics tensor.
    3. During compression scheduling, immediately before KV sorting, bias
        is recomputed and reapplied using the current sequence length.
        Concretely, we subtract the previous bias (computed from the distance
        between the KV's token position and the sequence length during its
        last compression) and add the new bias (computed from distance
        between its token position and the current sequence length).
    4. No change occurs upon KV-eviction (due to either compression or
        sequence preemption).
    """

    def __init__(
        self,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        num_queries_per_kv: int,
        max_kv_per_sort: int,
        kv_head_bias_file: Optional[str],
        kv_head_bias_weight: float,
        device: str = "cuda:0",
        random: bool = False,
        even_layer_evict: bool = False,
        use_l2: bool = True,
        use_average: bool = False,
        record_decoding_metrics: bool = True,
        num_attention_sinks: int = 0,
    ) -> None:
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_queries_per_kv
        self.device = device
        self.num_sinks = num_attention_sinks

        # Random eviction baseline for testing purposes
        self.random = random

        self.even_layer_evict = even_layer_evict
        # Type of aggregation to use over recevied attention per KV
        self.use_l2 = use_l2
        self.use_average = use_average
        # Whether to continue recording KV attention during decoding
        self.record_decoding_metrics = record_decoding_metrics

        # Bias when aggregating metrics for each KV head.
        # Recorded metric for each KV will be the actual metric
        # plus bias for its corresponding KV head.
        print(f"Allocating head bias - Mem: {torch.cuda.memory_allocated(0) * 1e-9}")
        if kv_head_bias_file:
            expected_shape = num_layers, num_kv_heads
            self.kv_metric_head_bias = (
                _load_kv_head_bias(kv_head_bias_file).to(device)
            )
            if self.kv_metric_head_bias.bias.shape[:-1] != expected_shape:
                raise ValueError(f"expected shape {(*expected_shape, -1)} for "
                                 f"KV head bias tensor but got "
                                 f"{self.kv_metric_head_bias.bias.shape}")
        else:
            self.kv_metric_head_bias = KVHeadBias(
                torch.zeros(
                    (num_layers, num_kv_heads, 1),
                    dtype=torch.float,
                    device=device,
                ),
                torch.zeros((1,), dtype=torch.int, device=device),
            )
        self.kv_metric_bias_weight = kv_head_bias_weight

        CHECKPOINTER.checkpoint('kv_metric_head_bias__bias', self.kv_metric_head_bias.bias)
        CHECKPOINTER.checkpoint('kv_metric_head_bias__position_bins', self.kv_metric_head_bias.position_bins)

        # Crucial for limiting runtime and memory
        # overhead of sort during each iteration.
        # Should be < 100,000,000, but can be set
        # as low as ~5,000,000 while maintaining
        # good performance.
        self.max_kv_per_sort = max_kv_per_sort

        # Configured after profiling
        self.num_blocks = None

        self.unassigned_seq_idx = -1  # unassigned blocks cannot match valid indices

        # Allocated after profiling
        self.metrics = None
        self.temp_metrics = None
        self.temp_v2_metrics = None
        self.seq_index_by_block = None
        self.layer_index_by_block = None
        self.head_index_by_block = None
        self.logical_block_num_by_block = None
        self.token_positions = None

        # Used to diff when updating head bias
        self.prev_seq_lens: Dict[int, int] = {}

    def reinit_kv_metadata(self) -> None:
        num_blocks = self.num_blocks
        self.clear_kv_metadata()
        self.init_kv_metadata(num_blocks)

    def clear_kv_metadata(self) -> None:
        self.num_blocks = None
        self.metrics = None
        self.temp_metrics = None
        self.temp_v2_metrics = None
        self.seq_index_by_block = None
        self.layer_index_by_block = None
        self.head_index_by_block = None
        self.logical_block_num_by_block = None
        self.token_positions = None

    def init_kv_metadata(self, num_blocks: int) -> None:
        assert self.num_blocks is None, "already initialized"
        self.num_blocks = num_blocks

        # Running metrics for every KV in cache
        print(f"Allocating kv metrics - Mem: {torch.cuda.memory_allocated(0) * 1e-9}")
        self.metrics = torch.empty(
            (num_blocks, self.block_size),
            dtype=torch.float32,
            device=self.device,
        )

        if self.random:
            # Sample metrics randomly for random eviction
            self.metrics.uniform_()

        # Reduction space where new metrics for each iteration
        # are recorded.
        # We record metrics of each key separately for each query
        # that attended to it and reduce over the queries
        # before aggregating into the running metrics tensor.
        print(f"Allocating temp kv metrics - Mem: {torch.cuda.memory_allocated(0) * 1e-9}")
        self.temp_metrics = torch.empty(
            (num_blocks, self.block_size, self.num_queries_per_kv),
            dtype=torch.float32,
            device=self.device,
        )
        self.temp_v2_metrics = torch.empty_like(self.temp_metrics)
        print(f"Allocating block metadata - Mem: {torch.cuda.memory_allocated(0) * 1e-9}")
        # Sequence index for every block in cache
        self.seq_index_by_block = torch.ones(
            (self.num_blocks,),
            dtype=torch.int,
            device=self.device,
        ) * self.unassigned_seq_idx
        self.layer_index_by_block = torch.empty(
            (self.num_blocks,),
            dtype=torch.int,
            device=self.device,
        )
        self.head_index_by_block = torch.empty(
            (self.num_blocks,),
            dtype=torch.int,
            device=self.device,
        )
        self.logical_block_num_by_block = torch.empty(
            (self.num_blocks,),
            dtype=torch.int,
            device=self.device,
        )
        self.token_positions = torch.empty(
            (self.num_blocks, self.block_size),
            dtype=torch.int,
            device=self.device,
        )
        self.validate_metadata()

    def profile_schedule_evictions(self):
        # Should not have begun handling requests
        assert self.num_blocks is None, "cannot profile after initialization"
        sort_blocks = (self.max_kv_per_sort + self.block_size - 1) // self.block_size
        total_heads = self.num_layers * self.num_kv_heads
        blocks_per_head = (sort_blocks + total_heads - 1) // total_heads
        total_blocks = blocks_per_head * total_heads
        self.init_kv_metadata(total_blocks)
        self.seq_index_by_block[:total_blocks] = 0
        self.head_index_by_block[:total_blocks] = (
            torch.arange(self.num_kv_heads)
                 .repeat_interleave(blocks_per_head)
                 .repeat(self.num_layers)
                 .to(self.device)
        )
        self.layer_index_by_block[:total_blocks] = (
            torch.arange(self.num_layers)
                 .repeat_interleave(self.num_kv_heads * blocks_per_head)
                 .to(self.device)
        )
        self.logical_block_num_by_block[:total_blocks] = (
            torch.arange(blocks_per_head)
                 .repeat(total_heads)
                 .to(self.device)
        )
        context_lens = (torch.ones((self.num_layers, 1, self.num_kv_heads),
                                   dtype=torch.int,
                                   device=self.device)
                        * self.block_size * blocks_per_head)
        hanging_token_count = torch.ones((self.num_layers, 1, self.num_kv_heads),
                                         dtype=torch.int,
                                         device=self.device) * self.block_size

        evicted_kv_offsets = (
            ((context_lens.transpose(0, 1) + self.block_size - 1) // self.block_size)
            * self.block_size
        ).flatten().cumsum(dim=0)
        evicted_kv_offsets = torch.cat(
            [torch.zeros_like(evicted_kv_offsets[:1]), evicted_kv_offsets[:-1]]
        ).reshape(*context_lens.transpose(0, 1).shape).type(torch.int)

        init_mem = torch.cuda.max_memory_allocated(
            torch.device(self.device))
        self.schedule_evictions(
            [0],
            [1],
            [total_blocks],
            context_lens,
            hanging_token_count,
            evicted_kv_offsets,
            [32],
            profile=True,
        )
        final_mem = torch.cuda.max_memory_allocated(
            torch.device(self.device))
        self.clear_kv_metadata()
        torch.cuda.empty_cache()
        print(f"PROFILED SORT: {final_mem - init_mem}")
        return final_mem - init_mem

    def clear_temp_metrics(self) -> None:
        """Temp metric output space must be set to zero before each
        forward pass to ensure that KVs for sequences that are not
        processed during the iteration remain unchanged after
        aggregation."""
        self.temp_metrics.zero_()

    def insert_metadata(
        self,
        metadata: BlockMetadata,
    ) -> None:
        """Insert metadata for newly allocated blocks."""
        # debug
        # new_mask = torch.zeros_like(self.seq_index_by_block, dtype=torch.bool)
        # new_mask[metadata.physical_blocks] = True
        # alloc_mask = self.seq_index_by_block >= 0
        # print(f"INSERTING METADATA: {alloc_mask.sum()} total allocated blocks")
        # assert not (new_mask & alloc_mask).any(), "slot already allocated"

        self.seq_index_by_block[metadata.physical_blocks] = metadata.seq_indices
        self.logical_block_num_by_block[metadata.physical_blocks] = (
            metadata.logical_blocks.type(torch.int))
        self.layer_index_by_block[metadata.physical_blocks] = metadata.layer_indices
        self.head_index_by_block[metadata.physical_blocks] = metadata.head_indices
        self.token_positions[metadata.physical_blocks] = metadata.token_positions
        # self.validate_metadata()
        # metadata.validate_even_layer_evict()
        # self.validate_metadata_even_layer_evict()

    def remove_metadata(self, physical_blocks: torch.Tensor) -> None:
        # print(f"REMOVING METADATA: {physical_blocks}")
        # Used to set seq index for evicted blocks back to -1 so they aren't
        # selected during sort
        self.seq_index_by_block[physical_blocks] = -1
        # self.validate_metadata()
        # self.validate_metadata_even_layer_evict()

    def validate_metadata(self) -> None:
        # All allocated blocks (with seq idx >= 0) should have valid setting
        # for head index
        allocated_mask = self.seq_index_by_block >= 0
        assert (self.head_index_by_block[allocated_mask] < self.num_kv_heads).all()

    def validate_metadata_even_layer_evict(self) -> None:
        if self.even_layer_evict:
            allocated_mask = self.seq_index_by_block >= 0
            # should have same number of allocated KVs per layer
            # randomly sample to make check less time-consuming
            check_idx = np.random.randint(self.num_layers)
            print(allocated_mask.sum())
            check_idx_count = (self.layer_index_by_block[allocated_mask] == check_idx).sum()
            per_layer_count = self.layer_index_by_block[allocated_mask].numel() / self.num_layers
            assert check_idx_count == per_layer_count, f'{check_idx_count=}, {per_layer_count=} (are you using control_layers?)'

    def randomize_metric_slots(self, slot_mapping: torch.Tensor) -> None:
        flat_indices = slot_mapping.flatten().type(torch.long)
        self.metrics[flat_indices] = self.metrics[flat_indices].uniform_()

    @BENCHMARKER.wrap()
    def aggregate_prefill(
        self,
        prefill_metrics: torch.Tensor,  # [num_prefill_tokens, num_q_heads]
        slot_mapping: torch.Tensor,     # [num_prefill_tokens, num_kv_heads]
    ) -> None:
        """Insert metrics and metadata for prefilled KVs at the current
        layer. Metrics returned from prefill are already L2 sums of key-attention
        and have shape [key_seq_len, query_heads]. Metrics should have already
        been initialized to the corresponding head bias.
        """
        if self.random:
            return  # keep random metrics when doing random eviction

        # TODO replace double gather with more efficient kernel
        seq_len, _ = prefill_metrics.shape
        # TODO validate dim order on reshape

        # Insert metrics
        per_head_metrics = (prefill_metrics
                            .view(seq_len, self.num_kv_heads, -1)
                            .sum(dim=-1))
        assert per_head_metrics.shape == slot_mapping.shape, \
            f"{per_head_metrics.shape} != {slot_mapping.shape}"
        flat_slots = slot_mapping.flatten()
        self.metrics.view(-1).scatter_(
            dim=0,
            index=flat_slots,
            src=(
                self.metrics.view(-1).gather(dim=0, index=flat_slots)
                + per_head_metrics.flatten()
            ),
        )

    def aggregate_decode(self):
        """Aggregate KV metrics recorded during last iteration using
        simple heuristic from the paper: Total squared attention.
        Then update the running KV metrics with this aggregation.
        """
        if self.random or not self.record_decoding_metrics:
            return  # keep random metrics when doing random eviction
        temp_metrics = (self.temp_metrics ** 2 if self.use_l2
                        else self.temp_metrics)

        self.metrics += temp_metrics.sum(dim=-1)

    def schedule_evictions(
        self,
        seq_indices: List[int],
        seq_positions: List[int],
        evicted_blocks_per_seq: List[int],
        context_lens: torch.Tensor,
        hanging_token_count: torch.Tensor,
        evicted_kv_offsets: torch.Tensor,
        num_protected: List[int],
        uniform_evict: bool = False,
        debug={},
        profile=False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(seq_indices) > 0

        # TODO make sure this method does not modify global KV metadata

        # TODO handle this better
        assert list(sorted(seq_indices)) == seq_indices, (
            "schedule_evictions input not ordered by ascending index")

        total_heads = self.num_layers * self.num_kv_heads

        # Build sequence mask
        seq_mask = self.seq_index_by_block == seq_indices[0]
        all_seq_positions = torch.empty((max(seq_indices) + 1,),
                                        dtype=torch.int,
                                        device=self.device)
        all_seq_positions[seq_indices[0]] = seq_positions[0]
        all_num_protected = torch.empty_like(all_seq_positions)
        all_num_protected[seq_indices[0]] = num_protected[0]
        all_seq_batch_slots = torch.empty_like(all_seq_positions)
        all_seq_batch_slots[seq_indices[0]] = 0
        for i, (seq_index, seq_pos, seq_num_protected) in enumerate(zip(
            seq_indices[1:], seq_positions[1:], num_protected[1:]
        ), start=1):
            seq_mask |= self.seq_index_by_block == seq_index
            # assert (self.seq_index_by_block == seq_index).sum() > 0
            all_seq_positions[seq_index] = seq_pos
            all_num_protected[seq_index] = seq_num_protected
            all_seq_batch_slots[seq_index] = i

        expanded_mask = seq_mask[...,None].expand_as(self.metrics)
        masked_metrics = self.metrics[expanded_mask]
        masked_seq_indices = self.seq_index_by_block[seq_mask]
        masked_layer_indices = self.layer_index_by_block[seq_mask]
        masked_head_indices = self.head_index_by_block[seq_mask]
        masked_logical_block_nums = self.logical_block_num_by_block[seq_mask]
        masked_token_position = self.token_positions[seq_mask]
        masked_logical_block_indices = (
            masked_logical_block_nums[:,None] * self.block_size
            + torch.arange(self.block_size, device=self.device, dtype=torch.int)[None]
        )

        if self.use_average:
            # Normalize KV metrics by the number of queries seen for each KV
            current_positions = all_seq_positions[
                masked_seq_indices.type(torch.int64)
            ]
            masked_query_count = current_positions[:,None] - masked_token_position
            masked_metrics /= masked_query_count.view(-1)

        bias = self.kv_metric_head_bias.get_bias_for_position(
            masked_token_position, masked_layer_indices, masked_head_indices
        )
        masked_metrics = masked_metrics + bias.view(-1) * self.kv_metric_bias_weight

        seq_offset = self.num_layers * self.num_kv_heads

        # 0. Set values of masked_metrics for keys that cannot be evicted
        # (due to being out of bounds or within the protected window) to inf
        # assert masked_seq_indices.max() < all_seq_positions.size(0)
        all_max_in_range = (all_seq_positions - all_num_protected).gather(
            dim=0, index=masked_seq_indices.type(torch.long)
        )
        # assert masked_seq_indices.max() < all_seq_batch_slots.size(0)
        masked_batch_slot_indices = all_seq_batch_slots.gather(
            dim=0, index=masked_seq_indices.type(torch.long)
        )
        masked_seq_layer_head_idx = (
            masked_batch_slot_indices * seq_offset
            + masked_layer_indices * self.num_kv_heads
            + masked_head_indices
        ).type(torch.long)
        # debug_unique_logical_block_nums = masked_logical_block_nums[masked_seq_layer_head_idx == 0]
        # assert debug_unique_logical_block_nums.size(0) == debug_unique_logical_block_nums.unique().size(0)
        # debug_unique_logical_indices = masked_logical_block_indices[masked_seq_layer_head_idx == 0].flatten()
        # assert debug_unique_logical_indices.size(0) == debug_unique_logical_indices.unique().size(0)
        # assert masked_seq_layer_head_idx.max() < context_lens.numel(), (masked_seq_layer_head_idx.max(), context_lens.numel())
        all_seq_context_lens = (
            context_lens.transpose(0, 1)
                        .flatten()
                        .gather(
                            dim=0,
                            index=masked_seq_layer_head_idx,
                        )
        )

        in_range_mask = (
            (masked_logical_block_nums < (all_seq_context_lens + self.block_size - 1) // self.block_size)[:,None]
            & (masked_token_position <= all_max_in_range[:,None])
            & (masked_token_position >= self.num_sinks)  # protect attention sinks for each head
        )
        masked_metrics[~in_range_mask.flatten()] = float('inf')


        # inactive_block_mask = (
        #     masked_logical_block_nums >= (all_seq_context_lens + self.block_size - 1) // self.block_size)[:,None]
        # # protect attention sinks for each head
        # # attn_sink_mask = masked_token_position < self.num_sinks
        # # masked_metrics[(attn_sink_mask | inactive_block_mask).flatten()] = float('inf')
        # masked_metrics[inactive_block_mask.flatten()] = float('inf')

        # in_range_mask = masked_token_position <= all_max_in_range[:,None]
        # # make sure empty KV slots come first in sort
        # masked_metrics[~in_range_mask.flatten()] = 0


        # 1.a. Sort masked_metrics (of size (total_blocks * block_size,))
        # by sequence, layer, head, metric value
        # masked_metrics -> sorted_masked_metrics
        sorted_masked_metrics, sorted_indices = masked_metrics.sort()
        # assert masked_seq_layer_head_idx.size(0) * self.block_size > sorted_indices.max()
        sorted_masked_seq_layer_head_idx, head_sorted_indices = (
            masked_seq_layer_head_idx.repeat_interleave(self.block_size)
                                      .gather(dim=0, index=sorted_indices)
                                      .sort(stable=True)
        )
        sorted_indices = sorted_indices[head_sorted_indices]
        sorted_masked_metrics = sorted_masked_metrics[head_sorted_indices]


        # 2. Gather metric for last KV that would be evicted for each block,
        # yielding (total_blocks,)-sized vector, masked_metric_blocks
        # with same order as sorted masked_metrics
        # sorted_masked_metrics -> sorted_masked_metric_blocks (offset by n for
        # each sequence, layer, head)

        # 2.a. Create sorted_hanging_token_blocks of size (total_blocks,)
        # containing the number of hanging tokens for the corresponding
        # sequence, head and layer of each block in sorted_masked_metrics
        # Sequence, layer, head indices per block ordered by (sequence, layer, head, block metric value)
        sorted_masked_seq_layer_head_idx_blocks = sorted_masked_seq_layer_head_idx.view(-1, self.block_size)[:,0]  # [ masked_blocks ]
        # assert hanging_token_count.numel() > sorted_masked_seq_layer_head_idx_blocks.max()
        sorted_hanging_tokens_blocks = (  # [ masked_blocks ]
            hanging_token_count.flatten()
                          .gather(dim=0, index=sorted_masked_seq_layer_head_idx_blocks.type(torch.long))
        )

        # 2.b. Create sorted_masked_metric_blocks
        # Block metric per block ordered by (sequence, layer, head, block metric value)
        reshape_sorted_masked_metrics = sorted_masked_metrics.view(-1, self.block_size)  # [ masked_blocks, block_size ]
        # assert reshape_sorted_masked_metrics.size(-1) > (sorted_hanging_tokens_blocks - 1).max()
        sorted_masked_metric_blocks = reshape_sorted_masked_metrics.gather(
            dim=-1, index=(sorted_hanging_tokens_blocks[:,None] - 1).type(torch.long)
        ).squeeze(-1)
        # sorted_masked_metric_blocks = reshape_sorted_masked_metrics[...,-1].squeeze(-1)
        # Empty blocks should not contain evictable KVs
        # assert not ((sorted_hanging_tokens_blocks < 1) & (sorted_masked_metric_blocks < float('inf'))).any()


        # 3. Select blocks to evict from sorted_masked_metric_blocks
        # and set eviction mask for sorted_masked_metrics
        sorted_masked_logical_indices = (
            masked_logical_block_indices.flatten()[sorted_indices].view(-1, self.block_size)
        )
        total_blocks_per_seq = (
            (context_lens + self.block_size - 1) // self.block_size
        ).sum(dim=0).sum(dim=-1)

        # Uniform evict second attempt - instead of forcing uniformity explicitly,
        # normalize metrics across heads and continue with variable-rate eviction
        # if uniform_evict:
        #     # Use head index from sorted_masked_seq_layer_head_idx_blocks to normalize
        #     # metrics in sorted_masked_metric_blocks within their respective attention head

        #     # collect per head sums
        #     sums = []
        #     block_counts = []
        #     for seq_layer_head_idx in tqdm(range(len(seq_indices) * total_heads)):
        #         mask = ((sorted_masked_seq_layer_head_idx_blocks == seq_layer_head_idx)
        #                 & (sorted_masked_metric_blocks != float('inf')))
        #         masked = sorted_masked_metric_blocks[mask]
        #         sums.append(masked.sum())
        #         block_counts.append(masked.numel())

        #     _, max_sum, max_block_count = list(sorted([(s / c, s, c) for s, c in zip(sums, block_counts)],
        #                                               reverse=True))[0]

        #     # apply normalization to sorted_masked_metric_blocks
        #     for seq_layer_head_idx, sum, block_count in tqdm(zip(range(len(seq_indices) * total_heads),
        #                                                          sums, block_counts)):
        #         mask = ((sorted_masked_seq_layer_head_idx_blocks == seq_layer_head_idx)
        #                 & (sorted_masked_metric_blocks != float('inf')))
        #         masked = sorted_masked_metric_blocks[mask]
        #         multiplier = max_sum * block_count / max_block_count / sum
        #         masked *= multiplier

        if uniform_evict:  # Uniform evict first attempt
            # (uniform eviction per head)
            # sorted_masked_metrics and sorted_masked_logical_indices already follow the ordering we want
            # so we can skip to (3.d.).
            # 3.d. Iterate over sequences and slice from each sequence's start to end offset
            # setting non-evicted KVs in sorted_masked_logical_indices to MAX INT
            offset = 0
            for i, blocks_to_evict in enumerate(evicted_blocks_per_seq):
                # will evict blocks_to_evict blocks per head, so divide by total
                # KV head count
                blocks_to_evict_per_head = blocks_to_evict // total_heads
                end_offset = offset + total_blocks_per_seq[i]

                # sorted_masked_logical_indices is sorted first by
                # (seq, layer, head) so we can reshape accordingly
                curr_seq_logical_indices = (
                    sorted_masked_logical_indices[offset:end_offset].reshape(total_heads, -1, self.block_size))

                curr_seq_logical_indices[:,blocks_to_evict_per_head:] = MAX_INT

                if not profile:
                    # import pdb; pdb.set_trace()
                    assert (sorted_masked_metric_blocks[offset:end_offset].view(total_heads, -1)[:,:blocks_to_evict_per_head] < float('inf')).all()

                # Remap eviction mask into sorted_masked_logical_indices
                sorted_masked_logical_indices[offset:end_offset] = curr_seq_logical_indices.view(-1, self.block_size)

                offset = end_offset
        else:
            # 3.a.(variable) Sort masked_metric_blocks by sequence, metric value to get
            # seq_sorted_masked_metric_blocks
            # sorted_masked_metric_blocks -> seq_sorted_masked_metric_blocks
            metric_sorted_masked_metric_blocks, metric_sorted_indices_blocks = sorted_masked_metric_blocks.sort()
            sorted_indices_blocks = sorted_indices.view(-1, self.block_size)[:,0] // self.block_size
            # assert sorted_indices_blocks.max() < masked_seq_indices.size(0), (sorted_indices_blocks.max(),  masked_seq_indices.size(0))
            # works because the KVs in each block represented by an element of sorted_indices_blocks will
            # have been part of the same sequence, layer and head (even though they are not necessarily
            # contiguous in the original ordering)
            sorted_seq_idx_blocks = masked_seq_indices[sorted_indices_blocks]
            metric_sorted_seq_idx_blocks = sorted_seq_idx_blocks[metric_sorted_indices_blocks]
            # masked -> sorted_indices -> sorted_indices_blocks (take first) -> metric_sorted_indices_blocks -> seq_sorted_indices_blocks
            # seq_sorted_indices_blocks goes from blocks in (seq, layer, head, metric)-ordered list
            # to blocks in (seq, metric)-ordered list.
            _, seq_sorted_indices_blocks = metric_sorted_seq_idx_blocks.sort(stable=True)
            seq_sorted_masked_metric_blocks = metric_sorted_masked_metric_blocks[seq_sorted_indices_blocks]
            seq_sorted_indices_blocks = metric_sorted_indices_blocks[seq_sorted_indices_blocks]

            # 3.c. Create sorted_masked_logical_indices and seq_sorted_masked_logical_indices
            # Note: We can't use sorted_indices_blocks here since the logical index ordering is
            # broken after re-ordering by sorted_indices.

            # print("TESTING NON MAX_INT INDICES")
            # for slh_idx in range(len(seq_indices) * 8 * 32):
            #     masked_sort_indices = sorted_masked_logical_indices[sorted_masked_seq_layer_head_idx_blocks == slh_idx].view(-1)
            #     ref_masked_sort_indices = debug['logical_indices_1'][debug['head_indices_1'] == slh_idx]
            #     assert (ref_masked_sort_indices[masked_sort_indices.size(0):] == MAX_INT).all()
            #     ref_masked_sort_indices = ref_masked_sort_indices[:masked_sort_indices.size(0)]
            #     evicted_mask = ref_masked_sort_indices != MAX_INT
            #     assert (ref_masked_sort_indices[evicted_mask] == masked_sort_indices[evicted_mask]).all(), f'{ref_masked_sort_indices=}\n{masked_sort_indices=}'
            seq_sorted_masked_logical_indices = sorted_masked_logical_indices[seq_sorted_indices_blocks,:]
            # seq_sorted_masked_seq_layer_head_indices = sorted_masked_seq_layer_head_idx[seq_sorted_indices_blocks]

            # 3.d. Iterate over sequences and slice from each sequence's start to end offset
            # setting non-evicted KVs in seq_sorted_masked_logical_indices to MAX INT
            # print(f'{total_blocks_per_seq=}')
            offset = 0
            # # debug
            # seq_sorted_seq_layer_head_idx = sorted_masked_seq_layer_head_idx_blocks[seq_sorted_indices_blocks]
            # debug_seq_block_count = debug['block_count'].sum(dim=-1).sum(dim=-1)
            # #
            for i, blocks_to_evict in enumerate(evicted_blocks_per_seq):
                end_offset = offset + total_blocks_per_seq[i]
                unevicted_offset = offset + blocks_to_evict
                # assert debug_seq_block_count[i] == blocks_to_evict
                # for l in range(32):
                #     for h in range(8):
                #         slh_idx = i * 32 * 8 + l * 8 + h
                #         block_count = (seq_sorted_seq_layer_head_idx[offset:unevicted_offset] == slh_idx).sum()
                #         assert block_count == debug['block_count'].view(-1)[slh_idx], f'{block_count=}, {debug["block_count"].view(-1)[slh_idx]=}, {debug["kv_count"].view(-1)[slh_idx]=}, {hanging_token_count[i,l,h]=}'
                evicted_infs, = torch.where(seq_sorted_masked_metric_blocks[:unevicted_offset] == float('inf'))
                num_evicted_infs = len(evicted_infs)
                if num_evicted_infs > 0:
                    unevicted_offset -= num_evicted_infs

                seq_sorted_masked_logical_indices[unevicted_offset:end_offset] = MAX_INT
                if not profile:
                    assert (seq_sorted_masked_metric_blocks[offset:unevicted_offset] < float('inf')).all()
                # print(f'{blocks_to_evict=}')
                # print(f'{seq_sorted_masked_logical_indices[:unevicted_offset].shape=}')
                # print(f'{seq_sorted_masked_seq_layer_head_indices[:unevicted_offset].shape=}')
                offset = end_offset

            # # debug
            # sorted_masked_seq_indices = (
            #     masked_seq_indices.repeat_interleave(self.block_size)[sorted_indices].view(-1, self.block_size)
            # )
            # seq_sorted_masked_seq_indices = sorted_masked_seq_indices[seq_sorted_indices_blocks,:]
            # for i, (seq_idx, blocks_to_evict) in enumerate(zip(seq_indices, evicted_blocks_per_seq)):
            #     seq_mask = seq_sorted_masked_seq_indices == seq_idx
            #     assert seq_mask.sum() == ((context_lens[:,i] + self.block_size - 1) // self.block_size * self.block_size).sum(), (seq_mask.sum(), context_lens[:,i].sum())
            #     evicted_kv_count_ = (
            #         (seq_sorted_masked_logical_indices[seq_sorted_masked_seq_indices == seq_idx] != MAX_INT).sum()
            #     )
            #     evicted_block_count_ = evicted_kv_count_ // self.block_size
            #     assert blocks_to_evict == evicted_block_count_, (i, seq_idx, blocks_to_evict, evicted_block_count_)
            #     assert debug_seq_block_count[i] == evicted_block_count_

            # # continue debug
            # print("TESTING BLOCK COUNT BEFORE")
            # for slh_idx in range(len(seq_indices) * 8 * 32):
            #     masked_sort_indices = seq_sorted_masked_logical_indices[seq_sorted_seq_layer_head_idx == slh_idx].view(-1)
            #     head_block_evictions = (masked_sort_indices != MAX_INT).sum() // self.block_size
            #     ref_head_block_evictions = debug['block_count'].view(-1)[slh_idx]
            #     assert head_block_evictions == ref_head_block_evictions, f'{head_block_evictions=}, {ref_head_block_evictions=}'

            # 3.e. Remap non-evicted blocks into sorted_masked_logical_indices
            sorted_masked_logical_indices[seq_sorted_indices_blocks,:] = seq_sorted_masked_logical_indices

            # print("TESTING BLOCK COUNT AFTER")
            # for slh_idx in range(len(seq_indices) * 8 * 32):
            #     masked_sort_indices = sorted_masked_logical_indices.view(-1, self.block_size)[sorted_masked_seq_layer_head_idx_blocks == slh_idx].view(-1)
            #     head_block_evictions = (masked_sort_indices != MAX_INT).sum() // self.block_size
            #     ref_head_block_evictions = debug['block_count'].view(-1)[slh_idx]
            #     assert head_block_evictions == ref_head_block_evictions, f'{head_block_evictions=}, {ref_head_block_evictions=}'

            # TODO ensure sorted_masked_logical_indices respected evicted_kv_offsets
            # TODO ensure that contiguous KVs in sorted_masked_logical_indices are also contiguous in
            # seq_sorted_masked_logical_indices

        sorted_masked_logical_indices = sorted_masked_logical_indices.flatten()

        # Count headwise block evictions. Kernel will also set logical indices for
        # any non-evicted KVs to MAX INT.
        # torch.empty_like on non-contiguous tensor yields a new non-contiguous tensor?
        evicted_block_count = torch.empty_like(evicted_kv_offsets)
        count_block_evictions(
            evicted_block_count,
            sorted_masked_logical_indices,
            evicted_kv_offsets,
            hanging_token_count,
            self.block_size,
            MAX_INT,
            evicted_blocks_per_seq,
        )

        # assert (debug['block_count'] == evicted_block_count).all()

        # print(evicted_blocks_per_seq.shape, evicted_block_count.sum(dim=-1).sum(dim=-1).shape)
        # assert (evicted_block_count.sum(dim=-1).sum(dim=-1) == torch.tensor(evicted_blocks_per_seq)).all()
        evicted_kv_count = torch.where(
            evicted_block_count > 0,
            (evicted_block_count - 1) * self.block_size + hanging_token_count,
            0,
        )
        if not profile:
            # import pdb; pdb.set_trace()
            pass

        if not profile:
            assert (context_lens.transpose(0, 1) >= evicted_kv_count).all()
            print(f"Max context length after compression: {(context_lens.transpose(0, 1) - evicted_kv_count).max()}")

        # assert (debug['kv_count'] == evicted_kv_count).all()

        # 4. Sort sorted_masked_logical_indices by logical index


        # for slh_idx in range(len(seq_indices) * 8 * 32):
        #     print(debug['kv_count'].view(-1)[slh_idx])
        #     logical_indices = sorted_masked_logical_indices[sorted_masked_seq_layer_head_idx == slh_idx]
        #     ref_logical_indices = debug['logical_indices_1'][debug['head_indices_1'] == slh_idx]
        #     kv_count = debug['kv_count'].view(-1)[slh_idx]
        #     assert (logical_indices[:kv_count] == ref_logical_indices[:kv_count]).all(), f'{logical_indices[:kv_count]=}, {ref_logical_indices[:kv_count]=}'

        # print("TEST BLOCK COUNT AGAIN")
        # for slh_idx in range(len(seq_indices) * 8 * 32):
        #     masked_sort_indices = sorted_masked_logical_indices.view(-1, self.block_size)[sorted_masked_seq_layer_head_idx_blocks == slh_idx].view(-1)
        #     head_block_evictions = (masked_sort_indices != MAX_INT).view(-1, self.block_size).any(dim=-1).sum()
        #     ref_head_block_evictions = debug['block_count'].view(-1)[slh_idx]
        #     assert head_block_evictions == ref_head_block_evictions, f'{slh_idx=}, {head_block_evictions=}, {ref_head_block_evictions=}'

        # 4.b. Sort
        # print(f"{sorted_masked_logical_indices.shape=}")
        logical_idx_sorted_masked_logical_indices, logical_idx_sorted_indices = (
            sorted_masked_logical_indices.sort()
        )
        # print(f"{logical_idx_sorted_indices.shape=}")
        # print(f"{sorted_masked_seq_layer_head_idx.shape=}")
        head_sorted_seq_layer_head_idx, head_sorted_indices = (
            sorted_masked_seq_layer_head_idx[logical_idx_sorted_indices].sort(stable=True)
        )
        # print(f"{head_sorted_indices.shape=}")
        # print(f"{logical_idx_sorted_masked_logical_indices.shape=}")
        head_sorted_masked_logical_indices = (
            logical_idx_sorted_masked_logical_indices[head_sorted_indices]
        )
        # print(f"{head_sorted_masked_logical_indices.shape=}")

        # for slh_idx in range(len(seq_indices) * 8 * 32):
        #     # print(debug['kv_count'].view(-1)[slh_idx])
        #     logical_indices = head_sorted_masked_logical_indices[head_sorted_seq_layer_head_idx == slh_idx]
        #     ref_logical_indices = debug['logical_indices_final'][debug['seq_layer_head_indices_final'] == slh_idx]
        #     kv_count = debug['kv_count'].view(-1)[slh_idx]
        #     assert (logical_indices[:kv_count] == ref_logical_indices[:kv_count]).all(), f'{slh_idx=}, {logical_indices[:kv_count+16]=}, {ref_logical_indices[:kv_count+16]=}'

        # print(torch.where((debug['logical_indices_final'][:head_sorted_masked_logical_indices.size(0)] != head_sorted_masked_logical_indices)))
        # assert (debug['logical_indices_final'][:head_sorted_masked_logical_indices.size(0)] == head_sorted_masked_logical_indices).all()

        return head_sorted_masked_logical_indices, evicted_kv_count, evicted_block_count

    @BENCHMARKER.wrap()
    def sort_seq_metrics(self, seq_indices: List[int], seq_positions: List[int], checkpoint: bool = True) -> SortedMetricOutputs:
        """Sort and return a view of the indices limited to a subset
        of all sequences.
        """
        assert len(seq_indices) > 0

        # TODO handle this better
        assert list(sorted(seq_indices)) == seq_indices, (
            "sort_seq_metrics input not ordered by ascending index")

        # Build metrics mask
        mask = self.seq_index_by_block == seq_indices[0]
        all_seq_positions = torch.empty((max(seq_indices) + 1,),
                                        dtype=torch.int,
                                        device=self.device)
        for seq_index, seq_pos in zip(seq_indices[1:], seq_positions[1:]):
            mask |= self.seq_index_by_block == seq_index
            # assert (self.seq_index_by_block == seq_index).sum() > 0
            all_seq_positions[seq_index] = seq_pos

        if checkpoint:
            CHECKPOINTER.checkpoint('sort__metrics_mask', mask)

        # allocated_mask = self.seq_index_by_block >= 0

        # debug
        # if checkpoint:
        #     assert not (mask & ~allocated_mask).any()
        #     self.validate_metadata()
        #     self.validate_metadata_even_layer_evict()

        # Mask
        expanded_mask = mask[...,None].expand_as(self.metrics)
        masked_metrics = self.metrics[expanded_mask]
        masked_seq_indices = self.seq_index_by_block[mask]
        masked_layer_indices = self.layer_index_by_block[mask]
        masked_head_indices = self.head_index_by_block[mask]
        masked_logical_block_nums = self.logical_block_num_by_block[mask]
        masked_token_position = self.token_positions[mask]

        # seq_pos_ - 1: last token to be processed (for sure not compressed and should therefore always have kv for all heads)
        # only raising for sequences that have been compressed
        # for seq_index, seq_pos_ in zip(seq_indices, seq_positions):
        #     # if seq_index in self.debug_seqs:
        #     #     continue
        #     mask_ = masked_seq_indices == seq_index
        #     for i in range(max(0, seq_pos_ - 50), seq_pos_, 10):
        #         last_pos_kv = masked_token_position[mask_] == i
        #         assert last_pos_kv.sum() >= self.num_layers * self.num_layers, (i, seq_pos_)
        #     non_evictable = masked_token_position[mask_] > seq_pos_ - 50
        #     assert non_evictable.sum() >= min(50, seq_pos_) * self.num_layers * self.num_layers, f"{non_evictable.sum()=} {seq_pos_ * self.num_layers * self.num_layers=}"

        # Normalize KV metrics by the number of queries seen for each KV
        # current_positions = all_seq_positions[
        #     masked_seq_indices.type(torch.int64)
        # ]
        # masked_query_count = current_positions[:,None] - masked_token_position
        # masked_metrics /= masked_query_count.view(-1)

        # Get bias for KVs being compressed based on their position bin
        bias = self.kv_metric_head_bias.get_bias_for_position(
            masked_token_position, masked_layer_indices, masked_head_indices
        )
        # assert masked_head_indices.max() < self.num_kv_heads, 'not working'

        if checkpoint:
            CHECKPOINTER.checkpoint('sort__bias', bias)
            CHECKPOINTER.checkpoint('sort__masked_metrics', masked_metrics)

        masked_metrics = masked_metrics + bias.view(-1) * self.kv_metric_bias_weight

        if checkpoint:
            CHECKPOINTER.checkpoint('sort__masked_seq_indices', masked_seq_indices)
            CHECKPOINTER.checkpoint('sort__masked_layer_indices', masked_layer_indices)
            CHECKPOINTER.checkpoint('sort__masked_head_indices', masked_head_indices)
            CHECKPOINTER.checkpoint('sort__masked_logical_block_nums', masked_logical_block_nums)

        # Sort by metric value then by sequence index
        # torch.sort uses ~8x memory of the input tensor (in addition
        # to memory of the input, itself)
        sorted_indices = masked_metrics.sort().indices.type(torch.int64)
        sorted_seq_indices = (
            masked_seq_indices.repeat_interleave(self.block_size)
                              .gather(dim=0, index=sorted_indices)
                              .sort()
        )

        # Sort by sequence index
        sorted_indices = sorted_indices.gather(
            dim=0,
            index=sorted_seq_indices.indices.type(torch.int64)
        )
        sorted_indices = sorted_indices.contiguous()

        if checkpoint:
            CHECKPOINTER.checkpoint('sort__sorted_indices', sorted_indices)

        # Get block offsets for each sequence
        seq_block_offsets = []
        for seq_index in seq_indices:
            seq_kv_indices, = torch.where(sorted_seq_indices.values == seq_index)
            kv_offset = seq_kv_indices.min()
            block_offset = kv_offset // self.block_size
            seq_block_offsets.append(block_offset.type(torch.int64))
        seq_block_offsets = torch.stack(seq_block_offsets)

        if checkpoint:
            CHECKPOINTER.checkpoint('sort__seq_block_offsets', seq_block_offsets)

        return SortedMetricOutputs(
            sorted_indices=sorted_indices,
            seq_block_offsets=seq_block_offsets,
            layer_by_block=masked_layer_indices.contiguous(),
            head_by_block=masked_head_indices.contiguous(),
            logical_block_num_by_block=masked_logical_block_nums.contiguous(),
            token_positions=masked_token_position.contiguous(),
        )

    def checkpoint(self) -> None:
        # Save metrics and metadata
        CHECKPOINTER.checkpoint('metrics__metrics', self.metrics)
        CHECKPOINTER.checkpoint('metrics__seq_index_by_block', self.seq_index_by_block)
        CHECKPOINTER.checkpoint('metrics__layer_index_by_block', self.layer_index_by_block)
        CHECKPOINTER.checkpoint('metrics__head_index_by_block', self.head_index_by_block)
        CHECKPOINTER.checkpoint('metrics__logical_block_num_by_block', self.logical_block_num_by_block)
        CHECKPOINTER.checkpoint('metrics__token_positions', self.token_positions)

