from dataclasses import dataclass
from typing import List, Optional, Dict
import torch
import numpy as np

from vllm.kvcompress.block import BlockMetadata
from vllm.debug import CHECKPOINTER

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
    ) -> None:
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_queries_per_kv
        self.device = device

        # Random eviction baseline for testing purposes
        self.random = random
        
        self.even_layer_evict = even_layer_evict

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

    def profile_sort(self):
        # Should not have begun handling requests
        assert self.num_blocks is None, "cannot profile after initialization"
        sort_blocks = (self.max_kv_per_sort + self.block_size - 1) // self.block_size
        self.init_kv_metadata(sort_blocks)
        self.seq_index_by_block[:] = 0
        self.head_index_by_block[:] = 0
        self.layer_index_by_block[:] = 0
        self.logical_block_num_by_block[:] = 0
        init_mem = torch.cuda.max_memory_allocated(
            torch.device(self.device))
        self.sort_seq_metrics([0], [1], checkpoint=False)
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
        new_mask = torch.zeros_like(self.seq_index_by_block, dtype=torch.bool)
        new_mask[metadata.physical_blocks] = True
        alloc_mask = self.seq_index_by_block >= 0
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
        if self.random:
            return  # keep random metrics when doing random eviction

        self.metrics += (self.temp_metrics ** 2).sum(dim=-1)

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

        # Normalize KV metrics by the number of queries seen for each KV
        current_positions = all_seq_positions[
            masked_seq_indices.type(torch.int64)
        ]
        masked_query_count = current_positions[:,None] - masked_token_position
        masked_metrics /= masked_query_count.view(-1)

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
