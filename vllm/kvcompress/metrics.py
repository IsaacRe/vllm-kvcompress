from dataclasses import dataclass
from typing import List, Optional, Dict
import torch

from vllm.kvcompress.block import BlockMetadata

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
        """Return head bias for input tensor of input token positions. All tensors
        should have the same shape. Zero bias is returned for elements of `positions`
        with a value less than zero.

        Args:
            positions: Tensor containing positions.
            layer_indices: Tensor containing layer indices.
            head_indices: Tensor containing KV head indices.
        """
        # [ num_blocks, num_bins ]
        bin_indices = positions[...,None] >= self.position_bins[None]
        # [ num_blocks ]
        bin_indices = bin_indices.cumsum(dim=-1).max(dim=-1).values - 1
        # [ num_blocks ]
        bias = self.bias[(layer_indices, head_indices, bin_indices)]
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
        device: str = "cuda:0",
        random: bool = False,
    ) -> None:
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_queries_per_kv
        self.device = device

        # Random eviction baseline for testing purposes
        self.random = random

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

    def profile_sort(self):
        # Should not have begun handling requests
        assert self.num_blocks is None, "cannot profile after initialization"
        sort_blocks = (self.max_kv_per_sort + self.block_size - 1) // self.block_size
        self.init_kv_metadata(sort_blocks)
        self.seq_index_by_block[:] = 0
        init_mem = torch.cuda.max_memory_allocated(
            torch.device(self.device))
        self.sort_seq_metrics([0])
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
    
    def update_bias_for_positions(
        self, seq_indices: List[int], seq_lens: List[int]
    ) -> None:
        """Updates KV metrics for each of passed sequences to apply the KV head bias
        corresponding to the current sequence length bin it is in.
        """
        # Get previous sequence lengths (used to determine the bias currently
        # applied to self.metrics).
        # If a sequence is being compressed for the first time set prev_seq_len
        # to -1 to signify that no bias has been applied yet and cause
        # get_bias_for_position to return 0 prev_bias for these KVs.
        prev_seq_lens = [self.prev_seq_lens.get(seq_idx, -1) for seq_idx in seq_indices]

        # Get position, layer index and KV head index
        # All final tensors have shape [ num_seq_blocks ]
        seq_mask = self.seq_index_by_block == seq_indices[0]
        positions = torch.empty_like(seq_mask, dtype=torch.int)
        prev_positions = torch.empty_like(seq_mask, dtype=torch.int)
        positions[seq_mask] = seq_lens[0]
        prev_positions[seq_mask] = prev_seq_lens[0]
        for seq_idx, seq_len, prev_seq_len in zip(
            seq_indices[1:], seq_lens[1:], prev_seq_lens[1:]
        ):
            next_seq_mask = self.seq_index_by_block == seq_idx
            positions[next_seq_mask] = seq_len
            prev_positions[next_seq_mask] = prev_seq_len
            seq_mask |= next_seq_mask
        positions = positions[seq_mask]
        prev_positions = prev_positions[seq_mask]
        layer_indices = self.layer_index_by_block[seq_mask]
        head_indices = self.head_index_by_block[seq_mask]
        bias = self.kv_metric_head_bias.get_bias_for_position(
            positions, layer_indices, head_indices
        )
        prev_bias = self.kv_metric_head_bias.get_bias_for_position(
            prev_positions, layer_indices, head_indices
        )

        # Update metric bias
        self.metrics[seq_mask] += (bias - prev_bias)[...,None]

        # Update prev_seq_lens
        for seq_idx, seq_len in zip(seq_indices, seq_lens):
            self.prev_seq_lens[seq_idx] = seq_len

    def insert_metadata(
        self,
        metadata: BlockMetadata,
    ) -> None:
        """Insert metadata for newly allocated blocks."""
        self.seq_index_by_block[metadata.physical_blocks] = metadata.seq_indices
        self.logical_block_num_by_block[metadata.physical_blocks] = (
            metadata.logical_blocks.type(torch.int))
        self.layer_index_by_block[metadata.physical_blocks] = metadata.layer_indices
        self.head_index_by_block[metadata.physical_blocks] = metadata.head_indices
        self.token_positions[metadata.physical_blocks] = metadata.token_positions

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

    def sort_seq_metrics(self, seq_indices: List[int]) -> SortedMetricOutputs:
        """Sort and return a view of the indices limited to a subset
        of all sequences.
        """
        # TODO handle this better
        assert list(sorted(seq_indices)) == seq_indices, (
            "sort_seq_metrics input not ordered by ascending index")

        # Build metrics mask
        mask = self.seq_index_by_block == seq_indices[0]
        for seq_index in seq_indices[1:]:
            mask |= self.seq_index_by_block == seq_index
            assert (self.seq_index_by_block == seq_index).sum() > 0

        # Mask
        masked_metrics = self.metrics[mask[...,None].expand_as(self.metrics)]
        masked_seq_indices = self.seq_index_by_block[mask]
        masked_layer_indices = self.layer_index_by_block[mask]
        masked_head_indices = self.head_index_by_block[mask]
        masked_logical_block_nums = self.logical_block_num_by_block[mask]

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

        # Get block offsets for each sequence
        seq_block_offsets = []
        for seq_index in seq_indices:
            seq_kv_indices, = torch.where(sorted_seq_indices.values == seq_index)
            kv_offset = seq_kv_indices.min()
            block_offset = kv_offset // self.block_size
            seq_block_offsets.append(block_offset.type(torch.int64))
        seq_block_offsets = torch.stack(seq_block_offsets)

        return SortedMetricOutputs(
            sorted_indices=sorted_indices,
            seq_block_offsets=seq_block_offsets,
            layer_by_block=masked_layer_indices.contiguous(),
            head_by_block=masked_head_indices.contiguous(),
            logical_block_num_by_block=masked_logical_block_nums.contiguous(),
        )
