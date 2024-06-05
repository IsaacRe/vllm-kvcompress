from dataclasses import dataclass
from typing import List, Optional
import torch

from vllm.kvcompress.block import BlockMetadata


def _load_kv_head_bias(path: str) -> torch.Tensor:
    # currently just read from local filesystem
    ext = path.split('.')[-1]
    if ext == "safetensors":
        from safetensors import safe_open
        f = safe_open(path)
        bias = f.get_tensor(f.keys()[0])
    elif ext in ["pt", "bin"]:
        bias = torch.load(path)
    elif ext == "npy":
        import numpy as np
        bias = torch.tensor(np.load(path))
    elif ext == "npz":
        import numpy as np
        f = np.load(path)
        bias = torch.tensor(f[f.files[0]])
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
    1. When a new KV is added to cache, the corresponding metrics slot
    is populated with the bias for its kv head.
    2. 



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
            expected_shape = torch.Size((num_layers, num_kv_heads))
            self.kv_metric_head_bias = (
                _load_kv_head_bias(kv_head_bias_file)
                .type(torch.float).to(device)
            )
            if self.kv_metric_head_bias.shape != expected_shape:
                raise ValueError(f"expected shape {expected_shape} for "
                                 f"KV head bias tensor but got "
                                 f"{self.kv_metric_head_bias.shape}")
        else:
            self.kv_metric_head_bias = torch.zeros(
                (num_layers, num_kv_heads),
                dtype=torch.float32,
                device=device,
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

    def clear_kv_metadata(self) -> None:
        self.num_blocks = None
        self.metrics = None
        self.temp_metrics = None
        self.seq_index_by_block = None
        self.layer_index_by_block = None
        self.head_index_by_block = None
        self.logical_block_num_by_block = None

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

    def insert_metadata(
        self,
        metadata: BlockMetadata,
    ) -> None:
        self.seq_index_by_block[metadata.physical_blocks] = metadata.seq_indices
        self.logical_block_num_by_block[metadata.physical_blocks] = (
            metadata.logical_blocks.type(torch.int))
        self.layer_index_by_block[metadata.physical_blocks] = metadata.layer_indices
        self.head_index_by_block[metadata.physical_blocks] = metadata.head_indices

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
