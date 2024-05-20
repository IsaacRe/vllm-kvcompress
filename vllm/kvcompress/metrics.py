from dataclasses import dataclass
from typing import List, Optional
import torch


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
        kv_metric_head_bias: Optional[torch.Tensor],
        device: str = "cuda:0"
    ) -> None:
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_queries_per_kv
        self.device = device

        # Bias when aggregating metrics for each KV head.
        # Recorded metric for each KV will be the actual metric
        # plus bias for its corresponding KV head.
        self.kv_metric_head_bias = kv_metric_head_bias
        if self.kv_metric_head_bias is None:
            print(f"Allocating head bias - Mem: {torch.cuda.memory_allocated(0) * 1e-9}")
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

        # Allocated after profiling
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
        self.seq_index_by_block = torch.empty(
            (num_blocks,),
            dtype=torch.int,
            device=self.device,
        )
        self.layer_index_by_block = torch.empty(
            (num_blocks,),
            dtype=torch.int,
            device=self.device,
        )
        self.head_index_by_block = torch.empty(
            (num_blocks,),
            dtype=torch.int,
            device=self.device,
        )
        self.logical_block_num_by_block = torch.empty(
            (num_blocks,),
            dtype=torch.int,
            device=self.device,
        )

    def clear_temp_metrics(self) -> None:
        """Temp metric output space must be set to zero before each
        forward pass to ensure that KVs for sequences that are not
        processed during the iteration remain unchanged after
        aggregation."""
        self.temp_metrics.zero_()

    def aggregate_prefill(
        self,
        prefill_metrics: torch.Tensor,  # [num_prefill_tokens, num_q_heads]
        slot_mapping: torch.Tensor,     # [num_prefill_tokens, num_kv_heads]
    ) -> None:
        """Metrics returned from prefill are already L2 sums of key-attention
        and have shape [key_seq_len, query_heads]"""
        # TODO replace double gather with more efficient kernel
        seq_len, _ = prefill_metrics.shape
        # TODO validate dim order on reshape
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
        """Simple heuristic from the paper: Total squared attention"""
        self.metrics += (self.temp_metrics ** 2).sum(dim=-1)

    def sort_seq_metrics(self, seq_indices: List[int]) -> SortedMetricOutputs:
        """Sort and return a view of the indices limited to a subset
        of all sequences
        """
        # Build metrics mask
        mask = self.seq_index_by_block == seq_indices[0]
        for seq_index in seq_indices[1:]:
            mask |= self.seq_index_by_block == seq_index
        
        # Mask
        masked_metrics = self.metrics[mask.expand_as(self.metrics)]
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