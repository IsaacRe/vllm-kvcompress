"""A block manager that manages token blocks."""
from abc import ABC, abstractmethod
from itertools import count, takewhile
from os.path import commonprefix
from typing import Dict, List, Tuple, Optional, Set
from typing import Sequence as GenericSequence
import itertools
import torch

from vllm.kvcompress.block import BlockState, PhysicalTokenBlock, BlockStateView
from vllm.benchmark import BENCHMARKER
from vllm.config import KVCompressConfig
from vllm.core.evictor import EvictionPolicy, Evictor, make_evictor
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.logger import init_logger
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device

logger = init_logger(__name__)


# seq_id -> counts [ num_layers, num_kv_heads ]
FreedBlockCounts = Dict[int, torch.Tensor]


class BlockAllocatorBase(ABC):
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    @abstractmethod
    def __init__(self,
                 device: Device,
                 block_size: int,
                 num_blocks: int,
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU):
        pass

    @abstractmethod
    def allocate(self,
                 block_hash: Optional[int] = None,
                 num_hashed_tokens: int = 0) -> PhysicalTokenBlock:
        pass

    @abstractmethod
    def free(self, block: PhysicalTokenBlock) -> None:
        pass

    @abstractmethod
    def get_num_free_blocks(self) -> int:
        pass

    @abstractmethod
    def contains_block(self, block_hash: int) -> bool:
        pass

    @abstractmethod
    def update_hash(self, block_hash: int, block: PhysicalTokenBlock):
        pass


class BlockTableView(torch.Tensor):

    def __init__(self, data: torch.Tensor):
        self.data = data

    def __setitem__(self, _, __):
        raise ValueError("Attempted modification of BlockTableView")


class ParallelBlockAllocator(BlockAllocatorBase):
    """Manages free physical token blocks for a device.
    
    Becuase KV-Compress uses (num_layers * num_kv_heads) times more blocks
    than the vanilla paged KV cache, I observe heavy latency in the scheduler
    (up to 80ms!) when naively iterating over allocated/freed blocks.

    To remedy this, the modified block allocator indexes blocks in parallel to
    avoid latency with large number of concurrent allocation/frees.

    Because block number and free state is stored on GPU, mem size per block
    must be modified accordingly when determining the number of GPU blocks in
    the CacheEngine whenever the ParallelBlockAllocator is used.
    """

    def __init__(
        self,
        num_blocks: int,
        device: str = "cuda:0",
    ) -> None:
        self.num_blocks = num_blocks
        self.device = device

        # Initialize the free blocks.
        self.block_numbers = torch.arange(
            num_blocks, device=device
        )
        self.free_mask = torch.ones(
            (num_blocks,), dtype=torch.bool, device=device
        )
        self.free_count = num_blocks

    def allocate(self, num_blocks: int) -> BlockTableView:
        if num_blocks > self.free_count:
            raise ValueError("Out of memory! No free blocks are available.")
        self.free_count -= num_blocks
        allocated = self.block_numbers[self.free_mask][:num_blocks]
        self.free_mask[allocated] = False
        return BlockTableView(allocated)

    def free(self, block_numbers: torch.Tensor) -> None:
        if self.free_mask[block_numbers].any():
            raise ValueError("Double free! One or more blocks were freed twice.")
        if block_numbers.unique(dim=0).numel() != block_numbers.numel():
            raise ValueError("Double free! One or more blocks were freed twice.")
        self.free_count += block_numbers.numel()
        self.free_mask[block_numbers] = True

    def free_all(self) -> None:
        self.free_count = self.num_blocks
        self.free_mask[:] = True

    def get_num_free_blocks(self) -> int:
        return self.free_count
    
    def contains_block(self, block_hash: int) -> bool:
        raise NotImplementedError
    
    def update_hash(self, block_hash: int, block: PhysicalTokenBlock):
        raise NotImplementedError


class BlockSpaceManagerKVC(BlockSpaceManager):
    """Manages the mapping between logical and physical token blocks."""

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        kvcompress_config: KVCompressConfig,
        block_state: BlockState,
        watermark: float = 0.01,
        device: str = "cuda:0",
    ) -> None:
        self.kvcompress_config = kvcompress_config
        self.block_size = block_size
        self.num_kv_heads = kvcompress_config.num_kv_heads
        self.num_layers = kvcompress_config.num_layers
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks
        self.device = device

        self.watermark = watermark
        assert watermark >= 0.0

        self.watermark_blocks = int(watermark * num_gpu_blocks)

        # No swapping with KV-Compress so we only use a GPU allocator
        self.gpu_allocator = ParallelBlockAllocator(num_gpu_blocks)
        
        # KV-Compress uses pre-allocated block tables that are shared between
        # the model executor and scheduler/block manager
        self.block_state = block_state
        self.free_batch_slots = set(range(block_state.max_num_seqs))
        self.batch_slot_mapping = {}

    def _add_sequence(self, seq_id: int, seq_len: int) -> None:
        # Currently only support insertion of sequences
        # with same number of KVs across all heads/layers
        # (ie no swap-in of swapped-out, compressed sequences).
        assert len(self.free_batch_slots) > 0
        # If we just filled a block, update the block count to add tokens
        # that will be generated during the next decoding step. 
        seq_block_count = (seq_len + self.block_size) // self.block_size
        total_blocks = self.num_layers * self.num_kv_heads * seq_block_count
        block_numbers = self.gpu_allocator.allocate(total_blocks)
        seq_blocks = block_numbers.reshape(
            self.num_layers, self.num_kv_heads, seq_block_count
        ).type(torch.int).to(self.device)
        batch_slot_idx = self.free_batch_slots.pop()
        self.batch_slot_mapping[seq_id] = batch_slot_idx
        self.block_state.context_lens[:,batch_slot_idx] = seq_len
        # TODO debug
        self.block_state.block_tables[:,batch_slot_idx] = seq_blocks

    def _remove_sequence(self, seq_id: int) -> None:
        batch_slot_idx = self.batch_slot_mapping.pop(seq_id)
        self.free_batch_slots.add(batch_slot_idx)
        self.gpu_allocator.free(
            self.block_state
                .get_block_state_seq_view(batch_slot_idx)
                .get_allocated_blocks()
                .type(torch.int64)
                .to(self.gpu_allocator.device)
        )
        self.block_state.context_lens[:,batch_slot_idx] = 0

    def _get_new_block_count(self, seq_id: int, token_count: int):
        batch_slot_idx = self.batch_slot_mapping[seq_id]
        new_kv_count = (
            self.block_state.context_lens[:,batch_slot_idx] + token_count
        )
        return (new_kv_count + self.block_size) // self.block_size

    def _append_to_sequence(self, seq_id: int, token_count: int):
        batch_slot_idx = self.batch_slot_mapping[seq_id]

        old_mask = self.block_state.get_block_state_seq_view(batch_slot_idx).allocated_block_mask()
        self.block_state.context_lens[:,batch_slot_idx] += token_count
        new_mask = self.block_state.get_block_state_seq_view(batch_slot_idx).allocated_block_mask()
        new_mask = (new_mask & ~old_mask)
        # TODO debug
        self.block_state.block_tables[:,batch_slot_idx][new_mask] = (self.gpu_allocator
                                                       .allocate(new_mask.sum())
                                                       .to(self.device)
                                                       .type(torch.int))        

    def get_block_table(self, seq: Sequence) -> List[int]:
        return self.block_state.get_block_state_seq_view(
            self.batch_slot_mapping[seq.seq_id]
        )

    def get_block_state_batch_view(self, seqs: List[Sequence]) -> BlockStateView:
        return self.block_state.get_block_state_batch_view(
            [self.batch_slot_mapping[seq.seq_id] for seq in seqs]
        )

    def get_sequence_kv_count(self, seq: Sequence) -> int:
        batch_slot_idx = self.batch_slot_mapping[seq.seq_id]
        return int(self.block_state.context_lens[:,batch_slot_idx].sum())

    def can_allocate(self, seq_group: SequenceGroup) -> AllocStatus:
        assert (seq_group.num_seqs() <= 1
            ), "multi-child SequenceGroups are not compatible with KV-Compress"
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]

        # Assume that sequence has not been previously compressed and has same
        # context length across all layers/heads
        num_required_blocks = (
            (seq.get_len() + self.block_size) // self.block_size
            * self.num_layers * self.num_kv_heads
        )

        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()

        # Use watermark to avoid frequent cache eviction.
        if (self.num_total_gpu_blocks - num_required_blocks <
                self.watermark_blocks):
            return AllocStatus.NEVER
        if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def allocate(self, seq_group: SequenceGroup) -> None:
        # NOTE: Here we assume that all sequences in the group have the same
        # prompt.
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]

        self._add_sequence(seq_id=seq.seq_id, seq_len=seq.get_len())

    def can_append_slots(self,
                         seq_group: SequenceGroup,
                         num_lookahead_slots: int = 0) -> bool:
        assert (num_lookahead_slots == 0
                ), "lookahead allocation with KV-Compress not supported"
        assert (seq_group.num_seqs() <= 1,
                ), "multi-sequence SequenceGroups with KV-Compress are not supported"
        seq = seq_group.get_seqs(status=SequenceStatus.RUNNING)[0]

        # Simple heuristic: If there are at least num_kv_heads * num_layers free blocks
        # for each sequence, we can append.
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        new_block_count = self._get_new_block_count(seq_id=seq.seq_id, token_count=1)
        return new_block_count.sum() <= num_free_gpu_blocks

    def append_slots(
        self,
        seq: Sequence,
        num_lookahead_slots: int = 0,
    ) -> Dict[int, List[int]]:
        """Allocate a physical slot for a new token."""
        with BENCHMARKER.time("append_slots"):
            self._append_to_sequence(seq_id=seq.seq_id, token_count=1)
        return {}  # no copy on writes since we disallow multi-seq block references

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        raise NotImplementedError("forking while running KV-Compress not supported")

    def can_swap_in(
        self,
        seq_group: SequenceGroup,
        num_lookahead_slots: int = 0,
    ) -> bool:
        raise NotImplementedError("KV-Compress with swap-preemption not supported")

    def swap_in(
        self,
        seq_group: SequenceGroup,
        num_lookahead_slots: int = 0,
    ) -> Dict[int, int]:
        raise NotImplementedError("KV-Compress with swap-preemption not supported")

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        raise NotImplementedError("KV-Compress with swap-preemption not supported")

    def swap_out(self, seq_group: SequenceGroup) -> Dict[int, int]:
        raise NotImplementedError("KV-Compress with swap-preemption not supported")

    def free(self, seq: Sequence) -> None:
        if seq.seq_id not in self.batch_slot_mapping:
            # Already freed or haven't been scheduled yet.
            return
        with BENCHMARKER.time("free"):
            self._remove_sequence(seq.seq_id)

    def free_compressed_blocks(self, freed_block_count: FreedBlockCounts) -> None:
        seq_indices, removed_blocks = zip(*[
            (self.batch_slot_mapping[seq_id], freed_block_count[seq_id])
            for seq_id in freed_block_count.keys()
        ])
        # Free blocks before updating context lengths
        freed_block_counts_tensor = torch.stack(removed_blocks, dim=1)
        self.gpu_allocator.free(
            self.block_state.get_block_state_batch_view(seq_indices)
                            .get_last_n_allocated_blocks(freed_block_counts_tensor)
        )
        # Update context lengths
        self.block_state.remove_trailing_blocks(
            seq_indices=seq_indices,
            removed_block_count=removed_blocks,
        )

    def reset(self) -> None:
        self.batch_slot_mapping = {}
        self.free_batch_slots = set(range(self.block_state.max_num_seqs))
        self.block_state.clear()
        self.gpu_allocator.free_all()

    def get_slot_index(self, seq: Sequence) -> int:
        return self.batch_slot_mapping[seq.seq_id]

    def get_num_free_gpu_blocks(self) -> int:
        return self.gpu_allocator.get_num_free_blocks()

    def get_num_free_cpu_blocks(self) -> int:
        raise NotImplementedError("KV-Compress with swap-preemption not supported")

    def access_all_blocks_in_seq(
        self,
        seq: Sequence,
        access_time: float,
    ) -> None:
        pass

    def get_common_computed_block_ids(
            self, seqs: List[Sequence]) -> GenericSequence[int]:
        """Return the block ids that are common for a given sequence group.

        Used in prefill (can skip prefill of some blocks).
        """
        # Prefix caching not supported.
        return []

    def mark_blocks_as_computed(self, seq_group: SequenceGroup):
        pass
