"""A block manager that manages token blocks."""
from abc import ABC, abstractmethod
from itertools import count, takewhile
from os.path import commonprefix
from typing import Dict, List, Tuple, Optional, Set
from typing import Sequence as GenericSequence
import itertools
import torch
from tqdm.auto import tqdm

from vllm.kvcompress.state import KVCompressState
from vllm.kvcompress.block import BlockMetadata, PhysicalTokenBlock, BlockStateView
from vllm.benchmark import BENCHMARKER
from vllm.debug import CHECKPOINTER
from vllm.config import KVCompressConfig
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
                 num_blocks: int):
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
            raise ValueError(f"Out of memory! Requested {num_blocks} out of "
                             f"{self.free_count} available blocks.")
        self.free_count -= num_blocks
        allocated = self.block_numbers[self.free_mask][:num_blocks]
        self.free_mask[allocated] = False
        return BlockTableView(allocated)

    @BENCHMARKER.wrap()
    def free(self, block_numbers: torch.Tensor) -> None:
        # if self.free_mask[block_numbers].any():
        #     raise ValueError("Double free! One or more blocks were freed twice.")
        # if block_numbers.unique(dim=0).numel() != block_numbers.numel():
        #     raise ValueError("Double free! One or more blocks were freed twice.")
        self.free_count += block_numbers.numel()
        self.free_mask[block_numbers] = True

    def free_all(self) -> None:
        self.free_count = self.num_blocks
        self.free_mask[:] = True

    def get_num_free_blocks(self) -> int:
        return int(self.free_count)

    def contains_block(self, block_hash: int) -> bool:
        raise NotImplementedError

    def update_hash(self, block_hash: int, block: PhysicalTokenBlock):
        raise NotImplementedError

    def checkpoint(self) -> None:
        # Save free mask
        CHECKPOINTER.checkpoint('allocator__free_mask', self.free_mask)


class BlockSpaceManagerKVC(BlockSpaceManager):
    """Manages the mapping between logical and physical token blocks."""

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        kvcompress_config: KVCompressConfig,
        shared_state: KVCompressState,
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
        self.block_state = shared_state.block_state
        self.kv_metrics = shared_state.kv_metrics
        self.free_batch_slots = set(range(self.block_state.max_num_seqs))
        self.batch_slot_mapping = {}

    def reinit(self):
        self.gpu_allocator.free_all()
        self.block_state.clear()
        self.kv_metrics.reinit_kv_metadata()

    def _validate_allocator(self) -> None:
        free_blocks = self.gpu_allocator.block_numbers[self.gpu_allocator.free_mask]
        allocated_blocks = self.block_state.get_allocated_blocks()
        # assert (self.block_state.context_lens == 0).all()
        failing_blocks = []
        for b in tqdm(allocated_blocks, total=allocated_blocks.shape[0]):
            if (free_blocks == b).any():
                failing_blocks.append(b.item())
        if failing_blocks:
            print(f'{failing_blocks=}')
        assert not failing_blocks

    def _add_sequence(self, seq_id: int, seq_len: int) -> None:
        # Currently only support insertion of sequences
        # with same number of KVs across all heads/layers
        # (ie no swap-in of swapped-out, compressed sequences).
        assert len(self.free_batch_slots) > 0
        # If we just filled a block, update the block count to add tokens
        # that will be generated during the next decoding step.
        seq_block_count = (seq_len + self.block_size - 1) // self.block_size
        total_blocks = self.num_layers * self.num_kv_heads * seq_block_count
        # self._validate_allocator()
        block_numbers = self.gpu_allocator.allocate(total_blocks)
        # print(f"PREFILL ALLOCATION: {block_numbers.numel()} blocks")
        seq_blocks = block_numbers.reshape(
            self.num_layers, self.num_kv_heads, seq_block_count
        ).type(torch.int).to(self.device)
        batch_slot_idx = self.free_batch_slots.pop()
        self.batch_slot_mapping[seq_id] = batch_slot_idx
        self.block_state.context_lens[:,batch_slot_idx] = seq_len
        # TODO debug
        self.block_state.block_tables[:,batch_slot_idx,:,:seq_block_count] = seq_blocks
        # self.block_state._validate()
        # self._validate_allocator()

        # Add metric metadata associated with the newly added sequence blocks
        block_state_view = self.block_state.get_block_state_seq_view(batch_slot_idx)
        metadata = block_state_view.get_allocated_block_metadata()
        self.kv_metrics.insert_metadata(metadata)

    def _remove_sequence(self, seq_id: int) -> torch.Tensor:
        batch_slot_idx = self.batch_slot_mapping.pop(seq_id)
        self.free_batch_slots.add(batch_slot_idx)
        freed_blocks = (
            self.block_state
                .get_block_state_seq_view(batch_slot_idx)
                .get_allocated_blocks()
                .type(torch.int64)
                .to(self.gpu_allocator.device)
        )
        self.gpu_allocator.free(freed_blocks)
        self.block_state.context_lens[:,batch_slot_idx] = 0
        assert seq_id not in self.batch_slot_mapping
        # self.block_state._validate()

        return freed_blocks

    def _remove_sequence_batch(self, seqs: List[Sequence]) -> Optional[torch.Tensor]:
        batch_slot_idxs = []
        for seq in seqs:
            if seq.seq_id in self.batch_slot_mapping:
                batch_slot_idx = self.batch_slot_mapping.pop(seq.seq_id)
                batch_slot_idxs.append(batch_slot_idx)
                self.free_batch_slots.add(batch_slot_idx)
        if not batch_slot_idxs:
            return
        freed_blocks = (
            self.block_state
                .get_block_state_batch_view(batch_slot_idxs)
                .get_allocated_blocks()
                .type(torch.long)
                .to(self.gpu_allocator.device)
        )
        self.gpu_allocator.free(freed_blocks)
        self.block_state.context_lens[:,batch_slot_idxs] = 0
        return freed_blocks

    def _get_new_block_count(self, seq_id: int, token_count: int) -> int:
        assert token_count == 1
        batch_slot_idx = self.batch_slot_mapping[seq_id]
        new_kv_count = (
            self.block_state.context_lens[:,batch_slot_idx] + token_count
        )
        return int((new_kv_count % self.block_size == 1).sum())

    def _append_to_sequence_batch(self, seqs: List[Sequence], token_count: int = 1):
        batch_slots_idxs = torch.tensor(
            [self.batch_slot_mapping[seq.seq_id] for seq in seqs],
            dtype=torch.long,
            device=self.device,
        )
        block_state_view = self.block_state.get_block_state_batch_view(batch_slots_idxs)
        old_mask = block_state_view.allocated_block_mask()
        self.block_state.context_lens[:,batch_slots_idxs] += token_count
        new_mask = block_state_view.allocated_block_mask()
        new_mask = (new_mask & ~old_mask)
        new_blocks = new_mask.sum()

        if new_blocks > 0:
            # print(f"DECODE ALLOCATION: {new_blocks} blocks")
            tmp = self.block_state.block_tables[:,batch_slots_idxs]
            newly_allocated = (
                self.gpu_allocator.allocate(new_blocks).to(self.device).type(torch.int)
            )
            tmp[new_mask] = newly_allocated
            self.block_state.block_tables[:,batch_slots_idxs] = tmp
            metadata, _ = (
                self.block_state.get_block_state_batch_view(batch_slots_idxs)
                                .get_batch_new_block_metadata([seq.data.get_len() - 1 for seq in seqs])
            )
            self.kv_metrics.insert_metadata(metadata)
            # for seq in seqs:
            #     seq_pos = seq.data.get_len() - 1
            #     seq_index = self.batch_slot_mapping[seq.seq_id]
            #     mask_ = self.kv_metrics.seq_index_by_block == seq_index
            #     non_evictable = self.kv_metrics.token_positions[mask_] > seq_pos - 50
            #     non_evictable_ = self.kv_metrics.token_positions[mask_] > seq_pos - 50
            #     print(f"{non_evictable_.sum()=}")
            #     if non_evictable_.sum() < min(50, seq_pos) * 32 * 32:
            #         print(f"{non_evictable_.sum()=} {seq_pos * 32 * 32=}")
            #         max_ = self.kv_metrics.token_positions[mask_].max()
            #         print(f"{max_=}")
            #         print(f"{(self.kv_metrics.token_positions[mask_] == max_).sum()=}")
            #         raise Exception(seq_index)

    def _append_to_sequence(self, seq: Sequence, token_count: int):
        seq_id = seq.seq_id
        batch_slot_idx = self.batch_slot_mapping[seq_id]

        block_state_view = self.block_state.get_block_state_seq_view(batch_slot_idx)
        old_mask = block_state_view.allocated_block_mask()
        self.block_state.context_lens[:,batch_slot_idx] += token_count
        new_mask = block_state_view.allocated_block_mask()
        new_mask = (new_mask & ~old_mask)
        new_blocks = new_mask.sum()
        if new_blocks > 0:
            self.block_state.block_tables[:,batch_slot_idx][new_mask] = (self.gpu_allocator
                                                        .allocate(new_blocks)
                                                        .to(self.device)
                                                        .type(torch.int))

            # Add metric metadata associated with the newly allocated blocks
            metadata, mask = block_state_view.get_new_block_metadata(
                last_token_position=seq.data.get_len() - 1
            )
            self.kv_metrics.insert_metadata(metadata)

            # # assert parity between new_mask and mask computed in get_new_block_metadata
            # assert (new_mask == mask[:,0]).all()

    def get_batch_slot_index(self, seq: Sequence) -> int:
        return self.batch_slot_mapping[seq.seq_id]

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

    def get_sequence_block_count(self, seq: Sequence) -> int:
        batch_slot_idx = self.batch_slot_mapping[seq.seq_id]
        kv_counts = self.block_state.context_lens[:,batch_slot_idx]
        return int(((kv_counts + self.block_size - 1) // self.block_size).sum())

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
        # print(f"Checking PREFILL allocation: {num_required_blocks} blocks")

        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()

        # Use watermark to avoid frequent cache eviction.
        if (self.num_total_gpu_blocks - num_required_blocks <
                self.watermark_blocks):
            print(f"{self.num_total_gpu_blocks=}, {num_required_blocks=}, {self.num_total_gpu_blocks - num_required_blocks=}")
            raise Exception("Input length too long!")
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

    @BENCHMARKER.wrap()
    def can_append_slots(self,
                         seq_group: SequenceGroup,
                         num_lookahead_slots: int = 0,
                         num_free_blocks: Optional[int] = None) -> bool:
        assert (num_lookahead_slots == 0
                ), "lookahead allocation with KV-Compress not supported"
        assert (seq_group.num_seqs() <= 1,
                ), "multi-sequence SequenceGroups with KV-Compress are not supported"
        seq = seq_group.get_seqs(status=SequenceStatus.RUNNING)[0]

        # Simple heuristic: If there are at least num_kv_heads * num_layers free blocks
        # for each sequence, we can append.
        num_free_gpu_blocks = (self.gpu_allocator.get_num_free_blocks()
                               if num_free_blocks is None else num_free_blocks)
        new_block_count = self._get_new_block_count(seq_id=seq.seq_id, token_count=1)
        return new_block_count <= num_free_gpu_blocks

    def get_new_block_count(self, seq_group: SequenceGroup) -> int:
        seq = seq_group.get_seqs(status=SequenceStatus.RUNNING)[0]
        return self._get_new_block_count(seq_id=seq.seq_id, token_count=1)

    @BENCHMARKER.wrap()
    def batch_append_slots(self, seqs: List[Sequence]) -> None:
        self._append_to_sequence_batch(seqs)

    @BENCHMARKER.wrap()
    def append_slots(
        self,
        seq: Sequence,
        num_lookahead_slots: int = 0,
    ) -> Dict[int, List[int]]:
        """Allocate a physical slot for a new token."""
        self._append_to_sequence(seq=seq, token_count=1)
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

    @BENCHMARKER.wrap()
    def free(self, seq: Sequence) -> Optional[torch.Tensor]:
        """Returns torch.Tensor of freed blocks so that corresponding metadata slots
        can be de-allocated from KV metrics metadata (seq idx set to -1).
        """
        # print("freeing sequence")
        if seq.seq_id not in self.batch_slot_mapping:
            # Already freed or haven't been scheduled yet.
            return
        return self._remove_sequence(seq.seq_id)

    @BENCHMARKER.wrap()
    def free_batch(self, seqs: List[Sequence]) -> Optional[torch.Tensor]:
        return self._remove_sequence_batch(seqs)

    @BENCHMARKER.wrap()
    def free_compressed_blocks(self, freed_block_count: FreedBlockCounts) -> torch.Tensor:
        # self._validate_allocator()
        # self.block_state._validate()
        seq_indices, removed_blocks = zip(*[
            (self.batch_slot_mapping[seq_id], freed_block_count[seq_id])
            for seq_id in freed_block_count.keys()
        ])
        batch_view = self.block_state.get_block_state_batch_view(seq_indices)
        # Free blocks before updating context lengths
        freed_block_counts_tensor = torch.stack(removed_blocks, dim=1)
        freed_blocks, evicted_mask = batch_view.get_last_n_allocated_blocks(freed_block_counts_tensor)

        # assert not self.gpu_allocator.free_mask[freed_blocks].any(), "returned blocks should not be free yet"

        # freed_block_debug = None
        # if freed_blocks.numel() > 0 and freed_blocks[0] == 130:
        #     print('entered')
        #     freed_block_debug = torch.where(self.block_state.block_tables == freed_blocks[0])[:3]
        #     print(freed_block_debug)
        #     print(freed_block_counts_tensor[freed_block_debug])
        #     assert freed_block_counts_tensor[freed_block_debug] > 0
        # evicted_indices = torch.where(evicted_mask)
        self.gpu_allocator.free(
            freed_blocks
        )

        # print(f"start_where: {torch.where(self.block_state.block_tables == freed_blocks[0])}")

        # self.block_state._validate()
        # In case of empty last block, move the empty block to the new final
        # logical block position after evicting non-empty blocks.
        # if freed_block_debug is None:
        #     empty_blocks = batch_view.move_empty_trailing_blocks(freed_block_counts_tensor)
        # Update context lengths
        # init_context_lens = self.block_state.context_lens.clone()
        # self.block_state._validate()

        # start_mask = self.block_state.get_block_state_full_view().allocated_block_mask()
        # print(f"start_idxs: {torch.where(start_mask[freed_block_debug])}")
        seq_indices = torch.tensor(seq_indices, dtype=torch.long, device=self.device)
        self.block_state.remove_trailing_blocks(
            seq_indices=seq_indices,
            removed_block_count=removed_blocks,
            debug_freed_idx=evicted_mask,
        )
        # end_mask = self.block_state.get_block_state_full_view().allocated_block_mask()
        # print(f"end_idxs: {torch.where(end_mask[freed_block_debug])}")
        # print(f"end_where: {torch.where(self.block_state.block_tables == freed_blocks[0])}")

        # print(f"{freed_block_debug=}")
        # print(f"{removed_blocks[0][0,14]}")
        # print(evicted_mask.shape)
        # assert not (self.block_state.get_allocated_blocks() == freed_blocks[0]).any()

        # self.block_state._validate()
        # print(f'{evicted_indices=}')
        # print(f'freed: {freed_blocks[:5]} ({freed_blocks.shape[0]} total)')
        # if freed_blocks.numel() > 0:
        #     print(freed_block_debug)
        #     print(self.block_state.context_lens[freed_block_debug])
        #     print(init_context_lens[freed_block_debug])
        # print(f'{seq_indices=}')
        # self._validate_allocator()

        return freed_blocks

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

    def mark_blocks_as_computed(self, seq_group: SequenceGroup, chunk_size: int):
        pass

    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        """Prefix cache hit rate. -1 means not supported or disabled."""
        return -1

    def checkpoint(self) -> None:
        self.block_state.checkpoint()
        self.gpu_allocator.checkpoint()
