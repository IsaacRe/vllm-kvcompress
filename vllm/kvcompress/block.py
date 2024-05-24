"""Token blocks."""
from typing import Any, List, Tuple, Dict, Optional
import torch
import math

from vllm.utils import Device


class PhysicalTokenBlock:
    """Represents the state of a block in the KV cache.
    """

    def __init__(
        self,
        device: Device,
        block_number: int,
        block_size: int,
        num_inserted_tokens: int,
    ) -> None:
        self.device = device
        self.block_number = block_number
        self.block_size = block_size
        self.num_inserted_tokens = num_inserted_tokens

        self.ref_count = 0

        self.computed = False

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "ref_count" and value > 1:
            raise ValueError("Tried to add more than 1 reference to physical block."
                             " This is disallowed when using KV-Compress.")
        super().__setattr__(name, value)

    def __repr__(self) -> str:
        return (f'PhysicalTokenBlock(device={self.device}, '
                f'block_number={self.block_number}, '
                f'num_inserted_tokens={self.num_inserted_tokens}, '
                f'ref_count={self.ref_count}, '
                f'computed={self.computed})')
    

class BlockState:

    def __init__(
        self,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        max_num_seqs: int,
        max_num_blocks_per_head: int,
        max_num_t1_blocks: int,
        use_tiered_block_tables: bool,
    ) -> None:
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.max_num_seqs = max_num_seqs
        self.max_num_blocks_per_head = max_num_blocks_per_head
        self.max_num_t1_blocks = max_num_t1_blocks
        self.use_tiered_block_tables = use_tiered_block_tables
        self.block_tables = None
        self.t2_block_tables = None
        self.context_lens = None
        self._initialize()

    def _initialize(self):
        print(f"Allocating context_lens - Mem: {torch.cuda.memory_allocated(0) * 1e-9}")
        self.context_lens = torch.zeros(
            (self.num_layers,
             self.max_num_seqs,
             self.num_kv_heads),
            dtype=torch.int,
            device="cuda:0")
        if self.use_tiered_block_tables:
            self.block_tables = torch.empty(
                (self.num_layers,
                 self.max_num_seqs,
                 self.max_num_blocks_per_head),
                dtype=torch.int,
                device="cuda:0")
            self.t2_block_tables = torch.empty(
                (self.num_layers,
                 self.max_num_t1_blocks,
                 self.num_kv_heads),
                dtype=torch.int,
                device="cuda:0")
        else:
            print(f"Allocating block table - Mem: {torch.cuda.memory_allocated(0) * 1e-9}")
            self.block_tables = torch.empty(
                (self.num_layers,
                 self.max_num_seqs,
                 self.num_kv_heads,
                 self.max_num_blocks_per_head),
                dtype=torch.int,
                device="cuda:0")
            
    def _validate(self) -> None:
        seq_ids = range(self.context_lens.size(1))
        full_batch_view = self.get_block_state_batch_view(seq_ids)
        all_allocated_blocks = full_batch_view.get_allocated_blocks()
        assert all_allocated_blocks.unique().numel() == all_allocated_blocks.numel()

    def clear(self) -> None:
        self._initialize()

    def get_allocated_blocks(self) -> torch.Tensor:
        return self.get_block_state_full_view().get_allocated_blocks()
            
    def get_block_state_full_view(self) -> "BlockStateView":
        return BlockStateView(
            block_size=self.block_size,
            seq_indices=list(range(self.block_tables.size(1))),
            use_tiered_block_tables=self.use_tiered_block_tables,
            block_tables=self.block_tables,
            t2_block_tables=self.t2_block_tables,
            context_lens=self.context_lens,
            is_batch_view=True,
        )

    def get_block_state_batch_view(
        self, seq_indices: List[int]
    ) -> "BlockStateView":
        return BlockStateView(
            block_size=self.block_size,
            seq_indices=seq_indices,
            use_tiered_block_tables=self.use_tiered_block_tables,
            block_tables=self.block_tables,
            t2_block_tables=self.t2_block_tables,
            context_lens=self.context_lens,
            is_batch_view=True,
        )
            
    def get_block_state_seq_view(self, seq_index: int) -> "BlockStateView":
        # [ num_layers, num_kv_heads, max_num_blocks_per_seq]   if single-tier
        # [ num_layers, max_num_blocks_per_seq]                 if two-tier
        return BlockStateView(
            block_size=self.block_size,
            seq_indices=[seq_index],
            use_tiered_block_tables=self.use_tiered_block_tables,
            block_tables=self.block_tables,
            t2_block_tables=self.t2_block_tables,
            context_lens=self.context_lens,
            is_batch_view=False,
        )
    
    def get_last_non_empty_blocks(
        self, seq_indices: List[int]
    ) -> torch.Tensor:
        return (self.context_lens[:,seq_indices,:] - 1) // self.block_size
    
    def remove_trailing_blocks(
        self,
        seq_indices: List[int],
        removed_block_count: List[torch.Tensor],
        debug_freed_idx = None,
    ) -> None:
        """Update context_lens to account for blocks that were
        removed due to KV cache compression.
        Called by block manager after compression scheduling.
        Block manager is in charge of freeing the removed blocks.
        """
        self._validate()
        removed_kv_count = torch.stack(removed_block_count, dim=1) * self.block_size
        init_ctx = self.context_lens.clone()
        batch_view = self.get_block_state_batch_view(seq_indices)
        hanging_token_count = batch_view.get_hanging_token_counts()
        remaining_slots = self.block_size - hanging_token_count
        pre_mask = batch_view.allocated_block_mask()
        # batch_view.context_lens -= (removed_kv_count
        #                             - remaining_slots)

        self.context_lens[:,seq_indices] -= (removed_kv_count
                                             - remaining_slots)

        # TODO here try data ptr
        if debug_freed_idx is not None:
            post_mask = self.get_block_state_batch_view(seq_indices).allocated_block_mask()
            removed_mask = pre_mask & ~post_mask
            print(torch.where(removed_mask))
            print(torch.where(debug_freed_idx))
            assert (debug_freed_idx == removed_mask).all()

            print(debug_freed_idx)
            print(torch.stack(removed_block_count, dim=1)[debug_freed_idx])
            # assert (init_ctx != batch_view.context_lens)[debug_freed_idx]
            assert (init_ctx != self.context_lens)[debug_freed_idx]
        self._validate()


class BlockStateView:
    """View of KVC block tables for a specific sequence"""

    def __init__(
        self,
        block_size: int,
        seq_indices: List[int],
        use_tiered_block_tables: bool,
        block_tables: torch.Tensor,
        t2_block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        is_batch_view: bool,
    ) -> None:
        self.seq_indices = seq_indices
        self.is_batch_view = is_batch_view
        self.block_size = block_size
        self.use_tiered_block_tables = use_tiered_block_tables
        self.block_tables = block_tables
        self.t2_block_tables = t2_block_tables
        self.context_lens = context_lens
        self.block_table_indices = (
            torch.arange(block_tables.size(-1))[None,None,None]
                 .to(block_tables.device)
        )

    def _squeeze_out(self, out: torch.Tensor) -> torch.Tensor:
        return out if self.is_batch_view else out[:,0]
    
    def get_slot_mapping(self) -> torch.Tensor:
        """Return the slot mapping for each KV head of each layer at the next
        position into the physical KV cache.
        """
        assert not self.is_batch_view, "only called for single sequence view"
        next_pos = self.context_lens[:,self.seq_indices]  # [ num_layers, 1, num_kv_heads ]
        next_block = next_pos // self.block_size
        next_offset = next_pos % self.block_size
        assert next_block.max() < self.block_tables.shape[-1]
        block_numbers = (                   # [ num_layers, 1, num_kv_heads, 1 ]
            self.block_tables[:,self.seq_indices]
                .gather(dim=-1, index=next_block[...,None].type(torch.int64))
                .type(torch.int64)
        )
        return self._squeeze_out(block_numbers[...,0] * self.block_size + next_offset)

    def get_context_lens(self) -> torch.Tensor:
        return self._squeeze_out(self.context_lens[:,self.seq_indices])
    
    def get_block_tables(self) -> torch.Tensor:
        return self._squeeze_out(self.block_tables[:,self.seq_indices])
    
    def get_hanging_token_counts(self) -> torch.Tensor:
        """Returns the number of KVs in the final block"""
        remaining_kv = self.context_lens[:,self.seq_indices] % self.block_size
        # 0 if last block is not full else block_size
        full_last_block = (remaining_kv == 0).type(torch.int) * self.block_size
        return self._squeeze_out(torch.maximum(remaining_kv, full_last_block))
    
    def get_block_counts(self, increment_on_full: bool = False, squeeze: bool = True) -> torch.Tensor:
        """Returns count of non-empty blocks per head given the current state
        of context_lens.
        
        Args:
            increment_on_full: if set, increment the block count by one for heads
                whose last non-empty block is full.
        """
        context_lens = self.context_lens[:,self.seq_indices]
        counts = (
            (context_lens + self.block_size) // self.block_size
            if increment_on_full else
            (context_lens + self.block_size - 1) // self.block_size
        )
        empty_seq_mask = context_lens == 0
        counts[empty_seq_mask] = 0
        return self._squeeze_out(counts) if squeeze else counts
    
    def allocated_block_mask(self, squeeze: bool = True) -> torch.Tensor:
        """Returns a mask of allocated blocks per head. In the case of heads
        whose last block is full, this includes an additional empty block for
        KVs generated during the next iteration.
        """
        block_counts = self.get_block_counts(increment_on_full=True, squeeze=False)
        out = self.block_table_indices < block_counts[...,None]
        return self._squeeze_out(out) if squeeze else out
    
    def last_n_allocated_block_mask(self, n: torch.Tensor, squeeze: bool = True) -> torch.Tensor:
        """Returns a mask of the last n allocated blocks per head
        starting from the last block with at least one KV.
        """
        block_counts = self.get_block_counts(increment_on_full=False, squeeze=False)
        assert block_counts.shape == n.shape

        out = (
            (self.block_table_indices < block_counts[...,None])
            & (self.block_table_indices >= (block_counts - n)[...,None])
        )
        return self._squeeze_out(out) if squeeze else out

    def move_empty_trailing_blocks(self, n: torch.Tensor) -> None:
        """Moves the final block for each head back n positions ONLY if the
        block is empty.
        """
        last_is_empty = self.context_lens[:,self.seq_indices] % self.block_size == 0
        last_index = self.get_block_counts(increment_on_full=True, squeeze=False) - 1
        assert last_index.shape == n.shape

        # Mask heads with empty last blocks and move
        last_index = last_index[last_is_empty]
        n = n[last_is_empty]
        block_tables = self.block_tables[:,self.seq_indices]
        src = block_tables[last_is_empty].gather(
            dim=-1,
            index=last_index.type(torch.int64)[...,None],
        )
        block_tables[last_is_empty] = block_tables[last_is_empty].scatter(
            dim=-1,
            index=(last_index - n).type(torch.int64)[...,None],
            src=src,
        )
        self.block_tables[:,self.seq_indices] = block_tables
        return self._squeeze_out(src)

    def get_allocated_blocks(self) -> torch.Tensor:
        return self.block_tables[:,self.seq_indices][self.allocated_block_mask(squeeze=False)]
    
    def get_last_n_allocated_blocks(self, n: torch.Tensor) -> torch.Tensor:
        mask = self.last_n_allocated_block_mask(n, squeeze=False)
        return self.block_tables[:,self.seq_indices][mask], mask


def _get_empty_block_tables(
    batch_size: int,
    max_blocks: int,
    num_layers: int,
    num_kv_heads: int,
) -> torch.Tensor:
    # Currently only support single-tier
    return torch.zeros(
        (num_layers, batch_size, num_kv_heads, max_blocks),
        dtype=torch.int,
        device="cuda:0",
    )


def merge_block_table_views(
    views: List[Optional[BlockStateView]],
) -> torch.Tensor:
    first_not_empty = next(view for view in views if view is not None)
    num_layers, num_kv_heads, max_blocks = first_not_empty.shape
    return torch.stack(
        [
            view if view is not None else
            _get_empty_block_tables(1, max_blocks, num_layers, num_kv_heads)
            for view in views
        ],
        dim=1,
    )
