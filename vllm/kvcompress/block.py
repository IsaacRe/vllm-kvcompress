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
            
    def clear(self):
        self._initialize()

    def get_block_state_batch_view(
        self, seq_indices: List[int]
    ) -> "BlockStateView":
        return BlockStateView(
            block_size=self.block_size,
            use_tiered_block_tables=self.use_tiered_block_tables,
            block_tables=self.block_tables[:,seq_indices],
            t2_block_tables=self.t2_block_tables,
            context_lens=self.context_lens[:,seq_indices],
        )
            
    def get_block_state_seq_view(self, seq_index: int) -> "BlockStateView":
        # [ num_layers, num_kv_heads, max_num_blocks_per_seq]   if single-tier
        # [ num_layers, max_num_blocks_per_seq]                 if two-tier
        return BlockStateView(
            block_size=self.block_size,
            use_tiered_block_tables=self.use_tiered_block_tables,
            block_tables=self.block_tables[:,seq_index],
            t2_block_tables=self.t2_block_tables,
            context_lens=self.context_lens[:,seq_index],
        )
    
    def get_last_non_empty_blocks(
        self, seq_indices: List[int]
    ) -> torch.Tensor:
        return (self.context_lens[:,seq_indices,:] - 1) // self.block_size
    
    def remove_trailing_blocks(
        self,
        seq_indices: List[int],
        removed_block_count: torch.Tensor,
    ) -> None:
        """Update context_lens to account for blocks that were
        removed due to KV cache compression.
        Called by block manager after compression scheduling.
        Block manager is in charge of freeing the removed blocks.
        """
        batch_view = self.get_block_state_batch_view(seq_indices)
        hanging_token_count = batch_view.get_hanging_token_counts()
        remaining_slots = self.block_size - hanging_token_count
        batch_view.context_lens -= (removed_block_count - remaining_slots)


class BlockStateView:
    """View of KVC block tables for a specific sequence"""

    def __init__(
        self,
        block_size: int,
        use_tiered_block_tables: bool,
        block_tables: torch.Tensor,
        t2_block_tables: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> None:
        self.is_batch_view = context_lens.dim() <= 2
        self.block_size = block_size
        self.use_tiered_block_tables = use_tiered_block_tables
        self.block_tables = block_tables
        self.t2_block_tables = t2_block_tables
        self.context_lens = context_lens
    
    def get_slot_mapping(self) -> torch.Tensor:
        """Return the slot mapping for each KV head of each layer at the next
        position into the physical KV cache.
        """
        assert not self.is_batch_view, "only called for single sequence view"
        next_pos = self.context_lens  # [ num_layers, num_kv_heads ]
        next_block = next_pos // self.block_size
        next_offset = next_pos % self.block_size
        assert next_block.max() < self.block_tables.shape[-1]
        if self.use_tiered_block_tables:
            t2_block_numbers = (                # [ num_layers, 1 ]
                self.block_tables
                    .gather(
                        dim=-1,
                        index=next_block.max(dim=-1).values[...,None])
                    .type(torch.int64)
            )
            block_numbers = (                   # [ num_layers, 1, num_kv_heads ]
                self.t2_block_tables
                    .gather(dim=1, index=t2_block_numbers[:,None,:])
                    .type(torch.int64)
            )
            return block_numbers.unsqueeze(1) * self.block_size + next_offset
        else:
            block_numbers = (                   # [ num_layers, num_kv_heads, 1 ]
                self.block_tables
                    .gather(dim=-1, index=next_block[...,None])
                    .type(torch.int64)
            )
            return block_numbers.unsqueeze(-1) * self.block_size + next_offset

    def get_context_lens(self) -> torch.Tensor:
        return self.context_lens
    
    def get_block_tables(self) -> torch.Tensor:
        return self.block_tables
    
    def get_hanging_token_counts(self) -> torch.Tensor:
        """Returns the number of KVs in the final block"""
        remaining_kv = self.context_lens % self.block_size
        # 0 if last block is not full else block_size
        full_last_block = (remaining_kv == 0).type(torch.int) * self.block_size
        return torch.maximum(remaining_kv, full_last_block)


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
    first_not_empty = next(view for view in views if view)
    num_layers, _, num_kv_heads, max_blocks = first_not_empty.shape
    return torch.stack(
        [
            view if view else
            _get_empty_block_tables(1, max_blocks, num_layers, num_kv_heads)
            for view in views
        ],
        dim=1,
    )
