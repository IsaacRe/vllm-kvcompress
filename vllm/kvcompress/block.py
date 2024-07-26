"""Token blocks."""
from typing import Any, List, Tuple, Dict, Optional, NamedTuple, Union
import torch
import math
import numpy as np

from vllm.utils import Device
from vllm.debug import CHECKPOINTER
from vllm.benchmark import BENCHMARKER


class BlockMetadata(NamedTuple):
    physical_blocks: torch.Tensor
    logical_blocks: torch.Tensor
    seq_indices: torch.Tensor
    layer_indices: torch.Tensor
    head_indices: torch.Tensor
    token_positions: torch.Tensor

    @staticmethod
    def concat(metadata_list: List["BlockMetadata"]) -> "BlockMetadata":
        return BlockMetadata(*(torch.cat(tensors) for tensors in zip(*metadata_list)))

    def validate_even_layer_evict(self):
        num_layers = self.layer_indices.max().item() + 1
        
        # should have same number of allocated KVs per layer
        # randomly sample to make check less time-consuming
        check_idx = np.random.randint(num_layers)
        check_idx_count = (self.layer_indices == check_idx).sum()
        per_layer_count = self.layer_indices.numel() / num_layers
        assert check_idx_count == per_layer_count, f'{check_idx_count=}, {per_layer_count=} (are you running even-layer eviction?)'


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
        self.block_table_indices = None
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
        self.block_table_indices = (
            torch.arange(self.block_tables.size(-1))[None,None,None].to("cuda:0")
        )
            
    def _validate(self) -> None:
        seq_ids = range(self.context_lens.size(1))
        full_batch_view = self.get_block_state_batch_view(seq_ids)
        all_allocated_blocks = full_batch_view.get_allocated_blocks()
        assert all_allocated_blocks.unique().numel() == all_allocated_blocks.numel(), "invalid block_state"

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
            all_logical_block_nums=self.block_table_indices,
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
    
    @BENCHMARKER.wrap()
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
        # self._validate()
        removed_kv_count = torch.stack(removed_block_count, dim=1) * self.block_size
        # init_ctx = self.context_lens.clone()
        batch_view = self.get_block_state_batch_view(seq_indices)
        hanging_token_count = batch_view.get_hanging_token_counts()
        remaining_slots = self.block_size - hanging_token_count
        # pre_mask = batch_view.allocated_block_mask()
        # batch_view.context_lens -= (removed_kv_count
        #                             - remaining_slots)

        # assert ((removed_kv_count - remaining_slots)[removed_kv_count > 0] >= 0).all()
        removed = torch.clamp(
            removed_kv_count - remaining_slots, min=0
        )
        # ctx = self.context_lens[:,seq_indices].clone()
        self.context_lens[:,seq_indices] -= removed
        # mask = self.context_lens[:,seq_indices] < 0
        # print(f"{removed_kv_count[mask]=}")
        # print(f"{remaining_slots[mask]=}")
        # print(f"{removed[mask]=}")
        # print(f"{ctx[mask]=}")
        # assert (self.context_lens >= 0).all(), self.context_lens.min()

        # if debug_freed_idx is not None:
        #     post_mask = self.get_block_state_batch_view(seq_indices).allocated_block_mask()
        #     removed_mask = pre_mask & ~post_mask
        #     print(torch.where(removed_mask))
        #     print(torch.where(debug_freed_idx))
        #     assert (debug_freed_idx == removed_mask).all()
        #     # should not add any allocated blocks
        #     assert not (post_mask & ~pre_mask).any()

        #     print(debug_freed_idx)
        #     print(torch.stack(removed_block_count, dim=1)[debug_freed_idx])
        #     assert (init_ctx != batch_view.context_lens)[debug_freed_idx]
        #     assert (init_ctx != self.context_lens)[debug_freed_idx]
        # self._validate()

    def checkpoint(self) -> None:
        # Save context lens and block tables
        CHECKPOINTER.checkpoint('block_state__context_lens', self.context_lens)
        CHECKPOINTER.checkpoint('block_state__block_tables', self.block_tables)


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
        all_logical_block_nums: Optional[torch.Tensor] = None,
        all_seq_indices: Optional[torch.Tensor] = None,
    ) -> None:
        self.seq_indices = seq_indices
        self.is_batch_view = is_batch_view
        self.block_size = block_size
        self.use_tiered_block_tables = use_tiered_block_tables
        self.block_tables = block_tables
        self.t2_block_tables = t2_block_tables
        self.context_lens = context_lens
        self.all_logical_block_nums = (
            torch.arange(block_tables.size(-1))[None,None,None]
                 .to(block_tables.device)
            if all_logical_block_nums is None else all_logical_block_nums
        )
        self.all_seq_indices = (
            torch.arange(block_tables.size(1))[None,:,None,None]
                 .to(device=block_tables.device, dtype=torch.int)
            if all_seq_indices is None else all_seq_indices
        )

    def _squeeze_out(self, out: torch.Tensor) -> torch.Tensor:
        return out if self.is_batch_view else out[:,0]
    
    def get_prefill_slot_mapping(self) -> torch.Tensor:
        """Return the slot mapping for each KV head of each layer for every
        token position in the prefill sequence. Since the sequence hasn't
        been processed yet, we assume the same context length per head
        (ie. no compression yet).
        """
        assert not self.is_batch_view, "only called for single sequence view"
        ctx_lens = self.context_lens[:,self.seq_indices]  # [ num_layers, 1, num_kv_heads ]
        # assert ctx_lens.unique().numel() == 1, (
        #     "varying context lengths for prefill sequence")
        ctx_length = ctx_lens.view(-1)[0]
        # [num_layers, num_tokens, num_kv_heads]
        positions = (torch.arange(ctx_length,
                                  dtype=torch.int,
                                  device=ctx_lens.device)[None,:,None]
                          .expand(ctx_lens.size(0),
                                  ctx_length,
                                  ctx_lens.size(2)))
        logical_blocks = positions // self.block_size
        offsets = positions % self.block_size
        # assert logical_blocks.max() < self.block_tables.shape[-1]
        block_numbers = (                   # [ num_layers, num_tokens, num_kv_heads ]
            self.block_tables[:,self.seq_indices]
                .squeeze(1)
                .transpose(1, 2)
                .gather(dim=1, index=logical_blocks.type(torch.int64))
                .type(torch.int64)
        )
        return block_numbers * self.block_size + offsets
    
    def get_decode_slot_mapping(self) -> torch.Tensor:
        """Return the slot mapping for each KV head of each layer at the next
        position into the physical KV cache.
        """
        # [ num_layers, 1, num_kv_heads ]
        # assert self.context_lens[:,self.seq_indices].min() > 0, self.context_lens[:,self.seq_indices].min()
        next_pos = self.context_lens[:,self.seq_indices] - 1
        # assert next_pos.min() >= 0
        next_block = next_pos // self.block_size
        next_offset = next_pos % self.block_size
        # assert next_block.max() < self.block_tables.shape[-1] and next_block.min() >= 0
        block_numbers = (                   # [ num_layers, 1, num_kv_heads, 1 ]
            self.block_tables[:,self.seq_indices]
                .gather(dim=-1, index=next_block[...,None].type(torch.int64))
                .type(torch.int64)
        )
        return block_numbers[...,0] * self.block_size + next_offset

    def get_context_lens(self) -> torch.Tensor:
        return self.context_lens[:,self.seq_indices]
    
    def get_block_tables(self) -> torch.Tensor:
        return self.block_tables[:,self.seq_indices]
    
    @BENCHMARKER.wrap()
    def get_hanging_token_counts(self) -> torch.Tensor:
        """Returns the number of KVs in the final block"""
        remaining_kv = self.context_lens[:,self.seq_indices] % self.block_size
        # 0 if last block is not full else block_size
        full_last_block = (remaining_kv == 0).type(torch.int) * self.block_size
        return self._squeeze_out(torch.maximum(remaining_kv, full_last_block))
    
    @BENCHMARKER.wrap()
    def get_block_counts(self, increment_on_full: bool = False, squeeze: bool = True, benchmark: bool = False) -> torch.Tensor:
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
    
    @BENCHMARKER.wrap()
    def allocated_block_mask(self, squeeze: bool = True) -> torch.Tensor:
        """Returns a mask of allocated blocks per head. In the case of heads
        whose last block is full, this includes an additional empty block for
        KVs generated during the next iteration.
        """
        block_counts = self.get_block_counts(increment_on_full=False, squeeze=False)
        out = self.all_logical_block_nums < block_counts[...,None]
        return self._squeeze_out(out) if squeeze else out
    
    @BENCHMARKER.wrap()
    def last_n_allocated_block_mask(self, n: Union[int, torch.Tensor], squeeze: bool = True, benchmark: bool = False) -> torch.Tensor:
        """Returns a mask of the last n allocated blocks per head
        starting from the last block with at least one KV.
        """
        block_counts = self.get_block_counts(increment_on_full=False, squeeze=False, benchmark=benchmark)
        if isinstance(n, torch.Tensor):
            assert block_counts.shape == n.shape

        out = (
            (self.all_logical_block_nums < block_counts[...,None])
            & (self.all_logical_block_nums >= (block_counts - n)[...,None])
        )
        return self._squeeze_out(out) if squeeze else out

    @BENCHMARKER.wrap()
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
    
    @BENCHMARKER.wrap()
    def get_last_n_allocated_blocks(self, n: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = self.last_n_allocated_block_mask(n, squeeze=False, benchmark=True)
        return self.block_tables[:,self.seq_indices][mask], mask

    def get_allocated_block_metadata(self) -> BlockMetadata:
        """Returns block metadata for all allocated blocks.
        WARNING: currently assumes that token position is aligned with logical
        index and will cause issues if sequences in this view have been compressed
        already.
        """
        assert not self.is_batch_view
        mask = self.allocated_block_mask(squeeze=False)
        logical_blocks = (
            self.all_logical_block_nums
                .expand_as(self.block_tables)[:,self.seq_indices][mask]
        )
        # Assume alignment between logical index and token position
        token_positions = (
            logical_blocks[:,None] * self.block_size
            + torch.arange(self.block_size, dtype=torch.int, device=logical_blocks.device)[None]
        )
        # assert mask.sum() > 0
        physical_blocks = self.block_tables[:,self.seq_indices][mask]
        seq_indices = torch.tensor(
            self.seq_indices,
            dtype=torch.int,
            device=logical_blocks.device)
        seq_indices = seq_indices.expand_as(logical_blocks)
        layer_indices, _, head_indices, _ = torch.where(mask)
        return BlockMetadata(
            physical_blocks=physical_blocks,
            logical_blocks=logical_blocks,
            seq_indices=seq_indices,
            layer_indices=layer_indices.type(torch.int),
            head_indices=head_indices.type(torch.int),
            token_positions=token_positions.type(torch.int),
        )
    
    def get_new_block_metadata(self, last_token_position: int) -> Optional[BlockMetadata]:
        """Return block metadata for all blocks that were added
        during the last scheduling iteration.
        If there are no new blocks returns None.
        Uses last token position to infer contiguous token position for each KV in the block.
        WARNING: we assume that the first KV in the last block will always have a token
        position that aligns with its logical index, since we evict down to the last full
        block, so any hanging tokens must have been added after the last compression.
        """
        last_block_mask = self.last_n_allocated_block_mask(1, squeeze=False)

        assert not self.is_batch_view
        # num_heads = self.context_lens.size(2)
        # indices = self.block_table_indices.expand_as(self.block_tables)[:,self.seq_indices]
        # # divide w/o round so that equality is satisfied for one KV per block
        # last_block = ((self.context_lens[:,self.seq_indices] + self.block_size - 1)
        #         / self.block_size)[...,None].expand_as(indices)
        # print(f'{indices.shape=}, {last_block.shape=}')
        # mask = (
        #     indices
        #     == last_block
        # )

        first_token_mask = ((self.context_lens[:,self.seq_indices,:,None] - 1)
                            % self.block_size == 0)
        mask = last_block_mask & first_token_mask
        # assert not (mask & ~last_block_mask).any(), 'heyooo'

        if mask.sum() == 0:
            return None, None

        logical_blocks = (self.all_logical_block_nums
                              .expand_as(self.block_tables)[:,self.seq_indices][mask])
        token_positions = (
            torch.tensor([last_token_position], dtype=torch.int, device=logical_blocks.device)
                 .expand_as(logical_blocks)[:,None]
            + torch.arange(self.block_size, dtype=torch.int, device=logical_blocks.device)[None]
        )
        physical_blocks = self.block_tables[:,self.seq_indices][mask]
        seq_indices = torch.tensor(
            self.seq_indices,
            dtype=torch.int,
            device=logical_blocks.device)
        seq_indices = seq_indices.expand_as(logical_blocks)
        layer_indices, _, head_indices, _ = torch.where(mask)

        # # should have same number of allocated KVs per layer
        # # randomly sample to make check less time-consuming
        # check_idx = np.random.randint(num_layers)
        # check_idx_count = (layer_indices == check_idx).sum()
        # per_layer_count = layer_indices.numel() / num_layers
        # assert check_idx_count == per_layer_count, f'{check_idx_count=}, {per_layer_count=} (are you running even-layer eviction?)'

        # print(f'{physical_blocks=}, {physical_blocks.unique()=}')
        # assert physical_blocks.shape[0] == physical_blocks.unique(dim=0).shape[0]
        
        return BlockMetadata(
            physical_blocks=physical_blocks,
            logical_blocks=logical_blocks,
            seq_indices=seq_indices,
            layer_indices=layer_indices.type(torch.int),
            head_indices=head_indices.type(torch.int),
            token_positions=token_positions,
        ), mask

    def get_batch_new_block_metadata(self, last_token_position: List[int]) -> Optional[BlockMetadata]:
        """Return block metadata for all blocks that were added
        during the last scheduling iteration.
        If there are no new blocks returns None.
        Uses last token position to infer contiguous token position for each KV in the block.
        WARNING: we assume that the first KV in the last block will always have a token
        position that aligns with its logical index, since we evict down to the last full
        block, so any hanging tokens must have been added after the last compression.
        """
        last_block_mask = self.last_n_allocated_block_mask(1, squeeze=False)

        assert self.is_batch_view
        # num_heads = self.context_lens.size(2)
        # indices = self.block_table_indices.expand_as(self.block_tables)[:,self.seq_indices]
        # # divide w/o round so that equality is satisfied for one KV per block
        # last_block = ((self.context_lens[:,self.seq_indices] + self.block_size - 1)
        #         / self.block_size)[...,None].expand_as(indices)
        # print(f'{indices.shape=}, {last_block.shape=}')
        # mask = (
        #     indices
        #     == last_block
        # )

        seq_indices_tensor = (
            self.seq_indices if isinstance(self.seq_indices, torch.Tensor) else
            torch.tensor(
                self.seq_indices,
                dtype=torch.long,
                device=self.block_tables.device,
            )
        )
        first_token_mask = ((self.context_lens[:,seq_indices_tensor,:,None] - 1)
                            % self.block_size == 0)
        mask = last_block_mask & first_token_mask
        # assert not (mask & ~last_block_mask).any(), 'heyooo'

        if mask.sum() == 0:
            return None, None
        logical_blocks = (self.all_logical_block_nums
                              .expand_as(self.block_tables)[:,seq_indices_tensor][mask])
        physical_blocks = self.block_tables[:,seq_indices_tensor][mask]
        seq_indices = (self.all_seq_indices
                           .expand_as(self.block_tables)[:,seq_indices_tensor][mask])
        # Use seq_indices to extract the last_position for each block's corresponding sequence
        last_token_position = torch.tensor(
            last_token_position,
            dtype=torch.int,
            device=self.block_tables.device,
        )
        token_positions = (
            self.all_seq_indices.scatter(
                dim=1,
                index=seq_indices_tensor[None,:,None,None],
                src=last_token_position[None,:,None,None],
            ).expand_as(self.block_tables)[:,seq_indices_tensor][mask][:,None]
            + torch.arange(self.block_size, dtype=torch.int, device=logical_blocks.device)[None]
        )
        layer_indices, _, head_indices, _ = torch.where(mask)

        # # should have same number of allocated KVs per layer
        # # randomly sample to make check less time-consuming
        # check_idx = np.random.randint(num_layers)
        # check_idx_count = (layer_indices == check_idx).sum()
        # per_layer_count = layer_indices.numel() / num_layers
        # assert check_idx_count == per_layer_count, f'{check_idx_count=}, {per_layer_count=} (are you running even-layer eviction?)'

        # print(f'{physical_blocks=}, {physical_blocks.unique()=}')
        # assert physical_blocks.shape[0] == physical_blocks.unique(dim=0).shape[0]

        for seq_index, last_pos in zip(self.seq_indices, last_token_position):
            mask_ = seq_indices == seq_index
            assert (token_positions[mask_] == token_positions[mask_][0:1]).all()
            assert (token_positions[mask_][:,0] == last_pos).all(), (token_positions[mask_][:,0], last_pos)

        return BlockMetadata(
            physical_blocks=physical_blocks,
            logical_blocks=logical_blocks,
            seq_indices=seq_indices,
            layer_indices=layer_indices.type(torch.int),
            head_indices=head_indices.type(torch.int),
            token_positions=token_positions,
        ), mask
