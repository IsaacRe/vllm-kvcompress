from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import torch
import math

from vllm._custom_ops import schedule_cache_evictions, schedule_cache_moves
from vllm.config import KVCompressConfig
from vllm.sequence import Sequence
from vllm.kvcompress.block_manager import BlockSpaceManagerKVC, FreedBlockCounts
from vllm.kvcompress.metrics import CompressionMetrics

MAX_INT = 2147483000


@dataclass
class CacheMoves:
    """Inputs to the execute_cache_moves kernel called after scheduling
    compression and before the model forward pass. Inputs are passed from
    the scheduler to the cache engine for execution.
    
    index: physical KV indices into cache for move source/destination.
    count: count of KV moves for each KV head.
    """

    index: torch.Tensor
    count: torch.Tensor


@dataclass
class CompressionOutputs:
    """This class defined the physical KV cache slot moves and block frees
    that must occur before all other scheduled cache actions and model-running.
    """

    cache_moves: CacheMoves
    freed_block_count: FreedBlockCounts


class CompressionScheduler:
    """Scheduler responsible for determining when to launch iterations
    of the KVCompressEngine. Lives in LLMEngine.scheduler.kvcompress_scheduler.
    """

    def __init__(
        self,
        config: KVCompressConfig,
        block_manager: BlockSpaceManagerKVC,
        compression_metrics: CompressionMetrics,
    ) -> None:
        self.device = block_manager.device
        self.block_size = block_manager.block_size
        self.config = config
        self.block_manager = block_manager
        self.compression_metrics = compression_metrics
        self.iteration_count = 0
        # Mapping: seq_id -> num iters
        self._iters_since_compression: Dict[int, int] = {}
        self.total_evicted_kvs = {}
        self.total_evicted_blocks = {}

    def _update_sequences(self, seqs: List[Sequence]) -> None:
        all_seq_ids = set([seq.seq_id for seq in seqs])
        for seq_id in list(self._iters_since_compression):
            if seq_id not in all_seq_ids:
                del self._iters_since_compression[seq_id]
        for seq_id in all_seq_ids:
            if seq_id not in self._iters_since_compression:
                self._iters_since_compression[seq_id] = 0
    
    def _increment_iters_since_compression(self) -> None:
        for seq_id in self._iters_since_compression:
            self._iters_since_compression[seq_id] += 1

    def _schedule_seq_evictions(
        self,
        seq: Sequence,
        compression_rate: Optional[float] = None,
    ) -> Tuple[int, int]:
        """Return the number of this sequence's blocks to be freed during
        the next compression iteration.
        """
        if compression_rate is None:
            compression_rate = self.config.target_compression_rate
        # Total KV count should not go below protected_window_size * num_kv_heads
        compressible_token_count = (
            seq.data.get_len() - self.config.protected_window_size
        )
        if compressible_token_count <= 0:
            return 0, 0
        # Total count of KVs in compressible range if this sequence had never
        # been compressed
        uncompressed_kv_count = (
            compressible_token_count
            * self.config.num_layers * self.config.num_kv_heads
        )
        # Actual count of KVs in compressible range
        compressed_kv_count = (
            self.block_manager.get_sequence_kv_count(seq)
            - self.config.protected_window_size * self.config.num_layers
            * self.config.num_kv_heads
        )
        # Target count of KVs in compressible range that will yield the
        # desired compression rate
        target_kv_count = math.ceil(uncompressed_kv_count * compression_rate)
        evict_kv_count = max(0, compressed_kv_count - target_kv_count)
        evict_block_count = evict_kv_count // self.block_size
        return evict_kv_count, evict_block_count
    
    def _schedule_compression(self, seqs: List[Sequence]) -> Optional[CompressionOutputs]:
        self._update_sequences(seqs)

        # Select sequences to compress this iteration and determine blocks to
        # evict.
        total_kv_count = 0
        seqs_to_compress: List[Sequence] = []
        evicted_blocks_per_seq = []
        max_evicted_tokens = 0
        for _, _, seq in sorted(
            [(self._iters_since_compression[s.seq_id], s.seq_id, s) for s in seqs],
            reverse=True,
        ):
            # If sequence is not long enough for compression, skip.
            evicted_kv_count, evicted_block_count = self._schedule_seq_evictions(seq)
            if evicted_block_count == 0:
                continue

            # Stop once we reach the maximum number of KVs to compress.
            total_kv_count += self.block_manager.get_sequence_kv_count(seq)
            if total_kv_count > self.config.max_kv_per_compression:
                break

            max_evicted_tokens = max(max_evicted_tokens, evicted_kv_count)

            seqs_to_compress.append(seq)
            evicted_blocks_per_seq.append(evicted_block_count)
            self._iters_since_compression[seq] = 0

        if not seqs_to_compress:
            return

        # Sort sequences by batch_slot_index index
        seqs_to_compress, evicted_blocks_per_seq = zip(*sorted(
            zip(seqs_to_compress, evicted_blocks_per_seq),
            key=lambda x: self.block_manager.get_slot_index(x[0]),
        ))
        seqs_to_compress = list(seqs_to_compress)
        evicted_blocks_per_seq = list(evicted_blocks_per_seq)

        batch_size = len(seqs_to_compress)
        b_l_h = batch_size, self.config.num_layers, self.config.num_kv_heads

        evicted_blocks_per_seq = torch.tensor(
            evicted_blocks_per_seq, dtype=torch.int, device=self.device
        )

        slot_indices = [self.block_manager.get_slot_index(seq) for seq in seqs_to_compress]
        seq_lens = [seq.data.get_len() for seq in seqs_to_compress]

        # Update bias for KVs being compressed based on their seq_len bin
        self.compression_metrics.update_bias_for_positions(slot_indices, seq_lens)

        # Sort compression metrics
        # Should not have begun handling requests
        init_mem = torch.cuda.max_memory_allocated(
            torch.device(self.device))
        sort_output = self.compression_metrics.sort_seq_metrics(slot_indices)
        final_mem = torch.cuda.max_memory_allocated(
            torch.device(self.device))
        # print(f"RAN SORT: {final_mem - init_mem}")

        # Get context lengths, block tables and hanging token counts
        batch_block_state = self.block_manager.get_block_state_batch_view(
            seqs_to_compress
        )
        context_lens = batch_block_state.get_context_lens().contiguous()
        block_tables = batch_block_state.get_block_tables().contiguous()
        hanging_token_count = (
            batch_block_state.get_hanging_token_counts()
                             .transpose(0, 1)
                             .contiguous()
        )

        # Schedule evictions
        evicted_kv_indices = torch.empty(
            (*b_l_h, max_evicted_tokens),
            dtype=torch.int,
            device=self.device,
        )
        evicted_kv_count = torch.empty(
            b_l_h,
            dtype=torch.int,
            device=self.device,
        )

        schedule_cache_evictions(
            evicted_kv_indices,
            evicted_kv_count,
            sort_output.sorted_indices,
            sort_output.seq_block_offsets,
            sort_output.layer_by_block,
            sort_output.head_by_block,
            sort_output.logical_block_num_by_block,
            evicted_blocks_per_seq,
            context_lens,
            hanging_token_count,
            self.block_size,
        )

        # Truncate eviction counts to last full evicted block
        no_eviction = evicted_kv_count < hanging_token_count
        evicted_kv_count = torch.where(
            no_eviction,
            torch.zeros_like(evicted_kv_count),
            evicted_kv_count - (evicted_kv_count - hanging_token_count)
            % self.block_size,
        )
        evicted_block_count = (evicted_kv_count + self.block_size - 1) // self.block_size
        non_evicted_mask = (
            torch.arange(evicted_kv_indices.shape[-1])[None,None,None]
                 .to(evicted_kv_count.device)
            >= evicted_kv_count[...,None]
        )

        # print("TOTAL EVICTED KVS:")
        # for i, seq in enumerate(seqs_to_compress):
        #     if seq.seq_id not in self.total_evicted_kvs:
        #         self.total_evicted_kvs[seq.seq_id] = 0
        #         self.total_evicted_blocks[seq.seq_id] = 0
        #     tot_evicted_pre = self.total_evicted_kvs[seq.seq_id]
        #     new_evictions = evicted_kv_count[i].sum().item()
        #     self.total_evicted_kvs[seq.seq_id] += new_evictions
        #     self.block_manager.total_seq_lens[seq.seq_id] = seq.data.get_len()
        #     tot_evicted = tot_evicted_pre + new_evictions
        #     tot_kv = self.block_manager.total_seq_lens[seq.seq_id] * 32 * 32
        #     tot_remaining = batch_block_state.context_lens[:,i].sum().item()
        #     self.total_evicted_blocks[seq.seq_id] += evicted_block_count[i].sum().item()
        #     tot_evicted_block = self.total_evicted_blocks[seq.seq_id]
        #     print(f"{tot_evicted=}")
        #     print(f"{tot_remaining}/{tot_kv}={tot_remaining/tot_kv * 100}% remaining")
        #     print(f"{tot_evicted_pre}/{tot_kv}={tot_evicted/tot_kv * 100}% previously evicted")
        #     print(f"{tot_evicted}/{tot_kv}={tot_evicted/tot_kv * 100}% now evicted")
        #     print(f"{tot_evicted_block * self.block_size}/{tot_kv}={tot_evicted_block * self.block_size / tot_kv * 100}% now evicted (block)")

        # Set non-evicted slots to inf so that they are last after sort
        evicted_kv_indices[non_evicted_mask.expand_as(evicted_kv_indices)] = MAX_INT

        # Sort evicted indices
        evicted_kv_indices = evicted_kv_indices.sort(dim=-1).values

        # Schedule cache moves
        cache_moves_indices = torch.empty(
            (*b_l_h, max_evicted_tokens, 2),
            dtype=torch.int64,
            device=self.device,
        )
        cache_moves_count = torch.empty(
            b_l_h,
            dtype=torch.int64,
            device=self.device,
        )
        schedule_cache_moves(
            cache_moves_indices,
            cache_moves_count,
            evicted_kv_indices,
            evicted_kv_count,
            block_tables,
            context_lens,
            self.block_size,
        )
        cache_moves = CacheMoves(cache_moves_indices, cache_moves_count)
        
        freed_block_count = {
            seq.seq_id: freed_blocks
            for seq, freed_blocks in zip(seqs_to_compress, evicted_block_count)
        }

        self._increment_iters_since_compression()
        return CompressionOutputs(cache_moves, freed_block_count)

    def schedule_compression(self, seqs: List[Sequence]) -> Optional[CompressionOutputs]:
        """Returns number of KV evictions per sequence"""
        if self.config.target_compression_rate == 1.0:
            # No compression
            return
        self.iteration_count += 1
        if self.iteration_count >= self.config.compression_interval:
            self.iteration_count = 0
            return self._schedule_compression(seqs)
