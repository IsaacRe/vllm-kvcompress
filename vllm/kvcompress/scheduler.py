from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import torch
import math

from vllm.debug import CHECKPOINTER
from vllm._custom_ops import schedule_cache_evictions, schedule_cache_moves
from vllm.config import KVCompressConfig
from vllm.sequence import Sequence
from vllm.kvcompress.block_manager import BlockSpaceManagerKVC, FreedBlockCounts
from vllm.kvcompress.metrics import CompressionMetrics
from vllm.benchmark import BENCHMARKER

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
    offsets: torch.Tensor


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
        self.control_layers = (
            torch.tensor(self.config.control_layers,
                         device=self.device,
                         dtype=torch.int)
            if self.config.control_layers else None
        )
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
        max_cache_tokens: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Return the number of this sequence's blocks to be freed during
        the next compression iteration.
        """
        if compression_rate is None:
            compression_rate = self.config.target_compression_rate
        if max_cache_tokens is None:
            max_cache_tokens = self.config.max_cache_tokens

        if compression_rate < 1.0 and max_cache_tokens > 0:
            raise RuntimeError("both compression_rate and max_cache_tokens "
                               "specified during compression")

        if max_cache_tokens > 0:
            # Evict by max number of KV per sequence.
            max_cache_kv = (
                max_cache_tokens * self.config.num_layers * self.config.num_kv_heads
            )
            evict_kv_count = max(
                0,
                self.block_manager.get_sequence_kv_count(seq) - max_cache_kv,
            )
        else:
            # Evict by target compression rate.
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

        # make sure # block evictions is evenly divisible by num_layers if even_layer_evict
        if self.config.even_layer_evict:
            evict_block_count = ((evict_block_count // self.config.num_layers)
                                 * self.config.num_layers)

        evict_kv_count = evict_block_count * self.block_size
        return evict_kv_count, evict_block_count
    
    def _schedule_compression(self, seqs: List[Sequence]) -> Optional[CompressionOutputs]:
        self._update_sequences(seqs)

        # Select sequences to compress this iteration and determine blocks to
        # evict.
        total_kv_count = 0
        seqs_to_compress: List[Sequence] = []
        evicted_blocks_per_seq = []
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

            seqs_to_compress.append(seq)
            evicted_blocks_per_seq.append(evicted_block_count)
            self._iters_since_compression[seq] = 0

        if not seqs_to_compress:
            return

        # Checkpoint
        if CHECKPOINTER.do_checkpoint:
            self.block_manager.checkpoint()
            self.compression_metrics.checkpoint()

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

        CHECKPOINTER.checkpoint('schedule_compression__evicted_blocks_per_seq', evicted_blocks_per_seq)
        CHECKPOINTER.checkpoint('schedule_compression__slot_indices', torch.tensor(slot_indices))
        CHECKPOINTER.checkpoint('schedule_compression__seq_lens', torch.tensor(seq_lens))

        last_token_positions = torch.tensor(
            seq_lens, dtype=torch.int, device=self.device) - 1

        # Sort compression metrics
        # Should not have begun handling requests
        # init_mem = torch.cuda.max_memory_allocated(
        #     torch.device(self.device))
        sort_output = self.compression_metrics.sort_seq_metrics(
            slot_indices, last_token_positions)
        # final_mem = torch.cuda.max_memory_allocated(
        #     torch.device(self.device))
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

        CHECKPOINTER.checkpoint('schedule_compression__context_lens', context_lens)
        CHECKPOINTER.checkpoint('schedule_compression__block_tables', block_tables)
        CHECKPOINTER.checkpoint('schedule_compression__hanging_token_count', hanging_token_count)

        # Schedule evictions
        evicted_kv_indices = torch.empty(
            (self.config.max_kv_per_compression,),
            dtype=torch.int,
            device=self.device,
        )
        evicted_logical_indices = torch.empty_like(evicted_kv_indices)
        evicted_kv_count = torch.empty(
            b_l_h,
            dtype=torch.int,
            device=self.device,
        )
        evicted_kv_offsets = torch.empty_like(evicted_kv_count)
        schedule_cache_evictions(
            evicted_kv_indices,
            evicted_logical_indices,
            evicted_kv_count,
            evicted_kv_offsets,
            sort_output.sorted_indices,
            sort_output.seq_block_offsets,
            sort_output.seq_block_offsets * self.block_size,
            sort_output.layer_by_block,
            sort_output.head_by_block,
            sort_output.logical_block_num_by_block,
            evicted_blocks_per_seq,
            context_lens,
            hanging_token_count,
            sort_output.token_positions,
            last_token_positions,
            self.block_size,
            self.config.protected_window_size,
            self.config.max_kv_per_compression,
            MAX_INT,
            True,
            self.config.even_layer_evict,
            self.control_layers,
        )
        if self.config.even_layer_evict:
            raise NotImplementedError("need to implement for flat evicted_kv_indices")
            # debug
            layerwise_eviction_sums = evicted_kv_count.sum(dim=-1)
            if self.control_layers is not None:
                non_control_mask = torch.ones(layerwise_eviction_sums.size(1), dtype=torch.bool, device=self.device)
                non_control_mask[self.control_layers.type(torch.int64)] = False
                layerwise_eviction_sums = layerwise_eviction_sums[:,non_control_mask]
            # assert (layerwise_eviction_sums[0,:1] == layerwise_eviction_sums[0]).all()
            # print("PASSED EVEN LAYER ASSERTION")

        CHECKPOINTER.checkpoint('schedule_compression__evicted_kv_indices', evicted_kv_indices)
        CHECKPOINTER.checkpoint('schedule_compression__evicted_kv_count', evicted_kv_count)

        # # Truncate eviction counts to last full evicted block
        # no_eviction = evicted_kv_count < hanging_token_count
        # evicted_kv_count = torch.where(
        #     no_eviction,
        #     torch.zeros_like(evicted_kv_count),
        #     evicted_kv_count - (evicted_kv_count - hanging_token_count)
        #     % self.block_size,
        # )
        # evicted_block_count = (evicted_kv_count + self.block_size - 1) // self.block_size
        # non_evicted_mask = (
        #     torch.arange(evicted_kv_indices.shape[-1])[None,None,None]
        #          .to(evicted_kv_count.device)
        #     >= evicted_kv_count[...,None]
        # )
        evicted_block_count = evicted_kv_count // self.block_size
        assert (evicted_block_count * self.block_size == evicted_kv_count).all()

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

        # # Set non-evicted slots to inf so that they are last after sort
        # evicted_kv_indices[non_evicted_mask.expand_as(evicted_kv_indices)] = MAX_INT

        # Sort evicted indices
        logical_index_sort = evicted_logical_indices.type(torch.long).sort(dim=0).indices

        seq_layer_head_logical_index_sort = (
            evicted_kv_indices[logical_index_sort].sort(dim=0).indices
        )
        evicted_logical_indices = evicted_logical_indices[seq_layer_head_logical_index_sort]

        CHECKPOINTER.checkpoint('schedule_compression__evicted_kv_indices_sorted', evicted_kv_indices)
        CHECKPOINTER.checkpoint('schedule_compression__evicted_kv_count_truncated', evicted_kv_count)

        # Schedule cache moves
        cache_moves_indices = torch.empty(
            (self.config.max_kv_per_compression // 2, 2),
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
            evicted_logical_indices,
            evicted_kv_count,
            evicted_kv_offsets,
            block_tables,
            context_lens,
            self.block_size,
        )
        cache_moves = CacheMoves(cache_moves_indices, cache_moves_count, evicted_kv_offsets)
        
        freed_block_count = {
            seq.seq_id: freed_blocks
            for seq, freed_blocks in zip(seqs_to_compress, evicted_block_count)
        }

        CHECKPOINTER.checkpoint('schedule_compression__cache_moves_indices', cache_moves_indices)
        CHECKPOINTER.checkpoint('schedule_compression__cache_moves_count', cache_moves_count)
        CHECKPOINTER.checkpoint('schedule_compression__freed_block_count', evicted_block_count)

        self._increment_iters_since_compression()

        # Free blocks that were removed by compression
        with BENCHMARKER.time("free_compressed_blocks"):
            freed_blocks = self.block_manager.free_compressed_blocks(freed_block_count)

        self.compression_metrics.remove_metadata(freed_blocks)

        return CompressionOutputs(cache_moves, freed_block_count)

    def schedule_compression(self, seqs: List[Sequence]) -> Optional[CompressionOutputs]:
        """Returns number of KV evictions per sequence"""
        if (self.config.target_compression_rate == 1.0 and
            self.config.max_cache_tokens <= 0):
            # No compression
            return
        self.iteration_count += 1
        if self.iteration_count >= self.config.compression_interval:
            self.iteration_count = 0
            return self._schedule_compression(seqs)
