from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import torch
import math
from copy import deepcopy

from vllm.debug import CHECKPOINTER
from vllm._custom_ops import schedule_cache_evictions, schedule_cache_moves
from vllm.config import KVCompressConfig
from vllm.sequence import Sequence
from vllm.kvcompress.block_manager import BlockSpaceManagerKVC, FreedBlockCounts
from vllm.kvcompress.metrics import CompressionMetrics
from vllm.benchmark import BENCHMARKER
from vllm.sampling_params import SamplingParams

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
        self.new_tokens = 0
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
        self.total_evicted = 0

        # Allocate workspace for compression
        self.evicted_head_indices = torch.empty(
            (self.config.max_kv_per_compression,),
            dtype=torch.int,
            device=self.device,
        )
        self.evicted_logical_indices = torch.empty_like(
            self.evicted_head_indices)
        self.cache_move_indices = torch.empty(
            (self.config.max_kv_per_compression, 2),
            dtype=torch.int,
            device=self.device,
        )

    def complete_seqs(self, completed_seqs: List[Sequence]) -> None:
        for seq in completed_seqs:
            if seq.seq_id in self._iters_since_compression:
                del self._iters_since_compression[seq.seq_id]

    def _increment_iters_since_compression(self, compressed_seqs: List[Sequence]) -> None:
        for seq_id in self._iters_since_compression:
            self._iters_since_compression[seq_id] += 1
        for seq in compressed_seqs:
            self._iters_since_compression[seq.seq_id] = 0

    @BENCHMARKER.wrap()
    def _schedule_seq_evictions(
        self,
        seq: Sequence,
        target_compression_rate: float,
        max_cache_tokens: int,
        protected_window_size: int,
        compress_once: bool,
    ) -> Tuple[int, int]:
        """Return the number of this sequence's blocks to be freed during
        the next compression iteration.
        """
        # If sequence was configured to be compressed exactly once after prefill
        # and this compression has already occurred then return
        if compress_once and seq.compressed:
            return 0, 0

        seq.compressed = True

        # Round up to nearest block to avoid freeing blocks with KVs below max.
        if max_cache_tokens > 0:
            max_cache_tokens = (
                max_cache_tokens + self.block_size - 1
            ) // self.block_size * self.block_size

        if target_compression_rate < 1.0 and max_cache_tokens > 0:
            raise RuntimeError("both compression_rate and max_cache_tokens "
                               "specified during compression")

        total_kv_heads = self.config.num_layers * self.config.num_kv_heads

        max_cache_kv = protected_window_size * total_kv_heads
        evict_kv_count = max(
            0,
            self.block_manager.get_sequence_kv_count(seq) - max_cache_kv,
        )
        if max_cache_tokens >= 0:
            # Evict by max number of KV per sequence.
            max_cache_kv = max_cache_tokens * total_kv_heads
            max_cache_blocks = (max_cache_kv + self.block_size - 1) // self.block_size
            evict_block_count = max(
                0, self.block_manager.get_sequence_block_count(seq) - max_cache_blocks,
            )
        else:
            # Evict by target compression rate.
            # Total KV count should not go below protected_window_size * num_kv_heads.
            # Need to round up to next full block as remaining tokens in block cannot be
            # evicted without evicting any protected tokens in the block as well.
            protected_tokens = (
                (protected_window_size + self.block_size - 1) // self.block_size
                * self.block_size
            )
            compressible_token_count = (seq.data.get_len() - protected_tokens)
            if compressible_token_count <= 0:
                return 0, 0
            # Total count of KVs in compressible range if this sequence had never
            # been compressed
            compressible_kv_count = compressible_token_count * total_kv_heads
            # Actual count of KVs currently in cache within compressible range
            compressed_kv_count = self.block_manager.get_sequence_kv_count(seq)
            protected_kv = protected_tokens * total_kv_heads
            # Target count of KVs in compressible range that will yield the
            # desired compression rate
            target_kv_count = (
                math.ceil(compressible_kv_count * target_compression_rate) + protected_kv
            )

            evict_kv_count = max(0, compressed_kv_count - target_kv_count)
            evict_block_count = (evict_kv_count + self.block_size - 1) // self.block_size

        # make sure # block evictions is evenly divisible by num_layers if even_layer_evict
        if self.config.even_layer_evict:
            evict_block_count = ((evict_block_count // self.config.num_layers)
                                 * self.config.num_layers)

        assert evict_block_count <= max(
            self.block_manager.get_sequence_block_count(seq)
            - (protected_window_size + self.block_size - 1) // self.block_size * total_kv_heads,
            0,
        )

        evict_kv_count = evict_block_count * self.block_size
        return evict_kv_count, evict_block_count

    @BENCHMARKER.wrap()
    def _schedule_compression(self, seqs: List[Sequence], sampling_params: List[SamplingParams]) -> Optional[CompressionOutputs]:
        # Benchmark - 1
        # BENCHMARKER.start_range("_schedule_compression - 1")
        # print(f"(seq_idx, iters_since_compression){[(self.block_manager.batch_slot_mapping[seq.seq_id], self._iters_since_compression.get(seq.seq_id)) for seq in seqs]}")

        # Select sequences to compress this iteration and determine blocks to
        # evict.
        total_kv_count = 0
        seqs_to_compress: List[Sequence] = []
        protected_window_sizes: List[int] = []
        evicted_blocks_per_seq = []
        for _, _, seq, sample_params in sorted(
            [(self._iters_since_compression.get(s.seq_id, 0), s.seq_id, s, sp) for s, sp in zip(seqs, sampling_params)],
            reverse=True,
        ):
            # If sequence is not long enough for compression, skip.
            _, evicted_block_count = self._schedule_seq_evictions(
                seq,
                target_compression_rate=sample_params.target_compression_rate,
                max_cache_tokens=sample_params.max_cache_tokens,
                protected_window_size=sample_params.protected_window_size,
                compress_once=sample_params.compress_once,
            )
            if evicted_block_count == 0:
                # print(f"Skipping compression for sequence {seq.seq_id}")
                continue

            # Stop once we reach the maximum number of KVs to compress.
            total_kv_count += self.block_manager.get_sequence_block_count(seq) * self.block_size
            if total_kv_count > self.config.max_kv_per_compression:
                print(f"Reached maximum number of KVs for compression: {total_kv_count > self.config.max_kv_per_compression}")
                break

            seqs_to_compress.append(seq)
            protected_window_sizes.append(sample_params.protected_window_size)
            evicted_blocks_per_seq.append(evicted_block_count)
            self._iters_since_compression[seq] = 0

        if not seqs_to_compress:
            return

        # Benchmark - 2
        # BENCHMARKER.end_range("_schedule_compression - 1")
        # BENCHMARKER.start_range("_schedule_compression - 2")

        # Checkpoint
        if CHECKPOINTER.do_checkpoint:
            self.block_manager.checkpoint()
            self.compression_metrics.checkpoint()

        # Sort sequences by batch_slot_index index
        seqs_to_compress, evicted_blocks_per_seq, protected_window_sizes = zip(*sorted(
            zip(seqs_to_compress, evicted_blocks_per_seq, protected_window_sizes),
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

        # Last token in sequence was predicted during last iteration and its KVs are
        # not yet cached so we decrement by 1 to get position of that token when it is
        # added to KV cache.
        last_token_positions = torch.tensor(
            seq_lens, dtype=torch.int, device=self.device) - 1

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

        evicted_kv_offsets = (
            ((context_lens.transpose(0, 1) + self.block_size - 1) // self.block_size)
            * self.block_size
        ).flatten().cumsum(dim=0)
        evicted_kv_offsets = torch.cat(
            [torch.zeros_like(evicted_kv_offsets[:1]), evicted_kv_offsets[:-1]]
        ).reshape(*context_lens.transpose(0, 1).shape).type(torch.int)


        v2 = False
        ####### V1
        if False:
            v2 = False
            BENCHMARKER.start_range("schedule_evictions_v1")
            # Sort compression metrics
            # Should not have begun handling requests
            # init_mem = torch.cuda.max_memory_allocated(
            #     torch.device(self.device))
            sort_output = self.compression_metrics.sort_seq_metrics(
                slot_indices, last_token_positions)
            # final_mem = torch.cuda.max_memory_allocated(
            #     torch.device(self.device))
            # print(f"RAN SORT: {final_mem - init_mem}")

            # Benchmark - 3
            # BENCHMARKER.end_range("_schedule_compression - 2")
            # BENCHMARKER.start_range("_schedule_compression - 3")

            CHECKPOINTER.checkpoint('schedule_compression__context_lens', context_lens)
            CHECKPOINTER.checkpoint('schedule_compression__block_tables', block_tables)
            CHECKPOINTER.checkpoint('schedule_compression__hanging_token_count', hanging_token_count)

            # Schedule evictions
            evicted_kv_count = torch.empty(
                b_l_h,
                dtype=torch.int,
                device=self.device,
            )
            evicted_kv_offsets = (
                ((context_lens.transpose(0, 1) + self.block_size - 1) // self.block_size)
                * self.block_size
            ).flatten().cumsum(dim=0)
            evicted_kv_offsets = torch.cat(
                [torch.zeros_like(evicted_kv_offsets[:1]), evicted_kv_offsets[:-1]]
            ).reshape(*context_lens.transpose(0, 1).shape).type(torch.int)

            # for i in range(last_token_positions.shape[0] - 1):
            #     last_evictable_position = last_token_positions[i] - self.config.protected_window_size
            #     start_offset = sort_output.seq_block_offsets[i]
            #     end_offset = sort_output.seq_block_offsets[i+1]
            #     out_of_range_seq_kvs = sort_output.token_positions[start_offset:end_offset] > last_evictable_position
            #     protected_kv_count = self.config.protected_window_size * self.config.num_layers * self.config.num_kv_heads
            #     total_kv_count = (last_token_positions[i] + 1) * self.config.num_layers * self.config.num_kv_heads
            #     assert out_of_range_seq_kvs.sum() >= min(protected_kv_count, total_kv_count), f"{out_of_range_seq_kvs.sum()=} {protected_kv_count=} {total_kv_count=}"
            # last_evictable_position = last_token_positions[-1] - self.config.protected_window_size
            # start_offset = sort_output.seq_block_offsets[-1]
            # end_offset = sort_output.sorted_indices.numel()
            # out_of_range_seq_kvs = sort_output.token_positions[start_offset:end_offset] > last_evictable_position
            # protected_kv_count = self.config.protected_window_size * self.config.num_layers * self.config.num_kv_heads
            # total_kv_count = (last_token_positions[i] + 1 )* self.config.num_layers * self.config.num_kv_heads
            # assert out_of_range_seq_kvs.sum() >= min(protected_kv_count, total_kv_count), f"{out_of_range_seq_kvs.sum()=} {protected_kv_count=} {total_kv_count=}"

            protected_window_sizes = torch.tensor(
                protected_window_sizes,
                dtype=torch.int,
                device=self.device,
            )
            schedule_cache_evictions(
                self.evicted_head_indices,
                self.evicted_logical_indices,
                evicted_kv_count,
                evicted_kv_offsets,
                sort_output.sorted_indices,
                sort_output.seq_block_offsets,
                sort_output.layer_by_block,
                sort_output.head_by_block,
                sort_output.logical_block_num_by_block,
                evicted_blocks_per_seq,
                context_lens,
                hanging_token_count,
                sort_output.token_positions,
                last_token_positions,
                protected_window_sizes,
                self.block_size,
                self.config.max_kv_per_compression,
                MAX_INT,
                True,
                self.config.even_layer_evict,
                self.control_layers,
            )

            debug_dict = {}
            debug_dict['logical_indices_1'] = self.evicted_logical_indices.clone()
            debug_dict['head_indices_1'] = self.evicted_head_indices.clone()

            # for i in range(last_token_positions.shape[0] - 1):
            #     last_evictable_position = last_token_positions[i] - self.config.protected_window_size
            #     start_offset = sort_output.seq_block_offsets[i]
            #     end_offset = sort_output.seq_block_offsets[i+1]
            #     mask = self.evicted_logical_indices[start_offset * self.block_size:end_offset * self.block_size] != MAX_INT
            #     # assert (sort_output.token_positions[start_offset:end_offset].flatten()[mask] <= last_evictable_position).all()
            #     assert (self.evicted_logical_indices[start_offset * self.block_size:end_offset * self.block_size][mask] <= last_evictable_position).all()
            # last_evictable_position = last_token_positions[-1] - self.config.protected_window_size
            # start_offset = sort_output.seq_block_offsets[-1]
            # end_offset = sort_output.sorted_indices.numel()
            # mask = self.evicted_logical_indices[start_offset * self.block_size:end_offset * self.block_size] != MAX_INT
            # # assert (sort_output.token_positions[start_offset:end_offset].flatten()[mask] <= last_evictable_position).all()
            # assert (self.evicted_logical_indices[start_offset * self.block_size:end_offset * self.block_size][mask] <= last_evictable_position).all()
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

            CHECKPOINTER.checkpoint('schedule_compression__evicted_head_indices', self.evicted_head_indices)
            CHECKPOINTER.checkpoint('schedule_compression__evicted_logical_indices', self.evicted_logical_indices)
            CHECKPOINTER.checkpoint('schedule_compression__evicted_kv_count', evicted_kv_count)


            # Benchmark - 4
            # BENCHMARKER.end_range("_schedule_compression - 3")
            # BENCHMARKER.start_range("_schedule_compression - 4")

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
            self.total_evicted += evicted_kv_count.sum().item()
            print(f"TOTAL EVICTED:\nKVs: {self.total_evicted}\nTokens: {self.total_evicted / self.config.num_kv_heads / self.config.num_layers}")
            evicted_block_count = (evicted_kv_count + self.block_size - 1) // self.block_size
            # assert (evicted_block_count_ == evicted_block_count).all()
            # assert (evicted_kv_count == evicted_kv_count_).all()
            # protected_blocks = (self.config.protected_window_size + self.block_size - 1) // self.block_size
            # assert (context_lens.transpose(0, 1) >= torch.minimum(context_lens.transpose(0, 1), torch.tensor(self.config.protected_window_size))).all()
            # num_blocks = (context_lens.transpose(0, 1) + self.block_size - 1) // self.block_size
            # check = (num_blocks - evicted_block_count >= torch.minimum(num_blocks, torch.tensor(protected_blocks))).all()
            # if not check:
            #     mask = (num_blocks - evicted_block_count) < torch.minimum(num_blocks, torch.tensor(protected_blocks))
            #     print(torch.where(mask))
            #     print(context_lens.transpose(0, 1)[mask])
            #     print(evicted_block_count[mask])
            #     print((num_blocks - evicted_block_count).min())

            # assert check

            # pos_flat = sort_output.token_positions.flatten()
            # for i in range(sort_output.seq_block_offsets.shape[0] - 1):
            #     for l in range(self.config.num_layers):
            #         for h in range(self.config.num_kv_heads):
            #             evicted_logical = self.evicted_logical_indices[evicted_kv_offsets[i,l,h]:evicted_kv_offsets[i,l,h]+evicted_kv_count[i,l,h]]
            #             kv_idxs = block_tables[l,i,h,evicted_logical // self.block_size] * self.block_size + evicted_logical % self.block_size
            #             if kv_idxs.numel() > 0:
            #                 assert pos_flat[kv_idxs].max() <= (last_token_positions[i]) - self.config.protected_window_size, f"{pos_flat[kv_idxs].max()=}, {last_token_positions[i]}, {self.config.protected_window_size=}"
            #     # evicted = sort_output.sorted_indices[sort_output.seq_block_offsets[i] * self.block_size: sort_output.seq_block_offsets[i+1] * self.block_size]
            #     # evicted_pos = pos_flat[evicted.type(torch.long)]
            #     # assert evicted_pos.max() <= (last_token_positions[i]) - self.config.protected_window_size, "please"

            # assert (evicted_block_count * self.block_size == evicted_kv_count).all()

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
            logical_index_sort = self.evicted_logical_indices.sort(dim=0).indices

            seq_layer_head_indices_sort, indices = self.evicted_head_indices[logical_index_sort].sort(dim=0, stable=True)

            seq_layer_head_logical_index_sort = logical_index_sort[indices]
            self.evicted_logical_indices[:] = self.evicted_logical_indices[seq_layer_head_logical_index_sort]

            debug_dict['block_count'] = evicted_block_count
            debug_dict['kv_count'] = evicted_kv_count
            debug_dict['seq_layer_head_indices_final'] = seq_layer_head_indices_sort
            debug_dict['logical_indices_final'] = self.evicted_logical_indices
            # assert (self.evicted_logical_indices[:evicted_logical_indices.shape[0]] == evicted_logical_indices).all()

            CHECKPOINTER.checkpoint('schedule_compression__evicted_logical_indices_sorted', self.evicted_logical_indices)
            BENCHMARKER.end_range("schedule_evictions_v1")

        ####### V2
        if True:
            v2 = True
            BENCHMARKER.start_range("schedule_evictions_v2")
            evicted_logical_indices, evicted_kv_count, evicted_block_count = self.compression_metrics.schedule_evictions(
                slot_indices,
                last_token_positions,
                evicted_blocks_per_seq,
                context_lens,
                hanging_token_count,
                evicted_kv_offsets,
                protected_window_sizes,
                # debug=debug_dict,
            )
            BENCHMARKER.end_range("schedule_evictions_v2")

        #######

        ######

        # Schedule cache moves
        cache_moves_count = torch.empty(
            b_l_h,
            dtype=torch.int,
            device=self.device,
        )
        schedule_cache_moves(
            self.cache_move_indices,
            cache_moves_count,
            evicted_logical_indices if v2 else self.evicted_logical_indices,
            evicted_kv_count,
            evicted_kv_offsets,
            block_tables,
            context_lens,
            self.block_size,
        )

        # Benchmark - 5
        # BENCHMARKER.end_range("_schedule_compression - 4")
        # BENCHMARKER.start_range("_schedule_compression - 5")

        cache_moves = CacheMoves(self.cache_move_indices, cache_moves_count, evicted_kv_offsets)

        freed_block_count = {
            seq.seq_id: freed_blocks
            for seq, freed_blocks in zip(seqs_to_compress, evicted_block_count)
        }

        for seq in seqs_to_compress:
            populated_slots = seq.data.get_len() % self.block_size
            populated_slots = self.block_size if populated_slots == 0 else populated_slots
            empty_slots = self.block_size - populated_slots
            self.total_evicted_kvs[seq.seq_id] = (
                self.total_evicted_kvs.get(seq.seq_id, 0) + (torch.clamp(freed_block_count[seq.seq_id] * self.block_size - empty_slots, min=0)).sum().item()
            )
            seq_evicted_kvs = self.total_evicted_kvs[seq.seq_id]
            print(f'Seq {seq.seq_id} evicted {seq_evicted_kvs} KVs (~{seq_evicted_kvs / self.config.num_kv_heads / self.config.num_layers} tokens) so far')

        CHECKPOINTER.checkpoint('schedule_compression__cache_moves_indices', self.cache_move_indices)
        CHECKPOINTER.checkpoint('schedule_compression__cache_moves_count', cache_moves_count)
        CHECKPOINTER.checkpoint('schedule_compression__freed_block_count', evicted_block_count)

        self._increment_iters_since_compression(seqs_to_compress)

        # Free blocks that were removed by compression
        freed_blocks = self.block_manager.free_compressed_blocks(freed_block_count)

        self.compression_metrics.remove_metadata(freed_blocks)

        # # End Benchmark - 5
        # BENCHMARKER.end_range("_schedule_compression - 5")

        return CompressionOutputs(cache_moves, freed_block_count)

    def increment_new_tokens(self, new_token_count: int) -> None:
        self.new_tokens += new_token_count

    def schedule_compression(self, seqs: List[Sequence], sampling_params: List[SamplingParams],
                             force: bool = False) -> Optional[CompressionOutputs]:
        """Returns number of KV evictions per sequence"""
        self.iteration_count += 1
        if force or (self.iteration_count >= self.config.compression_interval
            or (self.config.new_token_limit > -1 and
                self.new_tokens > self.config.new_token_limit)):
            self.iteration_count = 0
            self.new_tokens = 0
            return self._schedule_compression(seqs, sampling_params)

