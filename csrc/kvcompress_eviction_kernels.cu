/*
 * Written by Isaac Rehg (irehg@cloudflare.com)
 * Copyright (c) 2024, Cloudflare, inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

namespace kvcompress {

template<int BLOCK_SIZE> __global__ void schedule_cache_evictions_kernel(
  int* __restrict__ evicted_kv_indices,             // [num_seqs, num_layers, num_kv_heads, max_evicted_tokens]
  int* __restrict__ evicted_kv_count,               // [num_seqs, num_layers, num_kv_heads]
  const int* __restrict__ sorted_indices,           // [total_blocks * BLOCK_SIZE] sorted indices of concat([metrics_0, ..., metrics_N]) where metrics_i[j] is eviction metric for kv j%BLOCK_SIZE of block j/BLOCK_SIZE in sequence i
  const int* __restrict__ seq_block_offsets,        // [num_seqs]  (offset into indices post-sort)
  const int* __restrict__ layer_by_block,           // [total_blocks]  TODO: could use uint8
  const int* __restrict__ head_by_block,            // [total_blocks]  TODO: could use uint8
  const int* __restrict__ virtual_block_num_by_block,  // [total_blocks]
  const int* __restrict__ evicted_blocks_per_seq,   // [num_seqs]
  const int* __restrict__ context_lens,             // [num_seqs, num_layers, num_kv_heads]
  const int* __restrict__ hanging_token_count,      // [num_seqs, num_layers, num_kv_heads]  number of new generated tokens for each sequence modulo block_size
  const int* __restrict__ kv_position,              // [total_blocks, BLOCK_SIZE] token position for each KV by physical token index
  const int* __restrict__ last_position,            // [num_seqs]  position of last token for each sequence
  const int num_layers,
  const int num_kv_heads,
  const int total_blocks,     // Total number of blocks across all layers, seqs, heads
  const int max_evicted_tokens,
  const int protected_window_size,
  const bool evict_evenly_per_layer,
  const int num_control_layers,
  const int* __restrict__ control_layer_indices) {
  const int num_seqs = gridDim.x * blockDim.x;
  const int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;  // allow block-level or thread-level parallelization (or both)
  const int thread_layer_idx = blockIdx.y * blockDim.y + threadIdx.y;  // parallelize kernel over layers when evict_evenly_per_layer_is_set

  // If evicting evenly per layer, return threads that are evicting for a control layer
  if (evict_evenly_per_layer) {
    for (int i = 0; i < num_control_layers; ++i) {
      if (thread_layer_idx == control_layer_indices[i]) {
        return;
      }
    }
  }

  const int max_evictable_position = last_position[seq_idx] - protected_window_size;

  const int output_head_stride = max_evicted_tokens;
  const int output_seq_stride = num_layers * num_kv_heads;

  const int output_head_offset = seq_idx * output_seq_stride;

  const int seq_start_offset = seq_block_offsets[seq_idx];

  // printf("seq %d blk offset: %d\n", seq_idx, seq_start_offset);

  const int seq_end_offset = (seq_idx + 1 >= num_seqs) ? total_blocks : seq_block_offsets[seq_idx + 1];
  int blocks_to_evict = evicted_blocks_per_seq[seq_idx];

  if (evict_evenly_per_layer) {
    blocks_to_evict /= num_layers;
  }

  // initialize all counts at zero
  for (int i = 0; i < output_seq_stride; ++i) {
    evicted_kv_count[output_head_offset + i] = 0;
  }

  if (blocks_to_evict == 0) {
    return;
  }

  // printf("seq %d evictions: %d\n", seq_idx, blocks_to_evict);
  
  /*
  The unsorted metrics array is structured as [num_seqs (ragged), num_layers (ragged), num_heads (ragged), num_blocks, BLOCK_SIZE]
  so we can use index / BLOCK_SIZE to get the block index in the unsorted array,
  then lookup layer/head index using the block index.
  */

  const int* seq_sorted_indices = sorted_indices + seq_start_offset * BLOCK_SIZE;

  const int seq_total_tokens = (seq_end_offset - seq_start_offset) * BLOCK_SIZE;
  // printf("seq %d total toks: %d\n", seq_idx, seq_total_tokens);

  int evicted_block_count = 0;  // track number of evicted blocks for this sequence
  // printf("seq %d tot kv heads: %d/%d\n", seq_idx, output_seq_stride, output_seq_stride);

  // use shared memory since MAX_TOTAL_KV_HEADS will max out registers
  extern __shared__ char shmem[];
  int* remaining_kv_count = reinterpret_cast<int*>(shmem);  // [num_seqs_per_block, MAX_TOTAL_KV_HEADS]
  const int thread_offset = threadIdx.x * output_seq_stride;

  //int remaining_kv_count[MAX_TOTAL_KV_HEADS];  // track number of number of KVs remaining in current block for each head
  for (int i = 0; i < output_seq_stride; ++i) {
    remaining_kv_count[thread_offset + i] = hanging_token_count[output_head_offset + i];
    // printf("REMAINING KV COUNT: %d vs %d", remaining_kv_count[thread_offset + i], hanging_token_count[output_head_offset + i]);
  }

  // Iterate over the merged sorted list (KVs for all layers of all heads for current sequence)
  for (int i = 0; i < seq_total_tokens; ++i) {
    const int token_idx = seq_sorted_indices[i];
    const int block_idx = token_idx / BLOCK_SIZE;
    const int layer_idx = layer_by_block[block_idx];
    if (evict_evenly_per_layer && layer_idx != thread_layer_idx) {
      continue;
    }
    const int head_idx = head_by_block[block_idx];
    const int virtual_block_num = virtual_block_num_by_block[block_idx];
    const int block_offset = token_idx % BLOCK_SIZE;
    const int virtual_token_idx = virtual_block_num * BLOCK_SIZE + block_offset;

    const int layer_head_idx = layer_idx * num_kv_heads + head_idx;
    const int output_kv_idx = output_head_offset + layer_head_idx;
    // printf("loop %d, ctx_len %d, seq %d, token_idx %d, v_token_idx %d, output_kv_idx %d, evicted_cnt: %d, remaining: %d\n",
    //   i, context_lens[output_kv_idx], seq_idx, token_idx, virtual_token_idx, output_kv_idx,
    //   evicted_kv_count[output_kv_idx], remaining_kv_count[thread_offset + layer_head_idx]);

    // Skip empty token slots in fragmented cache blocks and token slots that are within the protected window
    if (virtual_token_idx >= context_lens[output_kv_idx]
        || kv_position[token_idx] > max_evictable_position) {
      continue;
    }

    // Add to evicted KVs, incrementing the evicted KV count for the current head
    // Note: only the first ( (evicted_kv_count / BLOCK_SIZE) * BLOCK_SIZE ) KVs for each head will be evicted
    evicted_kv_indices[output_kv_idx * output_head_stride + evicted_kv_count[output_kv_idx]++] = virtual_token_idx;

    // Update remaining_kv_count, incrementing total evicted blocks if we now have a full block of evicted
    // keys for the current head
    if (--(remaining_kv_count[thread_offset + layer_head_idx]) == 0) {
      if (++evicted_block_count >= blocks_to_evict) {
        return;
      }
      remaining_kv_count[thread_offset + layer_head_idx] = BLOCK_SIZE;
    }
  }
}

template<int BLOCK_SIZE> __global__ void single_tier_schedule_cache_moves_kernel(
  int* __restrict__ cache_moves_idx,               // [num_seqs, num_layers, num_kv_heads, max_evicted_tokens, 2]  (virtual token indices)
  int* __restrict__ cache_moves_count,             // [num_seqs, num_layers, num_kv_heads]
  const int* __restrict__ evicted_kv_indices,     // [num_seqs, num_layers, num_kv_heads, max_evicted_tokens]  *indices must be sorted in ascending order
  const int* __restrict__ evicted_kv_count,       // [num_seqs, num_layers, num_kv_heads]
  const int* __restrict__ block_tables,           // [num_layers, num_seqs, num_kv_heads, max_num_blocks_per_seq]
  const int* __restrict__ context_lens,           // [num_layers, num_seqs, num_kv_heads]
  const int num_seqs,
  const int num_layers,
  const int num_kv_heads,
  const int max_evicted_tokens,
  const int max_num_blocks_per_seq) {
  const int seq_head_idx = blockIdx.x * blockDim.x + threadIdx.x;  // allow block-level or thread-level parallelization (or both)
  const int layer_idx = blockIdx.y;
  const int seq_idx = seq_head_idx / num_kv_heads;
  const int head_idx = seq_head_idx % num_kv_heads;
  const int seq_layer_head_idx =
    seq_idx * num_layers * num_kv_heads +
    layer_idx * num_kv_heads +
    head_idx;
  const int layer_seq_head_idx = 
    layer_idx * num_seqs * num_kv_heads +
    seq_idx * num_kv_heads +
    head_idx;
  const int layer_seq_head_block_offset = layer_seq_head_idx * max_num_blocks_per_seq;
  const int seq_layer_head_offset = seq_layer_head_idx * max_evicted_tokens;
  const int cache_moves_offset = seq_layer_head_idx * max_evicted_tokens * 2;

  // printf("KERNEL: layer: %d, seq_layer_head_idx: %d, seq_head_idx: %d, seq_idx: %d\n", layer_idx, seq_layer_head_idx, seq_head_idx, seq_idx);
  // get range of src KVs that will be handled by this thread
  const int evicted_kv_cnt = evicted_kv_count[seq_layer_head_idx];

  int move_count = 0;  // number of KVs not scheduled for eviction that we've moved into earlier slots of KVs that HAVE been scheduled for eviction
  int evict_count = 0;  // number of KVs scheduled for eviction that we've skipped over without moving into an earlier slot (and will therefore be lost)
  for (int i = 0; i < evicted_kv_cnt; ++i) {
    const int src_kv_idx = context_lens[layer_seq_head_idx] - 1 - i;
    const int src_kv_stop_idx = evicted_kv_indices[seq_layer_head_offset + evicted_kv_cnt - 1 - evict_count];
    const int dst_kv_idx = evicted_kv_indices[seq_layer_head_offset + move_count];

    // printf("layer: %d, seq: %d, head: %d, i: %d, src_kv_idx: %d, src_kv_stop_idx: %d, dst_kv_idx: %d, move_cnt: %d, evict_cnt: %d\n",
    //   layer_idx, seq_idx, head_idx, i, src_kv_idx, src_kv_stop_idx, dst_kv_idx, move_count, evict_count);

    if (dst_kv_idx >= src_kv_idx) {
      cache_moves_count[seq_layer_head_idx] = move_count;  // record final move count
      return;  // all KVs in slots with index > src_kv_idx were either scheduled for eviction or have been moved into earlier slots, so we can safely return
    }

    if (src_kv_idx <= src_kv_stop_idx) {
      ++evict_count;
      continue;  // current KV is scheduled for eviction so we can continue without moving it
    }

    // otherwise we move the current KV
    const int src_physical_block_number = block_tables[layer_seq_head_block_offset + src_kv_idx / BLOCK_SIZE];
    const int src_physical_kv_idx = src_physical_block_number * BLOCK_SIZE + src_kv_idx % BLOCK_SIZE;
    const int dst_physical_block_number = block_tables[layer_seq_head_block_offset + dst_kv_idx / BLOCK_SIZE];
    const int dst_physical_kv_idx = dst_physical_block_number * BLOCK_SIZE + dst_kv_idx % BLOCK_SIZE;

    // printf("moving: cache_moves_offset: %d, moves_count: %d, seq_head_idx: %d, layer: %d, seq: %d, head: %d, i: %d, src_kv_idx: %d, src_p_idx: %d, dst_kv_idx: %d, dst_p_idx: %d\n",
    //   cache_moves_offset, move_count, layer_idx, seq_idx, head_idx, i, src_kv_idx, src_physical_kv_idx, dst_kv_idx, dst_physical_kv_idx);

    const int cache_moves_idx_idx = cache_moves_offset + (move_count++) * 2;
    cache_moves_idx[cache_moves_idx_idx] = dst_physical_kv_idx;
    cache_moves_idx[cache_moves_idx_idx + 1] = src_physical_kv_idx;
  }

  cache_moves_count[seq_layer_head_idx] = move_count;  // record final move count
}

template<int BLOCK_SIZE> __global__ void two_tier_schedule_cache_moves_kernel(
  int* __restrict__ cache_moves_idx,              // [num_seqs, num_kv_heads, max_evicted_tokens, 2]
  int* __restrict__ cache_moves_count,            // [num_seqs, num_kv_heads]
  const int* __restrict__ evicted_kv_indices,     // [num_seqs, num_layers, num_kv_heads, max_evicted_tokens]  *indices must be sorted in ascending order
  const int* __restrict__ evicted_kv_count,       // [num_seqs, num_layers, num_kv_heads]
  const int* __restrict__ t1_block_tables,        // [num_seqs, max_num_blocks_per_seq]
  const int* __restrict__ t2_block_tables,        // [num_t1_blocks, num_kv_heads]
  const int* __restrict__ context_lens,           // [num_seqs, num_kv_heads]
  const int layer_idx,
  const int num_layers,
  const int num_kv_heads,
  const int max_evicted_tokens,
  const int max_num_blocks_per_seq) {
  const int seq_head_idx = blockIdx.x * blockDim.x + threadIdx.x;  // allow block-level or thread-level parallelization (or both)
  const int seq_idx = seq_head_idx / num_kv_heads;
  const int seq_block_offset = seq_idx * max_num_blocks_per_seq;
  const int head_idx = seq_head_idx % num_kv_heads;
  const int seq_layer_head_idx =
    seq_idx * num_layers * num_kv_heads +
    layer_idx * num_kv_heads +
    head_idx;
  const int seq_layer_head_offset = seq_layer_head_idx * max_evicted_tokens;
  const int cache_moves_offset = seq_head_idx * max_evicted_tokens * 2;

  // printf("KERNEL: seq_layer_head_idx: %d, seq_head_idx: %d, seq_idx: %d\n", seq_layer_head_idx, seq_head_idx, seq_idx);
  // get range of src KVs that will be handled by this thread
  const int evicted_kv_cnt = evicted_kv_count[seq_layer_head_idx];

  int move_count = 0;  // number of KVs not scheduled for eviction that we've moved into earlier slots of KVs that HAVE been scheduled for eviction
  int evict_count = 0;  // number of KVs scheduled for eviction that we've skipped over without moving into an earlier slot (and will therefore be lost)
  for (int i = 0; i < evicted_kv_cnt; ++i) {
    const int src_kv_idx = context_lens[seq_head_idx] - 1 - i;
    const int src_kv_stop_idx = evicted_kv_indices[seq_layer_head_offset + evicted_kv_cnt - 1 - evict_count];
    const int dst_kv_idx = evicted_kv_indices[seq_layer_head_offset + move_count];

    // printf("seq: %d, head: %d, i: %d, src_kv_idx: %d, src_kv_stop_idx: %d, dst_kv_idx: %d, move_cnt: %d, evict_cnt: %d\n",
    //   seq_idx, head_idx, i, src_kv_idx, src_kv_stop_idx, dst_kv_idx, move_count, evict_count);

    if (dst_kv_idx >= src_kv_idx) {
      cache_moves_count[seq_head_idx] = move_count;  // record final move count
      return;  // all KVs in slots with index > src_kv_idx were either scheduled for eviction or have been moved into earlier slots, so we can safely return
    }

    if (src_kv_idx <= src_kv_stop_idx) {
      ++evict_count;
      continue;  // current KV is scheduled for eviction so we can continue without moving it
    }

    // otherwise we move the current KV
    const int src_t2_physical_block_number = t1_block_tables[seq_block_offset + src_kv_idx / BLOCK_SIZE];
    const int src_physical_block_number = t2_block_tables[src_t2_physical_block_number * num_kv_heads + head_idx];
    const int src_physical_kv_idx = src_physical_block_number * BLOCK_SIZE + src_kv_idx % BLOCK_SIZE;
    const int dst_t2_physical_block_number = t1_block_tables[seq_block_offset + dst_kv_idx / BLOCK_SIZE];
    const int dst_physical_block_number = t2_block_tables[dst_t2_physical_block_number * num_kv_heads + head_idx];
    const int dst_physical_kv_idx = dst_physical_block_number * BLOCK_SIZE + dst_kv_idx % BLOCK_SIZE;

    // printf("moving: seq: %d, head: %d, i: %d, src_kv_idx: %d, src_p_idx: %d, dst_kv_idx: %d, dst_p_idx: %d\n",
    //   seq_idx, head_idx, i, src_kv_idx, src_physical_kv_idx, dst_kv_idx, dst_physical_kv_idx);

    const int cache_moves_idx_idx = cache_moves_offset + (move_count++) * 2;
    cache_moves_idx[cache_moves_idx_idx] = dst_physical_kv_idx;
    cache_moves_idx[cache_moves_idx_idx + 1] = src_physical_kv_idx;
  }

  cache_moves_count[seq_head_idx] = move_count;  // record final move count
}

// WARNING: Will fail if cache moves of different sequences/heads specify same dst indices
template<
  typename cache_t,
  int HEAD_SIZE,
  int VEC_SIZE,
  int BLOCK_SIZE>
__global__ void execute_cache_moves_kernel(
  cache_t* __restrict__ k_cache,                  // [num_blocks, head_size/x, block_size, x]
  cache_t* __restrict__ v_cache,                  // [num_blocks, head_size, block_size]
  float* __restrict__ kv_metrics,                 // [num_blocks, block_size]
  int* __restrict__ kv_position,                  // [num_blocks, block_size]
  const int* __restrict__ cache_move_idx,         // [num_seqs, num_layers, num_kv_heads, max_num_moves, 2] indexes into [num_blocks, block_size]
  const int* __restrict__ cache_move_count,       // [num_seqs, num_layers, num_kv_heads]
  const int max_num_moves) {
  constexpr int BLOCK_STRIDE = BLOCK_SIZE * HEAD_SIZE;
  constexpr int K_STRIDE = BLOCK_SIZE * VEC_SIZE;
  const int seq_layer_head_idx = blockIdx.y * blockDim.y + threadIdx.y;  // allow block-level or thread-level parallelization (or both)

  // get range of src KVs that will be handled by this thread
  const int moved_kv_count = cache_move_count[seq_layer_head_idx];
  const int moved_kv_offset = blockIdx.x * blockDim.x + threadIdx.x;
  const int max_parallel_kv = gridDim.x * blockDim.x;
  const int move_table_offset = seq_layer_head_idx * max_num_moves;

  for (
    int i = move_table_offset + moved_kv_offset;
    i < move_table_offset + moved_kv_count;
    i += max_parallel_kv) {
    const int move_pair_idx = i * 2;
    const int src_idx = cache_move_idx[move_pair_idx+1];
    const int src_block_start = src_idx / BLOCK_SIZE * BLOCK_STRIDE;
    const int src_block_offset = src_idx % BLOCK_SIZE;
    const int dst_idx = cache_move_idx[move_pair_idx];
    const int dst_block_start = dst_idx / BLOCK_SIZE * BLOCK_STRIDE;
    const int dst_block_offset = dst_idx % BLOCK_SIZE;

    const int src_k_start = src_block_start + src_block_offset * VEC_SIZE;
    const int src_v_start = src_block_start + src_block_offset;
    const int dst_k_start = dst_block_start + dst_block_offset * VEC_SIZE;
    const int dst_v_start = dst_block_start + dst_block_offset;

    // printf("seq_layer_head_idx: %d, move_pair_idx: %d, src: %d, dst: %d, src_blk_start: %d, dst_blk_start: %d, src_k_start: %d, src_v_start: %d, dst_k_start: %d, dst_v_start: %d\n",
    //   seq_layer_head_idx, move_pair_idx, src_idx, dst_idx, src_block_start, dst_block_start, src_k_start, src_v_start, dst_k_start, dst_v_start);

    kv_metrics[dst_idx] = kv_metrics[src_idx];
    kv_position[dst_idx] = kv_position[src_idx];

#pragma unroll
    for (int j = 0; j < BLOCK_STRIDE; j += K_STRIDE) {
      const int dst_k_group_start = dst_k_start + j;
      const int src_k_group_start = src_k_start + j;
#pragma unroll
      for (int k = 0; k < VEC_SIZE; ++k) {
        // printf("(j=%d, k=%d), ", j, k);
        k_cache[dst_k_group_start + k] = k_cache[src_k_group_start + k];
      }
    }
    // printf("\nj: ");
#pragma unroll
    for (int j = 0; j < BLOCK_STRIDE; j += BLOCK_SIZE) {
      // printf("%d, ", j);
      v_cache[dst_v_start + j] = v_cache[src_v_start + j];
    }
    // printf("\n");
  }
}

}  // namespace kvcompress

#define SCHEDULE_EVICTIONS_KERNEL_BLOCK_SIZE(BLOCK_SIZE) \
  kvcompress::schedule_cache_evictions_kernel<BLOCK_SIZE><<<grid,block,shared_mem,stream>>>( \
    evicted_kv_indices_ptr, \
    evicted_kv_count_ptr, \
    sorted_indices_ptr, \
    seq_block_offsets_ptr, \
    layer_by_block_ptr, \
    head_by_block_ptr, \
    virtual_block_num_by_block_ptr, \
    evicted_blocks_per_seq_ptr, \
    context_lens_ptr, \
    hanging_token_count_ptr, \
    kv_position_ptr, \
    last_position_ptr, \
    num_layers, \
    num_kv_heads, \
    total_blocks, \
    max_evicted_tokens, \
    protected_window_size, \
    evict_evenly_per_layer, \
    num_control_layers, \
    control_layer_indices_ptr);

#define SCHEDULE_EVICTIONS_KERNEL(BLOCK_SIZE) \
  switch (BLOCK_SIZE) { \
    case 1: \
      SCHEDULE_EVICTIONS_KERNEL_BLOCK_SIZE(1) \
      break; \
    case 2: \
      SCHEDULE_EVICTIONS_KERNEL_BLOCK_SIZE(2) \
      break; \
    case 4: \
      SCHEDULE_EVICTIONS_KERNEL_BLOCK_SIZE(4) \
      break; \
    case 16: \
      SCHEDULE_EVICTIONS_KERNEL_BLOCK_SIZE(16) \
      break; \
    default: \
      TORCH_CHECK(false, "Unsupported block size: ", BLOCK_SIZE); \
  }

void schedule_cache_evictions(
  torch::Tensor& evicted_kv_indices,        // [num_seqs, num_layers, num_kv_heads, max_evicted_tokens]
  torch::Tensor& evicted_kv_count,          // [num_seqs, num_layers, num_kv_heads]
  torch::Tensor& sorted_indices,            // [total_blocks * BLOCK_SIZE] sorted indices of concat([metrics_0, ..., metrics_N]) where metrics_i[j] is eviction metric for kv j%BLOCK_SIZE of block j/BLOCK_SIZE in sequence i
  torch::Tensor& seq_block_offsets,         // [num_seqs]
  torch::Tensor& layer_by_block,            // [total_blocks]  TODO: could use uint8
  torch::Tensor& head_by_block,             // [total_blocks]  TODO: could use uint8
  torch::Tensor& virtual_block_num_by_block,  // [total_blocks]
  torch::Tensor& evicted_blocks_per_seq,    // [num_seqs]
  torch::Tensor& context_lens,              // [num_seqs, num_layers, num_kv_heads]
  torch::Tensor& hanging_token_count,       // [num_seqs, num_layers, num_kv_heads]
  torch::Tensor& kv_position,               // [total_blocks, BLOCK_SIZE]
  torch::Tensor& last_position,             // [num_seqs]
  const int block_size,
  const int protected_window_size,
  const bool evict_evenly_per_layer,
  const c10::optional<torch::Tensor>& control_layers) {
  const int num_seqs = evicted_kv_indices.size(0);
  const int num_layers = evicted_kv_indices.size(1);
  const int num_kv_heads = evicted_kv_indices.size(2);
  const int max_evicted_tokens = evicted_kv_indices.size(3);
  const int total_blocks = layer_by_block.size(0);

  int num_control_layers = 0;
  int* control_layer_indices_ptr = nullptr;
  if (control_layers) {
    num_control_layers = control_layers.value().size(0);
    control_layer_indices_ptr = reinterpret_cast<int*>(control_layers.value().data_ptr());
  }

  int* evicted_kv_indices_ptr = reinterpret_cast<int*>(evicted_kv_indices.data_ptr());
  int* evicted_kv_count_ptr = reinterpret_cast<int*>(evicted_kv_count.data_ptr());
  int* sorted_indices_ptr = reinterpret_cast<int*>(sorted_indices.data_ptr());
  int* seq_block_offsets_ptr = reinterpret_cast<int*>(seq_block_offsets.data_ptr());
  int* layer_by_block_ptr = reinterpret_cast<int*>(layer_by_block.data_ptr());
  int* head_by_block_ptr = reinterpret_cast<int*>(head_by_block.data_ptr());
  int* virtual_block_num_by_block_ptr = reinterpret_cast<int*>(virtual_block_num_by_block.data_ptr());
  int* evicted_blocks_per_seq_ptr = reinterpret_cast<int*>(evicted_blocks_per_seq.data_ptr());
  int* context_lens_ptr = reinterpret_cast<int*>(context_lens.data_ptr());
  int* hanging_token_count_ptr = reinterpret_cast<int*>(hanging_token_count.data_ptr());
  int* kv_position_ptr = reinterpret_cast<int*>(kv_position.data_ptr());
  int* last_position_ptr = reinterpret_cast<int*>(last_position.data_ptr());
  
  // If evicting evenly across layers, launch a seperate thread per layer
  const int num_layer_threads = evict_evenly_per_layer ? num_layers : 1;

  dim3 grid(1);
  dim3 block(num_seqs, num_layer_threads);
  const int shared_mem = sizeof(int) * num_seqs * num_layers * num_kv_heads;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(evicted_kv_indices));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  SCHEDULE_EVICTIONS_KERNEL(block_size)
}

#define T1_SCHEDULE_MOVES_KERNEL_BLOCK_SIZE(BLOCK_SIZE) \
  kvcompress::single_tier_schedule_cache_moves_kernel<BLOCK_SIZE><<<grid,block,0,stream>>>(\
    cache_moves_idx_ptr,\
    cache_moves_count_ptr,\
    evicted_kv_indices_ptr,\
    evicted_kv_count_ptr,\
    block_tables_ptr,\
    context_lens_ptr,\
    num_seqs,\
    num_layers,\
    num_kv_heads,\
    max_evicted_tokens,\
    max_num_blocks_per_seq);

#define T1_SCHEDULE_MOVES_KERNEL(BLOCK_SIZE) \
  switch (BLOCK_SIZE) { \
    case 1: \
      T1_SCHEDULE_MOVES_KERNEL_BLOCK_SIZE(1) \
      break; \
    case 2: \
      T1_SCHEDULE_MOVES_KERNEL_BLOCK_SIZE(2) \
      break; \
    case 4: \
      T1_SCHEDULE_MOVES_KERNEL_BLOCK_SIZE(4) \
      break; \
    case 16: \
      T1_SCHEDULE_MOVES_KERNEL_BLOCK_SIZE(16) \
      break; \
    default: \
      TORCH_CHECK(false, "Unsupported block size: ", BLOCK_SIZE); \
  }

void schedule_t1_cache_moves(
  torch::Tensor& cache_moves_idx,           // [num_seqs, num_layers, num_kv_heads, max_evicted_tokens, 2]  (virtual token indices)
  torch::Tensor& cache_moves_count,         // [num_seqs, num_layers, num_kv_heads]
  torch::Tensor& evicted_kv_indices,        // [num_seqs, num_layers, num_kv_heads, max_evicted_tokens]
  torch::Tensor& evicted_kv_count,          // [num_seqs, num_layers, num_kv_heads]
  torch::Tensor& block_tables,              // [num_layers, num_seqs, num_kv_heads, max_num_blocks_per_seq]
  torch::Tensor& context_lens,              // [num_layers, num_seqs, num_kv_heads]
  const int block_size) {
  const int num_seqs = evicted_kv_indices.size(0);
  const int num_layers = evicted_kv_indices.size(1);
  const int num_kv_heads = evicted_kv_indices.size(2);
  const int max_evicted_tokens = evicted_kv_indices.size(3);
  const int max_num_blocks_per_seq = block_tables.size(3);

  int* cache_moves_idx_ptr = reinterpret_cast<int*>(cache_moves_idx.data_ptr());
  int* cache_moves_count_ptr = reinterpret_cast<int*>(cache_moves_count.data_ptr());
  int* evicted_kv_indices_ptr = reinterpret_cast<int*>(evicted_kv_indices.data_ptr());
  int* evicted_kv_count_ptr = reinterpret_cast<int*>(evicted_kv_count.data_ptr());
  int* block_tables_ptr = reinterpret_cast<int*>(block_tables.data_ptr());
  int* context_lens_ptr = reinterpret_cast<int*>(context_lens.data_ptr());

  dim3 grid(num_seqs, num_layers);
  dim3 block(num_kv_heads, 1);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(evicted_kv_indices));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  T1_SCHEDULE_MOVES_KERNEL(block_size)
}

#define T2_SCHEDULE_MOVES_KERNEL_BLOCK_SIZE(BLOCK_SIZE) \
  kvcompress::two_tier_schedule_cache_moves_kernel<BLOCK_SIZE><<<grid,block,0,stream>>>(\
    cache_moves_idx_ptr,\
    cache_moves_count_ptr,\
    evicted_kv_indices_ptr,\
    evicted_kv_count_ptr,\
    t1_block_tables_ptr,\
    t2_block_tables_ptr,\
    context_lens_ptr,\
    layer_idx,\
    num_layers,\
    num_kv_heads,\
    max_evicted_tokens,\
    max_num_blocks_per_seq);

#define T2_SCHEDULE_MOVES_KERNEL(BLOCK_SIZE) \
  switch (BLOCK_SIZE) { \
    case 1: \
      T2_SCHEDULE_MOVES_KERNEL_BLOCK_SIZE(1) \
      break; \
    case 2: \
      T2_SCHEDULE_MOVES_KERNEL_BLOCK_SIZE(2) \
      break; \
    case 4: \
      T2_SCHEDULE_MOVES_KERNEL_BLOCK_SIZE(4) \
      break; \
    case 8: \
      T2_SCHEDULE_MOVES_KERNEL_BLOCK_SIZE(8) \
      break; \
    default: \
      TORCH_CHECK(false, "Unsupported block size: ", BLOCK_SIZE); \
  }

void schedule_t2_cache_moves(
  torch::Tensor& cache_moves_idx,           // [num_seqs, num_kv_heads, max_evicted_tokens, 2]  (virtual token indices)
  torch::Tensor& cache_moves_count,         // [num_seqs, num_kv_heads]
  torch::Tensor& evicted_kv_indices,        // [num_seqs, num_layers, num_kv_heads, max_evicted_tokens]
  torch::Tensor& evicted_kv_count,          // [num_seqs, num_layers, num_kv_heads]
  torch::Tensor& t1_block_tables,           // [num_seqs, max_num_blocks_per_seq]
  torch::Tensor& t2_block_tables,           // [num_t1_blocks, num_kv_heads]
  torch::Tensor& context_lens,              // [num_seqs, num_kv_heads]
  const int block_size,
  const int layer_idx) {
  const int num_seqs = evicted_kv_indices.size(0);
  const int num_layers = evicted_kv_indices.size(1);
  const int num_kv_heads = evicted_kv_indices.size(2);
  const int max_evicted_tokens = evicted_kv_indices.size(3);
  const int max_num_blocks_per_seq = t1_block_tables.size(1);

  int* cache_moves_idx_ptr = reinterpret_cast<int*>(cache_moves_idx.data_ptr());
  int* cache_moves_count_ptr = reinterpret_cast<int*>(cache_moves_count.data_ptr());
  int* evicted_kv_indices_ptr = reinterpret_cast<int*>(evicted_kv_indices.data_ptr());
  int* evicted_kv_count_ptr = reinterpret_cast<int*>(evicted_kv_count.data_ptr());
  int* t1_block_tables_ptr = reinterpret_cast<int*>(t1_block_tables.data_ptr());
  int* t2_block_tables_ptr = reinterpret_cast<int*>(t2_block_tables.data_ptr());
  int* context_lens_ptr = reinterpret_cast<int*>(context_lens.data_ptr());

  dim3 grid(num_seqs);
  dim3 block(num_kv_heads);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(evicted_kv_indices));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  T2_SCHEDULE_MOVES_KERNEL(block_size)
}

#define EXECUTE_MOVES_KERNEL_HEAD_SIZE_VEC_SIZE_BLOCK_SIZE(HEAD_SIZE, VEC_SIZE, BLOCK_SIZE) \
kvcompress::execute_cache_moves_kernel<CACHE_T, HEAD_SIZE, VEC_SIZE, BLOCK_SIZE><<<grid,block,0,stream>>>(\
  k_cache_ptr,\
  v_cache_ptr,\
  kv_metrics_ptr,\
  kv_position_ptr,\
  cache_moves_idx_ptr,\
  cache_moves_count_ptr,\
  max_num_moves);

#define EXECUTE_MOVES_KERNEL_HEAD_SIZE_VEC_SIZE(HEAD_SIZE, VEC_SIZE, BLOCK_SIZE) \
  switch (BLOCK_SIZE) { \
    case 1: \
      EXECUTE_MOVES_KERNEL_HEAD_SIZE_VEC_SIZE_BLOCK_SIZE(HEAD_SIZE, VEC_SIZE, 1) \
      break; \
    case 2: \
      EXECUTE_MOVES_KERNEL_HEAD_SIZE_VEC_SIZE_BLOCK_SIZE(HEAD_SIZE, VEC_SIZE, 2) \
      break; \
    case 4: \
      EXECUTE_MOVES_KERNEL_HEAD_SIZE_VEC_SIZE_BLOCK_SIZE(HEAD_SIZE, VEC_SIZE, 4) \
      break; \
    case 16: \
      EXECUTE_MOVES_KERNEL_HEAD_SIZE_VEC_SIZE_BLOCK_SIZE(HEAD_SIZE, VEC_SIZE, 16) \
      break; \
    default: \
      TORCH_CHECK(false, "Unsupported block size: ", BLOCK_SIZE); \
  }

#define EXECUTE_MOVES_KERNEL_HEAD_SIZE(HEAD_SIZE, VEC_SIZE, BLOCK_SIZE) \
  switch (VEC_SIZE) { \
    case 1: \
      EXECUTE_MOVES_KERNEL_HEAD_SIZE_VEC_SIZE(HEAD_SIZE, 1, BLOCK_SIZE) \
      break; \
    case 2: \
      EXECUTE_MOVES_KERNEL_HEAD_SIZE_VEC_SIZE(HEAD_SIZE, 2, BLOCK_SIZE) \
      break; \
    case 8: \
      EXECUTE_MOVES_KERNEL_HEAD_SIZE_VEC_SIZE(HEAD_SIZE, 8, BLOCK_SIZE) \
      break; \
    default: \
      TORCH_CHECK(false, "Unsupported vec size: ", VEC_SIZE); \
  }

#define EXECUTE_MOVES_KERNEL(HEAD_SIZE, VEC_SIZE, BLOCK_SIZE) \
  switch (HEAD_SIZE) { \
    case 1: \
      EXECUTE_MOVES_KERNEL_HEAD_SIZE(1, VEC_SIZE, BLOCK_SIZE) \
      break; \
    case 2: \
      EXECUTE_MOVES_KERNEL_HEAD_SIZE(2, VEC_SIZE, BLOCK_SIZE) \
      break; \
    case 4: \
      EXECUTE_MOVES_KERNEL_HEAD_SIZE(4, VEC_SIZE, BLOCK_SIZE) \
      break; \
    case 128: \
      EXECUTE_MOVES_KERNEL_HEAD_SIZE(128, VEC_SIZE, BLOCK_SIZE) \
      break; \
    default: \
      TORCH_CHECK(false, "Unsupported head size: ", HEAD_SIZE); \
  }

template<typename CACHE_T> void execute_cache_moves_launcher(
  torch::Tensor& k_cache,               // [num_blocks, head_size/x, block_size, x]
  torch::Tensor& v_cache,               // [num_blocks, head_size, block_size]
  torch::Tensor& kv_metrics,            // [num_blocks, block_size]
  torch::Tensor& kv_position,           // [num_blocks, block_size]
  torch::Tensor& cache_moves_idx,       // [num_seqs, num_layers, num_kv_heads, max_num_moves, 2] indexes into [num_blocks, block_size]
  torch::Tensor& cache_moves_count,     // [num_seqs, num_layers, num_kv_heads]
  const int blocks_per_head,
  const int threads_per_head) {
  const int max_num_moves = cache_moves_idx.size(3);
  const int head_size = v_cache.size(1);
  const int vec_size = k_cache.size(3);
  const int block_size = v_cache.size(2);
  const int num_seqs = cache_moves_idx.size(0);
  const int num_layers = cache_moves_idx.size(1);
  const int num_kv_heads = cache_moves_idx.size(2);

  CACHE_T* k_cache_ptr = reinterpret_cast<CACHE_T*>(k_cache.data_ptr());
  CACHE_T* v_cache_ptr = reinterpret_cast<CACHE_T*>(v_cache.data_ptr());
  float* kv_metrics_ptr = reinterpret_cast<float*>(kv_metrics.data_ptr());
  int* kv_position_ptr = reinterpret_cast<int*>(kv_position.data_ptr());
  int* cache_moves_idx_ptr = reinterpret_cast<int*>(cache_moves_idx.data_ptr());
  int* cache_moves_count_ptr = reinterpret_cast<int*>(cache_moves_count.data_ptr());

  dim3 grid(blocks_per_head, num_seqs * num_layers);
  dim3 block(threads_per_head, num_kv_heads);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(cache_moves_idx));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  EXECUTE_MOVES_KERNEL(head_size, vec_size, block_size)
}

#define CALL_EXECUTE_CACHE_MOVES_LAUNCHER(CACHE_T) \
  execute_cache_moves_launcher<CACHE_T>(  \
    k_cache,                              \
    v_cache,                              \
    kv_metrics,                           \
    kv_position,                          \
    cache_moves_idx,                      \
    cache_moves_count,                    \
    blocks_per_head,                      \
    threads_per_head);

void execute_cache_moves(
  torch::Tensor& k_cache,               // [num_blocks, head_size/x, block_size, x]
  torch::Tensor& v_cache,               // [num_blocks, head_size, block_size]
  torch::Tensor& kv_metrics,            // [num_blocks, block_size]
  torch::Tensor& kv_position,           // [num_blocks, block_size]
  torch::Tensor& cache_moves_idx,       // [num_seqs, num_layers, num_kv_heads, max_num_moves, 2] indexes into [num_blocks, block_size]
  torch::Tensor& cache_moves_count,     // [num_seqs, num_layers, num_kv_heads]
  int blocks_per_head,
  int threads_per_head) {
  if (k_cache.dtype() == at::ScalarType::Float) {
    CALL_EXECUTE_CACHE_MOVES_LAUNCHER(float);
  } else if (k_cache.dtype() == at::ScalarType::Half) {
    CALL_EXECUTE_CACHE_MOVES_LAUNCHER(uint16_t);
  } else if (k_cache.dtype() == at::ScalarType::BFloat16) {
    CALL_EXECUTE_CACHE_MOVES_LAUNCHER(__nv_bfloat16);
  } else if (k_cache.dtype() == at::ScalarType::Float8_e5m2) {
    CALL_EXECUTE_CACHE_MOVES_LAUNCHER(uint8_t);
  } else {
    TORCH_CHECK(false, "Unsupported data type: ", k_cache.dtype());
  }
}

#undef DIVIDE_ROUND_UP
