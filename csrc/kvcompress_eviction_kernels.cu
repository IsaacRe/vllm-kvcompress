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

#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

namespace kvcompress {

template<int BLOCK_SIZE, int MAX_TOTAL_KV_HEADS> __global__ void schedule_evictions_kernel(
  int* __restrict__ evicted_kv_indices,             // [num_seqs, num_layers, num_kv_heads, max_evicted_blocks, BLOCK_SIZE]
  int* __restrict__ evicted_kv_count,               // [num_seqs, num_layers, num_kv_heads]
  const int* __restrict__ sorted_indices,           // [total_blocks * BLOCK_SIZE] sorted indices of concat([metrics_0, ..., metrics_N]) where metrics_i[j] is eviction metric for kv j%BLOCK_SIZE of block j/BLOCK_SIZE in sequence i
  const int* __restrict__ seq_block_offsets,        // [num_seqs]
  const int* __restrict__ layer_by_block,           // [total_blocks]  TODO: could use uint8
  const int* __restrict__ head_by_block,            // [total_blocks]  TODO: could use uint8
  const int* __restrict__ evicted_blocks_per_seq,   // [num_seqs]
  const int num_layers,
  const int num_kv_heads,
  const int total_blocks,     // Total number of blocks across all layers, seqs, heads
  const int max_evicted_blocks) {
  const int num_seqs = gridDim.x * blockDim.x;
  const int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;  // allow block-level or thread-level parallelization (or both)

  const int output_head_stride = max_evicted_blocks * BLOCK_SIZE;
  const int output_seq_stride = num_layers * num_kv_heads;

  const int output_head_offset = seq_idx * output_seq_stride;

  const int seq_start_offset = seq_block_offsets[seq_idx];

  const int seq_end_offset = (seq_idx + 1 >= num_seqs) ? total_blocks : seq_block_offsets[seq_idx + 1];
  const int blocks_to_evict = evicted_blocks_per_seq[seq_idx];
  
  /*
  The unsorted metrics array is structured as [num_seqs (ragged), num_layers (ragged), num_heads (ragged), num_blocks, BLOCK_SIZE]
  so we can use index / BLOCK_SIZE to get the block index in the unsorted array,
  then lookup layer/head index using the block index.
  */

  const int* seq_sorted_indices = sorted_indices + seq_start_offset * BLOCK_SIZE;

  const int seq_total_tokens = (seq_end_offset - seq_start_offset) * BLOCK_SIZE;

  int evicted_block_count = 0;  // track number of evicted blocks for this sequence
  const int total_heads = num_layers * num_kv_heads;
  int remaining_kv_count[MAX_TOTAL_KV_HEADS];  // track number of number of KVs remaining in current block for each head
#pragma unroll
  for (int i = 0; i < total_heads; ++i) {
    remaining_kv_count[i] = BLOCK_SIZE;
  }

  int token_idx;
  int block_idx;
  int layer_idx;
  int head_idx;

  int layer_head_idx;
  int output_kv_idx;

  // Iterate over the merged sorted list (KVs for all layers of all heads for current sequence)
  for (int i = 0; i < seq_total_tokens; ++i) {
    token_idx = seq_sorted_indices[i];
    block_idx = token_idx / BLOCK_SIZE;
    layer_idx = layer_by_block[block_idx];
    head_idx = head_by_block[block_idx];

    layer_head_idx = layer_idx * num_kv_heads + head_idx;
    output_kv_idx = output_head_offset + layer_head_idx;

    // Add to evicted KVs, incrementing the evicted KV count for the current head
    // Note: only the first ( (evicted_kv_count / BLOCK_SIZE) * BLOCK_SIZE ) KVs for each head will be evicted
    evicted_kv_indices[output_kv_idx * output_head_stride + evicted_kv_count[output_kv_idx]++] = token_idx;

    // Update remaining_kv_count, incrementing total evicted blocks if we now have a full block of evicted
    // keys for the current head
    if (--(remaining_kv_count[layer_head_idx]) == 0) {
      if (++evicted_block_count >= blocks_to_evict) {
        return;
      }
      remaining_kv_count[layer_head_idx] = BLOCK_SIZE;
    }
  }
}

template<
  typename cache_t,
  int HEAD_SIZE,
  int VEC_SIZE,
  int BLOCK_SIZE>
__global__ void singe_tier_evict_kernel(
  cache_t* __restrict__ k_cache,                  // [num_blocks, head_size/x, block_size, x]
  cache_t* __restrict__ v_cache,                  // [num_blocks, head_size, block_size]
  const int* __restrict__ evicted_kv_indices,     // [num_seqs, num_layers, num_kv_heads, max_evicted_blocks, BLOCK_SIZE] indexes into [num_layers, num_blocks |(ragged), block_size]
  const int* __restrict__ evicted_kv_count,       // [num_seqs, num_layers, num_kv_heads]
  const int* __restrict__ block_tables,           // [num_seqs, num_kv_heads, max_num_blocks_per_seq]
  const int* __restrict__ context_lens,           // [num_seqs, num_kv_heads]
  const int layer_idx,
  const int layer_offset,
  const int num_layers,
  const int num_kv_heads,
  const int max_evicted_blocks,
  const int max_num_blocks_per_seq) {
  constexpr int BLOCK_STRIDE = BLOCK_SIZE * HEAD_SIZE;
  constexpr int K_STRIDE = BLOCK_SIZE * VEC_SIZE;
  const int seq_head_idx = blockIdx.y * blockDim.y + threadIdx.y;  // allow block-level or thread-level parallelization (or both)
  const int seq_head_block_idx = seq_head_idx * max_num_blocks_per_seq;
  const int seq_idx = seq_head_idx / num_kv_heads;
  const int head_idx = seq_head_idx % num_kv_heads;
  const int seq_layer_head_idx =
    seq_idx * num_layers * num_kv_heads +
    layer_idx * num_kv_heads +
    head_idx;

  // get range of src KVs that will be handled by this thread
  const int evicted_kv_cnt = evicted_kv_count[seq_layer_head_idx];
  const int evicted_kv_offset = blockIdx.x * blockDim.x + threadIdx.x;
  const int max_parallel_kv = gridDim.x * blockDim.x;

  // indices for src KVs - we move last n KVs into the slots of KVs being evicted
  int src_kv_idx = context_lens[seq_head_idx] - evicted_kv_cnt + evicted_kv_offset;
  int src_block_table_block_idx;  // index into block table of the current block
  int src_physical_block_number;
  int src_block_start; // index into k/v cache for first elem of current block
  int src_tok_offset;
  int src_k_start;
  int src_v_start;

  // indices for dst KVs
  int dst_kv_idx;       // index of KV in block table slice for the current head
  int dst_block_start;  // index into k/v cache for first elem of current block
  int dst_tok_offset;   // index of current token within block
  int dst_k_start;      // index into k cache of current token
  int dst_v_start;      // index into v cache of current token

  // used for inner loop of k-vector copy
  int src_k_group_start;
  int dst_k_group_start;

  for (
    int i = seq_layer_head_idx * max_evicted_blocks * BLOCK_SIZE + evicted_kv_offset;
    i < seq_layer_head_idx * max_evicted_blocks * BLOCK_SIZE + evicted_kv_cnt;
    i += max_parallel_kv, src_kv_idx += max_parallel_kv) {
    dst_kv_idx = evicted_kv_indices[i] - layer_offset;
    dst_block_start = dst_kv_idx / BLOCK_SIZE * BLOCK_STRIDE;
    dst_tok_offset = dst_kv_idx % BLOCK_SIZE;
    dst_k_start = dst_block_start + dst_tok_offset * VEC_SIZE;  // key cache has vector size as last dim
    dst_v_start = dst_block_start + dst_tok_offset;
    
    src_block_table_block_idx = src_kv_idx / BLOCK_SIZE;
    src_physical_block_number = block_tables[seq_head_block_idx + src_block_table_block_idx];  // block number
    src_block_start = src_physical_block_number * BLOCK_STRIDE;  // ptr to first elem in block in cache
    src_tok_offset = src_kv_idx % BLOCK_SIZE;
    src_k_start = src_block_start + src_tok_offset * VEC_SIZE;  // key cache has vector size as last dim
    src_v_start = src_block_start + src_tok_offset;

#pragma unroll
    for (int j = 0; j < BLOCK_STRIDE; j += K_STRIDE) {
      dst_k_group_start = dst_k_start + j;
      src_k_group_start = src_k_start + j;
#pragma unroll
      for (int k = 0; k < VEC_SIZE; ++k) {
        k_cache[dst_k_group_start + k] = k_cache[src_k_group_start + k];
      }
    }
  
#pragma unroll
    for (int j = 0; j < BLOCK_STRIDE; j += BLOCK_SIZE) {
      v_cache[dst_v_start + j] = v_cache[src_v_start + j];
    }
  }
}

template<
  typename cache_t,
  int HEAD_SIZE,
  int VEC_SIZE,
  int BLOCK_SIZE>
__global__ void two_tier_evict_kernel(
  cache_t* __restrict__ k_cache,                  // [num_t2_blocks, head_size/x, block_size, x]
  cache_t* __restrict__ v_cache,                  // [num_t2_blocks, head_size, block_size]
  const int* __restrict__ evicted_kv_indices,     // [num_seqs, num_layers, num_kv_heads, max_evicted_blocks, BLOCK_SIZE] indexes into [num_layers, num_t2_blocks |(ragged), block_size]
  const int* __restrict__ evicted_kv_count,       // [num_seqs, num_layers, num_kv_heads]
  const int* __restrict__ t1_block_tables,        // [num_seqs, max_num_blocks_per_seq]
  const int* __restrict__ t2_block_tables,        // [num_t1_blocks, num_kv_heads]
  const int* __restrict__ context_lens,           // [num_seqs, num_kv_heads]
  const int layer_idx,
  const int layer_offset,
  const int num_layers,
  const int num_kv_heads,
  const int max_evicted_blocks,
  const int max_num_blocks_per_seq) {
  constexpr int BLOCK_STRIDE = BLOCK_SIZE * HEAD_SIZE;
  constexpr int K_STRIDE = BLOCK_SIZE * VEC_SIZE;
  const int seq_head_idx = blockIdx.y * blockDim.y + threadIdx.y;  // allow block-level or thread-level parallelization (or both)
  const int seq_idx = seq_head_idx / num_kv_heads;
  const int seq_block_idx = seq_idx * max_num_blocks_per_seq;
  const int head_idx = seq_head_idx % num_kv_heads;
  const int seq_layer_head_idx =
    seq_idx * num_layers * num_kv_heads +
    layer_idx * num_kv_heads +
    head_idx;

  // get range of src KVs that will be handled by this thread
  const int evicted_kv_cnt = evicted_kv_count[seq_layer_head_idx];
  const int evicted_kv_offset = blockIdx.x * blockDim.x + threadIdx.x;
  const int max_parallel_kv = gridDim.x * blockDim.x;

  // indices for src KVs - we move last n KVs into the slots of KVs being evicted
  int src_kv_idx = context_lens[seq_head_idx] - evicted_kv_cnt + evicted_kv_offset;
  int src_block_table_block_idx;  // index into block table of the current block
  int64_t src_t2_physical_block_number;
  int64_t src_physical_block_number;
  int src_block_start; // index into k/v cache for first elem of current block
  int src_tok_offset;
  int src_k_start;
  int src_v_start;

  // indices for dst KVs
  int dst_kv_idx;       // index of KV in block table slice for the current head
  int dst_block_start;  // index into k/v cache for first elem of current block
  int dst_tok_offset;   // index of current token within block
  int dst_k_start;      // index into k cache of current token
  int dst_v_start;      // index into v cache of current token

  // used for inner loop of k-vector copy
  int src_k_group_start;
  int dst_k_group_start;

  for (
    int i = seq_layer_head_idx * max_evicted_blocks * BLOCK_SIZE + evicted_kv_offset;
    i < seq_layer_head_idx * max_evicted_blocks * BLOCK_SIZE + evicted_kv_cnt;
    i += max_parallel_kv, src_kv_idx += max_parallel_kv) {
    dst_kv_idx = evicted_kv_indices[i] - layer_offset;
    dst_block_start = dst_kv_idx / BLOCK_SIZE * BLOCK_STRIDE;
    dst_tok_offset = dst_kv_idx % BLOCK_SIZE;
    dst_k_start = dst_block_start + dst_tok_offset * VEC_SIZE;  // key cache has vector size as last dim
    dst_v_start = dst_block_start + dst_tok_offset;
    
    src_block_table_block_idx = src_kv_idx / BLOCK_SIZE;
    src_t2_physical_block_number = static_cast<int64_t>(t1_block_tables[seq_block_idx + src_block_table_block_idx]);
    src_physical_block_number = static_cast<int64_t>(t2_block_tables[src_t2_physical_block_number * num_kv_heads + head_idx]);  // block number
    src_block_start = src_physical_block_number * BLOCK_STRIDE;  // ptr to first elem in block in cache
    src_tok_offset = src_kv_idx % BLOCK_SIZE;
    src_k_start = src_block_start + src_tok_offset * VEC_SIZE;  // key cache has vector size as last dim
    src_v_start = src_block_start + src_tok_offset;

#pragma unroll
    for (int j = 0; j < BLOCK_STRIDE; j += K_STRIDE) {
      dst_k_group_start = dst_k_start + j;
      src_k_group_start = src_k_start + j;
#pragma unroll
      for (int k = 0; k < VEC_SIZE; ++k) {
        k_cache[dst_k_group_start + k] = k_cache[src_k_group_start + k];
      }
    }
  
#pragma unroll
    for (int j = 0; j < BLOCK_STRIDE; j += BLOCK_SIZE) {
      v_cache[dst_v_start + j] = v_cache[src_v_start + j];
    }
  }
}

}  // namespace kvcompress

void schedule_cache_evictions(
  torch::Tensor& evicted_kv_indices,        // [num_seqs, num_layers, num_kv_heads, max_evicted_blocks, BLOCK_SIZE]
  torch::Tensor& evicted_kv_count,          // [num_seqs, num_layers, num_kv_heads]
  torch::Tensor& sorted_indices,            // [total_blocks * BLOCK_SIZE] sorted indices of concat([metrics_0, ..., metrics_N]) where metrics_i[j] is eviction metric for kv j%BLOCK_SIZE of block j/BLOCK_SIZE in sequence i
  torch::Tensor& seq_block_offsets,         // [num_seqs]
  torch::Tensor& layer_by_block,            // [total_blocks]  TODO: could use uint8
  torch::Tensor& head_by_block,             // [total_blocks]  TODO: could use uint8
  torch::Tensor& evicted_blocks_per_seq,    // [num_seqs]
  int num_layers,
  int num_kv_heads,
  int total_blocks,     // Total number of blocks across all layers, seqs, heads
  int max_evicted_blocks) {
  const int block_size = evicted_kv_indices.size(4);
  const int num_layers = evicted_kv_indices.size(1);
  const int num_kv_heads = evicted_kv_indices.size(2);
  const int max_evicted_blocks = evicted_kv_indices.size(3);
  const int total_blocks = layer_by_block.size(0);
  const int total_kv_heads = num_layers * num_kv_heads;

  SCHEDULE_EVICTIONS_KERNEL(block_size, total_kv_heads)
}

#define SCHEDULE_EVICTIONS_KERNEL_BLOCK_SIZE_TOTAL_KV_HEADS(BLOCK_SIZE, TOTAL_KV_HEADS) \
  schedule_evictions_kernel<BLOCK_SIZE, TOTAL_KV_HEADS><<<1,num_seqs,0,stream>>>(\
    kv_idx,\
    kv_cnt,\
    sort_idx,\
    seq_blk_offsets,\
    layer_by_blk,\
    head_by_blk,\
    evicted_blks_per_seq,\
    num_layers,\
    num_kv_heads,\
    total_blocks,\
    max_evicted_blocks);

#define SCHEDULE_EVICTIONS_KERNEL_BLOCK_SIZE(BLOCK_SIZE, TOTAL_KV_HEADS) \
  if (TOTAL_KV_HEADS <= 1) { \
    SCHEDULE_EVICTIONS_KERNEL_BLOCK_SIZE_TOTAL_KV_HEADS(BLOCK_SIZE, 1) \
  } else if (TOTAL_KV_HEADS <= 2) { \
    SCHEDULE_EVICTIONS_KERNEL_BLOCK_SIZE_TOTAL_KV_HEADS(BLOCK_SIZE, 2) \
  } else if (TOTAL_KV_HEADS <= 4) { \
    SCHEDULE_EVICTIONS_KERNEL_BLOCK_SIZE_TOTAL_KV_HEADS(BLOCK_SIZE, 4) \
  } else if (TOTAL_KV_HEADS <= 8) { \
    SCHEDULE_EVICTIONS_KERNEL_BLOCK_SIZE_TOTAL_KV_HEADS(BLOCK_SIZE, 8) \
  } else { \
    TORCH_CHECK(false, "Unsupported num kv heads * num layers: ", TOTAL_KV_HEADS); \
  }

#define SCHEDULE_EVICTIONS_KERNEL(BLOCK_SIZE, TOTAL_KV_HEADS) \
  switch (BLOCK_SIZE) { \
    case 1: \
      SCHEDULE_EVICTIONS_KERNEL_BLOCK_SIZE(1, TOTAL_KV_HEADS) \
      break; \
    case 2: \
      SCHEDULE_EVICTIONS_KERNEL_BLOCK_SIZE(2, TOTAL_KV_HEADS) \
      break; \
    case 4: \
      SCHEDULE_EVICTIONS_KERNEL_BLOCK_SIZE(4, TOTAL_KV_HEADS) \
      break; \
    case 8: \
      SCHEDULE_EVICTIONS_KERNEL_BLOCK_SIZE(8, TOTAL_KV_HEADS) \
      break; \
    default: \
      TORCH_CHECK(false, "Unsupported block size: ", BLOCK_SIZE); \
  }

#define T1_EVICT_KERNEL_BLOCK_SIZE_VEC_SIZE_HEAD_SIZE(BLOCK_SIZE, VEC_SIZE, HEAD_SIZE, CACHE_T) \
  single_tier_evict_kernel<CACHE_T, HEAD_SIZE, VEC_SIZE, BLOCK_SIZE><<<grid,block,0,stream>>>(\
    k_cache_ptr,\
    v_cache_ptr,\
    evicted_kv_indices_ptr,\
    evicted_kv_count_ptr,\
    block_tables_ptr,\
    context_lens_ptr,\
    layer_idx,\
    layer_offset,\
    num_layers,\
    num_kv_heads,\
    max_evicted_blocks,\
    max_num_blocks_per_seq);
  
#define T1_EVICT_KERNEL_BLOCK_SIZE_VEC_SIZE(BLOCK_SIZE, VEC_SIZE, HEAD_SIZE, CACHE_T) \
  switch(HEAD_SIZE) { \
    case 1: \
      T1_EVICT_KERNEL_BLOCK_SIZE_VEC_SIZE_HEAD_SIZE(BLOCK_SIZE, VEC_SIZE, 1, CACHE_T) \
      break; \
    case 2: \
      T1_EVICT_KERNEL_BLOCK_SIZE_VEC_SIZE_HEAD_SIZE(BLOCK_SIZE, VEC_SIZE, 2, CACHE_T) \
      break; \
    case 4: \
      T1_EVICT_KERNEL_BLOCK_SIZE_VEC_SIZE_HEAD_SIZE(BLOCK_SIZE, VEC_SIZE, 4, CACHE_T) \
      break; \
    case 8: \
      T1_EVICT_KERNEL_BLOCK_SIZE_VEC_SIZE_HEAD_SIZE(BLOCK_SIZE, VEC_SIZE, 8, CACHE_T) \
      break; \
    default: \
      TORCH_CHECK(false, "Unsupported head size: ", HEAD_SIZE); \
  }
  
#define T1_EVICT_KERNEL_BLOCK_SIZE(BLOCK_SIZE, VEC_SIZE, HEAD_SIZE, CACHE_T) \
  switch(VEC_SIZE) { \
    case 1: \
      T1_EVICT_KERNEL_BLOCK_SIZE_VEC_SIZE(BLOCK_SIZE, 1, HEAD_SIZE, CACHE_T) \
      break; \
    case 2: \
      T1_EVICT_KERNEL_BLOCK_SIZE_VEC_SIZE(BLOCK_SIZE, 2, HEAD_SIZE, CACHE_T) \
      break; \
    case 4: \
      T1_EVICT_KERNEL_BLOCK_SIZE_VEC_SIZE(BLOCK_SIZE, 4, HEAD_SIZE, CACHE_T) \
      break; \
    case 8: \
      T1_EVICT_KERNEL_BLOCK_SIZE_VEC_SIZE(BLOCK_SIZE, 8, HEAD_SIZE, CACHE_T) \
      break; \
    default: \
      TORCH_CHECK(false, "Unsupported vec size for key cache: ", VEC_SIZE); \
  }

#define T1_EVICT_KERNEL(BLOCK_SIZE, VEC_SIZE, HEAD_SIZE, CACHE_T) \
  switch (BLOCK_SIZE) { \
    case 1: \
      T1_EVICT_KERNEL_BLOCK_SIZE(1, VEC_SIZE, HEAD_SIZE, CACHE_T) \
      break; \
    case 2: \
      T1_EVICT_KERNEL_BLOCK_SIZE(2, VEC_SIZE, HEAD_SIZE, CACHE_T) \
      break; \
    case 4: \
      T1_EVICT_KERNEL_BLOCK_SIZE(4, VEC_SIZE, HEAD_SIZE, CACHE_T) \
      break; \
    case 8: \
      T1_EVICT_KERNEL_BLOCK_SIZE(8, VEC_SIZE, HEAD_SIZE, CACHE_T) \
      break; \
    default: \
      TORCH_CHECK(false, "Unsupported block size: ", BLOCK_SIZE); \
  }

template<typename CACHE_T> void t1_evict_launcher(
  torch::Tensor& k_cache,                   // [num_blocks, head_size/x, block_size, x]
  torch::Tensor& v_cache,                   // [num_blocks, head_size, block_size]
  torch::Tensor& evicted_kv_indices,        // [num_seqs, num_layers, num_kv_heads, max_evicted_blocks, BLOCK_SIZE] indexes into [num_layers, num_blocks |(ragged), block_size]
  torch::Tensor& evicted_kv_count,          // [num_seqs, num_layers, num_kv_heads]
  torch::Tensor& block_tables,              // [num_seqs, num_kv_heads, max_num_blocks_per_seq]
  torch::Tensor& context_lens,              // [num_seqs, num_kv_heads]
  const int layer_idx,
  const int layer_offset,
  const int blocks_per_head,
  const int threads_per_head) {
  const int block_size = k_cache.size(2);
  const int vec_size = v_cache.size(2);
  const int head_size = v_cache.size(1);
  const int num_seqs = evicted_kv_indices.size(0);
  const int num_layers = evicted_kv_indices.size(1);
  const int num_kv_heads = evicted_kv_indices.size(2);
  const int max_evicted_blocks = evicted_kv_indices.size(3);
  const int max_num_blocks_per_seq = block_tables.size(2);

  CACHE_T* k_cache_ptr = reinterpret_cast<CACHE_T*>(k_cache.data_ptr());
  CACHE_T* v_cache_ptr = reinterpret_cast<CACHE_T*>(v_cache.data_ptr());
  int64_t* evicted_kv_indices_ptr = reinterpret_cast<int64_t*>(evicted_kv_indices.data_ptr());
  int64_t* evicted_kv_count_ptr = reinterpret_cast<int64_t*>(evicted_kv_count.data_ptr());
  int* block_tables_ptr = reinterpret_cast<int*>(block_tables.data_ptr());
  int* context_lens_ptr = reinterpret_cast<int*>(context_lens.data_ptr());

  dim3 grid(blocks_per_head, num_seqs);
  dim3 block(threads_per_head, num_kv_heads);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(k_cache));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  T1_EVICT_KERNEL(block_size, vec_size, head_size, CACHE_T)
}

#define CALL_T1_EVICT_LAUNCHER(CACHE_T) \
  t1_evict_launcher<CACHE_T>(\
    k_cache,\
    v_cache,\
    evicted_kv_indices,\
    evicted_kv_count,\
    block_tables,\
    context_lens,\
    layer_idx,\
    layer_offset,\
    blocks_per_head,\
    threads_per_head);

void evict_from_t1_cache(
  torch::Tensor& k_cache,                   // [num_blocks, head_size/x, block_size, x]
  torch::Tensor& v_cache,                   // [num_blocks, head_size, block_size]
  torch::Tensor& evicted_kv_indices,        // [num_seqs, num_layers, num_kv_heads, max_evicted_blocks, BLOCK_SIZE] indexes into [num_layers, num_blocks |(ragged), block_size]
  torch::Tensor& evicted_kv_count,          // [num_seqs, num_layers, num_kv_heads]
  torch::Tensor& block_tables,              // [num_seqs, num_kv_heads, max_num_blocks_per_seq]
  torch::Tensor& context_lens,              // [num_seqs, num_kv_heads]
  int layer_idx,
  int layer_offset,
  int blocks_per_head,
  int threads_per_head,
  const std::string& kv_cache_dtype) {
  if (kv_cache_dtype == "auto") {
    if (k_cache.dtype() == at::ScalarType::Float) {
      CALL_T1_EVICT_LAUNCHER(block_size, vec_size, head_size, float)
    } else if (k_cache.dtype() == at::ScalarType::Half) {
      CALL_T1_EVICT_LAUNCHER(block_size, vec_size, head_size, uint16_t)
    } else if (k_cache.dtype() == at::ScalarType::BFloat16) {
      CALL_T1_EVICT_LAUNCHER(block_size, vec_size, head_size, __nv_bfloat16)
    } else {
      TORCH_CHECK(false, "Unsupported data type of kv cache: ", k_cache.dtype());
    }
  } else if (kv_cache_dtype == "fp8_e5m2") {
    CALL_T1_EVICT_LAUNCHER(block_size, vec_size, head_size, uint8_t)
  } else {
    TORCH_CHECK(false, "Unsupported data type of kv cache: ", kv_cache_dtype);
  }
}
    
#define T2_EVICT_KERNEL_BLOCK_SIZE_VEC_SIZE_HEAD_SIZE(BLOCK_SIZE, VEC_SIZE, HEAD_SIZE, CACHE_T) \
  two_tier_evict_kernel<CACHE_T, HEAD_SIZE, VEC_SIZE, BLOCK_SIZE><<<grid,block,0,stream>>>(\
    k_cache_ptr,\
    v_cache_ptr,\
    evicted_kv_indices_ptr,\
    evicted_kv_count_ptr,\
    t1_block_tables_ptr,\
    t2_block_tables_ptr,\
    context_lens_ptr,\
    layer_idx,\
    layer_offset,\
    num_layers,\
    num_kv_heads,\
    max_evicted_blocks,\
    max_num_blocks_per_seq);
  
#define T2_EVICT_KERNEL_BLOCK_SIZE_VEC_SIZE(BLOCK_SIZE, VEC_SIZE, HEAD_SIZE, CACHE_T) \
  switch(HEAD_SIZE) { \
    case 1: \
      T2_EVICT_KERNEL_BLOCK_SIZE_VEC_SIZE_HEAD_SIZE(BLOCK_SIZE, VEC_SIZE, 1, CACHE_T) \
      break; \
    case 2: \
      T2_EVICT_KERNEL_BLOCK_SIZE_VEC_SIZE_HEAD_SIZE(BLOCK_SIZE, VEC_SIZE, 2, CACHE_T) \
      break; \
    case 4: \
      T2_EVICT_KERNEL_BLOCK_SIZE_VEC_SIZE_HEAD_SIZE(BLOCK_SIZE, VEC_SIZE, 4, CACHE_T) \
      break; \
    case 8: \
      T2_EVICT_KERNEL_BLOCK_SIZE_VEC_SIZE_HEAD_SIZE(BLOCK_SIZE, VEC_SIZE, 8, CACHE_T) \
      break; \
    default: \
      TORCH_CHECK(false, "Unsupported head size: ", HEAD_SIZE); \
  }

#define T2_EVICT_KERNEL_BLOCK_SIZE(BLOCK_SIZE, VEC_SIZE, HEAD_SIZE, CACHE_T) \
  switch(VEC_SIZE) { \
    case 1: \
      T2_EVICT_KERNEL_BLOCK_SIZE_VEC_SIZE(BLOCK_SIZE, 1, HEAD_SIZE, CACHE_T) \
      break; \
    case 2: \
      T2_EVICT_KERNEL_BLOCK_SIZE_VEC_SIZE(BLOCK_SIZE, 2, HEAD_SIZE, CACHE_T) \
      break; \
    case 4: \
      T2_EVICT_KERNEL_BLOCK_SIZE_VEC_SIZE(BLOCK_SIZE, 4, HEAD_SIZE, CACHE_T) \
      break; \
    case 8: \
      T2_EVICT_KERNEL_BLOCK_SIZE_VEC_SIZE(BLOCK_SIZE, 8, HEAD_SIZE, CACHE_T) \
      break; \
    default: \
      TORCH_CHECK(false, "Unsupported vec size for key cache: ", VEC_SIZE); \
  }

#define T2_EVICT_KERNEL(BLOCK_SIZE, VEC_SIZE, HEAD_SIZE, CACHE_T) \
  switch (BLOCK_SIZE) { \
    case 1: \
      T2_EVICT_KERNEL_BLOCK_SIZE(1, VEC_SIZE, HEAD_SIZE, CACHE_T) \
      break; \
    case 2: \
      T2_EVICT_KERNEL_BLOCK_SIZE(2, VEC_SIZE, HEAD_SIZE, CACHE_T) \
      break; \
    case 4: \
      T2_EVICT_KERNEL_BLOCK_SIZE(4, VEC_SIZE, HEAD_SIZE, CACHE_T) \
      break; \
    case 8: \
      T2_EVICT_KERNEL_BLOCK_SIZE(8, VEC_SIZE, HEAD_SIZE, CACHE_T) \
      break; \
    default: \
      TORCH_CHECK(false, "Unsupported block size: ", BLOCK_SIZE); \
  }

template<typename CACHE_T> void t2_evict_launcher(
  torch::Tensor& k_cache,                   // [num_t2_blocks, head_size/x, block_size, x]
  torch::Tensor& v_cache,                   // [num_t2_blocks, head_size, block_size]
  torch::Tensor& evicted_kv_indices,        // [num_seqs, num_layers, num_kv_heads, max_evicted_blocks, BLOCK_SIZE] indexes into [num_layers, num_t2_blocks |(ragged), block_size]
  torch::Tensor& evicted_kv_count,          // [num_seqs, num_layers, num_kv_heads]
  torch::Tensor& t1_block_tables,           // [num_seqs, max_num_blocks_per_seq]
  torch::Tensor& t2_block_tables,           // [num_t1_blocks, num_kv_heads]
  torch::Tensor& context_lens,              // [num_seqs, num_kv_heads]
  const int layer_idx,
  const int layer_offset,
  const int blocks_per_head,
  const int threads_per_head) {
  const int block_size = k_cache.size(2);
  const int vec_size = v_cache.size(2);
  const int head_size = v_cache.size(1);
  const int num_layers = evicted_kv_indices.size(1);
  const int num_kv_heads = evicted_kv_indices.size(2);
  const int max_evicted_blocks = evicted_kv_indices.size(3);
  const int max_num_blocks_per_seq = t1_block_tables.size(1);

  CACHE_T* k_cache_ptr = reinterpret_cast<CACHE_T*>(k_cache.data_ptr());
  CACHE_T* v_cache_ptr = reinterpret_cast<CACHE_T*>(v_cache.data_ptr());
  int64_t* evicted_kv_indices_ptr = reinterpret_cast<int64_t*>(evicted_kv_indices.data_ptr());
  int64_t* evicted_kv_count_ptr = reinterpret_cast<int64_t*>(evicted_kv_count.data_ptr());
  int* t1_block_tables_ptr = reinterpret_cast<int*>(t1_block_tables.data_ptr());
  int* t2_block_tables_ptr = reinterpret_cast<int*>(t2_block_tables.data_ptr());
  int* context_lens_ptr = reinterpret_cast<int*>(context_lens.data_ptr());

  dim3 grid(blocks_per_head, num_seqs);
  dim3 block(threads_per_head, num_kv_heads);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(k_cache));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  T2_EVICT_KERNEL(block_size, vec_size, head_size, CACHE_T)
}

#define CALL_T2_EVICT_LAUNCHER(CACHE_T) \
  t2_evict_launcher<CACHE_T>(\
    k_cache,\
    v_cache,\
    evicted_kv_indices,\
    evicted_kv_count,\
    t1_block_tables,\
    t2_block_tables,\
    context_lens,\
    layer_idx,\
    layer_offset,\
    blocks_per_head,\
    threads_per_head);

void evict_from_t2_cache(
  torch::Tensor& k_cache,                   // [num_t2_blocks, head_size/x, block_size, x]
  torch::Tensor& v_cache,                   // [num_t2_blocks, head_size, block_size]
  torch::Tensor& evicted_kv_indices,        // [num_seqs, num_layers, num_kv_heads, max_evicted_blocks, BLOCK_SIZE] indexes into [num_layers, num_t2_blocks |(ragged), block_size]
  torch::Tensor& evicted_kv_count,          // [num_seqs, num_layers, num_kv_heads]
  torch::Tensor& t1_block_tables,           // [num_seqs, max_num_blocks_per_seq]
  torch::Tensor& t2_block_tables,           // [num_t1_blocks, num_kv_heads]
  torch::Tensor& context_lens,              // [num_seqs, num_kv_heads]
  int layer_idx,
  int layer_offset,
  int blocks_per_head,
  int threads_per_head,
  const std::string& kv_cache_dtype) {
  if (kv_cache_dtype == "auto") {
    if (k_cache.dtype() == at::ScalarType::Float) {
      CALL_T2_EVICT_LAUNCHER(block_size, vec_size, head_size, float)
    } else if (k_cache.dtype() == at::ScalarType::Half) {
      CALL_T2_EVICT_LAUNCHER(block_size, vec_size, head_size, uint16_t)
    } else if (k_cache.dtype() == at::ScalarType::BFloat16) {
      CALL_T2_EVICT_LAUNCHER(block_size, vec_size, head_size, __nv_bfloat16)
    } else {
      TORCH_CHECK(false, "Unsupported data type of kv cache: ", k_cache.dtype());
    }
  } else if (kv_cache_dtype == "fp8_e5m2") {
    CALL_T2_EVICT_LAUNCHER(block_size, vec_size, head_size, uint8_t)
  } else {
    TORCH_CHECK(false, "Unsupported data type of kv cache: ", kv_cache_dtype);
  }
}

#undef DIVIDE_ROUND_UP
