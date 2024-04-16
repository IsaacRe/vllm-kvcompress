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
#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#endif

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "attention_dtypes.h"
#include "attention_utils.cuh"
#ifdef ENABLE_FP8_E5M2
#include "../quantization/fp8_e5m2_kvcache/quant_utils.cuh"
#endif

#include <algorithm>

/*
1. Sort KV metrics globally across all sequences, layers and heads
2. schedule headwise evictions:
  3. 

*/

namespace kvcompress {

/*
TODO force BLOCK_SIZE <= 256

indices: [num_seqs (ragged), num_layers (ragged), num_heads (ragged), num_blocks, BLOCK_SIZE]
sorted_indices: [total_blocks * BLOCK_SIZE] sorted indices of concat([metrics_0, ..., metrics_N]) where metrics_i[j] is eviction metric for kv j%BLOCK_SIZE of block j/BLOCK_SIZE in sequence i
*/
// Grid: (num_seqs, 1, 1)
template<int BLOCK_SIZE, int MAX_TOTAL_KV_HEADS> __global__ void schedule_evictions(
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
  const int num_seqs = gridDim.x;

  const int output_head_stride = max_evicted_blocks * BLOCK_SIZE;
  const int output_seq_stride = num_layers * num_kv_heads;

  const int seq_idx = blockIdx.x;

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

  const int seq_total_tokens = seq_end_offset * BLOCK_SIZE - seq_start_offset;

  int evicted_block_count = 0;  // track number of evicted blocks for this sequence
  const int total_heads = num_layers * num_kv_heads;
  int remaining_kv_count[MAX_TOTAL_KV_HEADS];  // track number of number of KVs remaining in current block for each head
#pragma unroll
  for (int i = 0; i < total_heads; ++i) {
    remaining_kv_count[i] = BLOCK_SIZE;
  }

  int token_idx;
  int block_idx;
  int token_offset;
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

/*

TODO: make sure KVs being slotted into place of evicted KVs were not being evicted themselves
TODO: make sure KVs being slotted in are being put into blocks of the same seq and head
*/
// template<
//   typename scalar_t,
//   int HEAD_SIZE,
//   int BLOCK_SIZE,
//   int NUM_THREADS>
// __device__ void evict_from_cache(
//   int* __restrict__ eviction_count,  // [num_layers, num_seqs, num_kv_heads]
//   const cache_t* __restrict__ k_cache,    // [num_blocks, head_size/x, block_size, x]  (single kv head per block)
//   const cache_t* __restrict__ v_cache,    // [num_blocks, head_size, block_size]  (single kv head per block)
//   const int num_kv_heads,                 // [num_heads]
//   const float scale,
//   const int* __restrict__ block_tables,   // [num_seqs, num_kv_heads, max_num_blocks_per_seq]
//   const int* __restrict__ context_lens,   // [num_seqs, num_kv_heads]  (track num KVs per attention head per batch elem)
//   const int max_num_blocks_per_seq) {
  

// }

}  // namespace kvcompress

#define LAUNCH_SCHEDULE_EVICTIONS(MAX_TOTAL_KV_HEADS) \
  kvcompress::schedule_evictions<BLOCK_SIZE, MAX_TOTAL_KV_HEADS> \
  <<<grid, block, shared_mem_size, stream>>>( \
    evicted_kv_indices_ptr, \
    evicted_kv_count_ptr, \
    sorted_indices_ptr, \
    seq_block_offsets_ptr, \
    layer_by_block_ptr, \
    head_by_block_ptr, \
    evicted_blocks_per_seq_ptr, \
    num_layers, \
    num_kv_heads, \
    total_blocks, \
    max_evicted_blocks);

// TODO make sure we're using separate process/cuda stream for sorting/eviction
template<int BLOCK_SIZE> void schedule_evictions_launcher(
  torch::Tensor& evicted_kv_indices,
  torch::Tensor& evicted_kv_count,
  torch::Tensor& sorted_indices,
  torch::Tensor& seq_block_offsets,
  torch::Tensor& layer_by_block,
  torch::Tensor& head_by_block,
  torch::Tensor& evicted_blocks_per_seq) {
  int num_seqs = evicted_kv_indices.size(0);
  int num_layers = evicted_kv_indices.size(1);
  int num_kv_heads = evicted_kv_indices.size(2);
  int total_blocks = layer_by_block.size(0);
  int max_evicted_blocks = evicted_kv_indices.size(3);
  int total_kv_heads = num_layers * num_kv_heads;

  int* evicted_kv_indices_ptr = evicted_kv_indices.data_ptr<int>();
  int* evicted_kv_count_ptr = evicted_kv_count.data_ptr<int>();
  int* sorted_indices_ptr = sorted_indices.data_ptr<int>();
  int* seq_block_offsets_ptr = seq_block_offsets.data_ptr<int>();
  int* layer_by_block_ptr = layer_by_block.data_ptr<int>();
  int* head_by_block_ptr = head_by_block.data_ptr<int>();
  int* evicted_blocks_per_seq_ptr = evicted_blocks_per_seq.data_ptr<int>();

  dim3 grid(num_seqs, 1, 1);
  dim3 block(1);
  int shared_mem_size = 0;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(sorted_indices));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (total_kv_heads <= 256) {          // Mistral-7B (GQA)
    LAUNCH_SCHEDULE_EVICTIONS(256);
  } else if (total_kv_heads <= 384) {   // Llama-34B (GQA)
    LAUNCH_SCHEDULE_EVICTIONS(384);
  } else if (total_kv_heads <= 640) {   // Llama-70B (GQA)
    LAUNCH_SCHEDULE_EVICTIONS(640);
  } else if (total_kv_heads <= 1024) {   // Llama-7B
    LAUNCH_SCHEDULE_EVICTIONS(1024);
  } else if (total_kv_heads <= 1600) {   // Llama-13B
    LAUNCH_SCHEDULE_EVICTIONS(1600);
  } else {
    TORCH_CHECK(false, "Unsupported total number of KV heads: ", total_kv_heads);
  }
  // switch (total_kv_heads) {
  //   case 256:   // Mistral-7B (GQA)
  //     LAUNCH_SCHEDULE_EVICTIONS(256);
  //     break;
  //   case 384:   // Llama-34B (GQA)
  //     LAUNCH_SCHEDULE_EVICTIONS(384);
  //     break;
  //   case 640:   // Llama-70B (GQA)
  //     LAUNCH_SCHEDULE_EVICTIONS(640);
  //     break;
  //   case 1024:  // Llama-7B
  //     LAUNCH_SCHEDULE_EVICTIONS(1024);
  //     break;
  //   case 1600:  // Llama-13B
  //     LAUNCH_SCHEDULE_EVICTIONS(1600);
  //     break;
  //   default:
  //     TORCH_CHECK(false, "Unsupported total number of KV heads: ", total_kv_heads);
  //     break;
  // }
}

#define CALL_SCHEDULE_EVICTIONS_LAUNCHER(BLOCK_SIZE)       \
  schedule_evictions_launcher<BLOCK_SIZE>( \
    evicted_kv_indices,                                             \
    evicted_kv_count,                                               \
    sorted_indices,                                                 \
    seq_block_offsets,                                              \
    layer_by_block,                                                 \
    head_by_block,                                                  \
    evicted_blocks_per_seq);

void kvcompress_schedule_evictions(
  torch::Tensor& evicted_kv_indices,      // [num_seqs, num_layers, num_kv_heads, max_evicted_blocks, BLOCK_SIZE]
  torch::Tensor& evicted_kv_count,        // [num_seqs, num_layers, num_kv_heads]
  torch::Tensor& sorted_indices,          // [total_blocks * BLOCK_SIZE]
  torch::Tensor& seq_block_offsets,       // [num_seqs]
  torch::Tensor& layer_by_block,          // [total_blocks]
  torch::Tensor& head_by_block,           // [total_blocks]
  torch::Tensor& evicted_blocks_per_seq,  // [num_seqs]
  int block_size) {
  switch (block_size) {
    case 8:
      CALL_SCHEDULE_EVICTIONS_LAUNCHER(8);
      break;
    // case 16:
    //   CALL_SCHEDULE_EVICTIONS_LAUNCHER(16);
    //   break;
    // case 32:
    //   CALL_SCHEDULE_EVICTIONS_LAUNCHER(32);
    //   break;
    default:
      TORCH_CHECK(false, "Unsupported block size: ", block_size);
      break;
  }
}
