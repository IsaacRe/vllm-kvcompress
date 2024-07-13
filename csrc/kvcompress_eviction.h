#pragma once

#include <torch/extension.h>

void schedule_cache_evictions(
  torch::Tensor& evicted_kv_indices,        // [max_evicted_kv]
  torch::Tensor& evicted_logical_indices,   // [max_evicted_kv]
  torch::Tensor& evicted_kv_count,          // [num_seqs, num_layers, num_kv_heads]
  torch::Tensor& evicted_kv_offsets,        // [num_seqs, num_layers, num_kv_heads]
  torch::Tensor& sorted_indices,            // [total_blocks * BLOCK_SIZE] sorted indices of concat([metrics_0, ..., metrics_N]) where metrics_i[j] is eviction metric for kv j%BLOCK_SIZE of block j/BLOCK_SIZE in sequence i
  torch::Tensor& seq_block_offsets,         // [num_seqs]
  torch::Tensor& seq_evicted_kv_offsets,    // [num_seqs]
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
  const c10::optional<torch::Tensor>& control_layers,
  const int max_evicted_kv,
  const int null_eviction_index,
  const bool truncate);

void truncate_cache_evictions(
  torch::Tensor& evicted_logical_indices,   // [max_evicted_kv]
  torch::Tensor& evicted_kv_count,          // [num_seqs, num_layers, num_kv_heads]
  torch::Tensor& evicted_kv_offsets,        // [num_seqs, num_layers, num_kv_heads]
  const int block_size,
  const int max_evicted_kv,
  const int null_eviction_index);

void schedule_t1_cache_moves(
  torch::Tensor& cache_moves_idx,           // [max_evicted_kv]
  torch::Tensor& cache_moves_count,         // [num_seqs, num_layers, num_kv_heads]
  torch::Tensor& evicted_logical_indices,   // [max_evicted_kv]
  torch::Tensor& evicted_kv_count,          // [num_seqs, num_layers, num_kv_heads]
  torch::Tensor& evicted_kv_offsets,        // [num_seqs, num_layers, num_kv_heads] (offset into evicted_kv_indices for each head)
  torch::Tensor& block_tables,              // [num_layers, num_seqs, num_kv_heads, max_num_blocks_per_seq]
  torch::Tensor& context_lens,              // [num_layers, num_seqs, num_kv_heads]
  const int block_size);

void schedule_t2_cache_moves(
  torch::Tensor& cache_moves_idx,           // [num_seqs, num_kv_heads, max_evicted_tokens, 2]  (virtual token indices)
  torch::Tensor& cache_moves_count,         // [num_seqs, num_kv_heads]
  torch::Tensor& evicted_kv_indices,        // [num_seqs, num_layers, num_kv_heads, max_evicted_tokens]
  torch::Tensor& evicted_kv_count,          // [num_seqs, num_layers, num_kv_heads]
  torch::Tensor& t1_block_tables,           // [num_seqs, max_num_blocks_per_seq]
  torch::Tensor& t2_block_tables,           // [num_t1_blocks, num_kv_heads]
  torch::Tensor& context_lens,              // [num_seqs, num_kv_heads]
  const int block_size,
  const int layer_idx);

void execute_cache_moves(
  torch::Tensor& k_cache,               // [num_blocks, head_size/x, block_size, x]
  torch::Tensor& v_cache,               // [num_blocks, head_size, block_size]
  torch::Tensor& kv_metrics,            // [num_blocks, block_size]
  torch::Tensor& kv_position,           // [num_blocks, block_size]
  torch::Tensor& cache_moves_idx,       // [max_evicted_kv] indexes into [num_blocks, block_size]
  torch::Tensor& cache_moves_count,     // [num_seqs, num_layers, num_kv_heads]
  torch::Tensor& evicted_kv_offsets,    // [num_seqs, num_layers, num_kv_heads]
  int blocks_per_head,
  int threads_per_head);

