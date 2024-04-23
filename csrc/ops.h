#pragma once

#include <torch/extension.h>

void paged_attention_v1(
  torch::Tensor& out,
  torch::Tensor& query,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  int num_kv_heads,
  float scale,
  torch::Tensor& block_tables,
  torch::Tensor& context_lens,
  int block_size,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes,
  const std::string& kv_cache_dtype);

void kvcompress_paged_attention_v1(
  torch::Tensor& out,
  torch::Tensor& kv_metric_out,
  torch::Tensor& query,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  int num_kv_heads,
  float scale,
  torch::Tensor& block_tables,
  torch::Tensor& context_lens,
  int block_size,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes,
  const std::string& kv_cache_dtype);

void kvcompress_t2_paged_attention_v1(
  torch::Tensor& out,
  torch::Tensor& kv_metric_out,
  torch::Tensor& query,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  int num_kv_heads,
  float scale,
  torch::Tensor& t1_block_tables,
  torch::Tensor& t2_block_tables,
  torch::Tensor& context_lens,
  int block_size,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes,
  const std::string& kv_cache_dtype);

void paged_attention_v2(
  torch::Tensor& out,
  torch::Tensor& exp_sums,
  torch::Tensor& max_logits,
  torch::Tensor& tmp_out,
  torch::Tensor& query,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  int num_kv_heads,
  float scale,
  torch::Tensor& block_tables,
  torch::Tensor& context_lens,
  int block_size,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes,
  const std::string& kv_cache_dtype);

void kvcompress_paged_attention_v2(
  torch::Tensor& out,
  torch::Tensor& kv_metric_out,
  torch::Tensor& exp_sums,
  torch::Tensor& max_logits,
  torch::Tensor& tmp_out,
  torch::Tensor& tmp_kv_metric_out,
  torch::Tensor& query,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  int num_kv_heads,
  float scale,
  torch::Tensor& block_tables,
  torch::Tensor& context_lens,
  int block_size,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes,
  const std::string& kv_cache_dtype);

void kvcompress_t2_paged_attention_v2(
  torch::Tensor& out,
  torch::Tensor& kv_metric_out,
  torch::Tensor& exp_sums,
  torch::Tensor& max_logits,
  torch::Tensor& tmp_out,
  torch::Tensor& tmp_kv_metric_out,
  torch::Tensor& query,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  int num_kv_heads,
  float scale,
  torch::Tensor& t1_block_tables,
  torch::Tensor& t2_block_tables,
  torch::Tensor& context_lens,
  int block_size,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes,
  const std::string& kv_cache_dtype);

void schedule_cache_evictions(
  torch::Tensor& evicted_kv_indices,        // [num_seqs, num_layers, num_kv_heads, max_evicted_blocks, BLOCK_SIZE]
  torch::Tensor& evicted_kv_count,          // [num_seqs, num_layers, num_kv_heads]
  torch::Tensor& sorted_indices,            // [total_blocks * BLOCK_SIZE] sorted indices of concat([metrics_0, ..., metrics_N]) where metrics_i[j] is eviction metric for kv j%BLOCK_SIZE of block j/BLOCK_SIZE in sequence i
  torch::Tensor& seq_block_offsets,         // [num_seqs]
  torch::Tensor& layer_block_offsets,       // [num_layers]
  torch::Tensor& layer_by_block,            // [total_blocks]
  torch::Tensor& head_by_block,             // [total_blocks]
  torch::Tensor& virtual_block_num_by_block,  // [total_blocks]
  torch::Tensor& evicted_blocks_per_seq);   // [num_seqs]

void schedule_t1_cache_moves(
  torch::Tensor& cache_moves_idx,           // [num_seqs, num_kv_heads, max_evicted_blocks * BLOCK_SIZE / 2, 2]  (virtual token indices)
  torch::Tensor& cache_moves_count,         // [num_seqs, num_kv_heads]
  torch::Tensor& evicted_kv_indices,        // [num_seqs, num_layers, num_kv_heads, max_evicted_blocks, BLOCK_SIZE] indexes into [num_layers, num_blocks |(ragged), block_size]
  torch::Tensor& evicted_kv_count,          // [num_seqs, num_layers, num_kv_heads]
  torch::Tensor& block_tables,              // [num_seqs, num_kv_heads, max_num_blocks_per_seq]
  torch::Tensor& context_lens,              // [num_seqs, num_kv_heads]
  const int block_size,
  const int layer_idx);

void schedule_t2_cache_moves(
  torch::Tensor& cache_moves_idx,           // [num_seqs, num_kv_heads, max_evicted_blocks * BLOCK_SIZE / 2, 2]  (virtual token indices)
  torch::Tensor& cache_moves_count,         // [num_seqs, num_kv_heads]
  torch::Tensor& evicted_kv_indices,        // [num_seqs, num_layers, num_kv_heads, max_evicted_blocks, BLOCK_SIZE] indexes into [num_layers, num_blocks |(ragged), block_size]
  torch::Tensor& evicted_kv_count,          // [num_seqs, num_layers, num_kv_heads]
  torch::Tensor& t1_block_tables,           // [num_seqs, max_num_blocks_per_seq]
  torch::Tensor& t2_block_tables,           // [num_t1_blocks, num_kv_heads]
  torch::Tensor& context_lens,              // [num_seqs, num_kv_heads]
  const int block_size,
  const int layer_idx);

void execute_cache_moves(
  torch::Tensor& k_cache,               // [num_blocks, head_size/x, block_size, x]
  torch::Tensor& v_cache,               // [num_blocks, head_size, block_size]
  torch::Tensor& cache_moves_idx,       // [num_seqs, num_kv_heads, max_num_moves, 2] indexes into [num_blocks, block_size]
  torch::Tensor& cache_moves_count,     // [num_seqs, num_kv_heads]
  int blocks_per_head,
  int threads_per_head);

void rms_norm(
  torch::Tensor& out,
  torch::Tensor& input,
  torch::Tensor& weight,
  float epsilon);

void fused_add_rms_norm(
  torch::Tensor& input,
  torch::Tensor& residual,
  torch::Tensor& weight,
  float epsilon);

void rotary_embedding(
  torch::Tensor& positions,
  torch::Tensor& query,
  torch::Tensor& key,
  int head_size,
  torch::Tensor& cos_sin_cache,
  bool is_neox);

void silu_and_mul(
  torch::Tensor& out,
  torch::Tensor& input);

void gelu_and_mul(
  torch::Tensor& out,
  torch::Tensor& input);

void gelu_new(
  torch::Tensor& out,
  torch::Tensor& input);

void gelu_fast(
  torch::Tensor& out,
  torch::Tensor& input);

#ifndef USE_ROCM
torch::Tensor awq_gemm(
  torch::Tensor _in_feats,
  torch::Tensor _kernel,
  torch::Tensor _scaling_factors,
  torch::Tensor _zeros,
  int split_k_iters);

torch::Tensor awq_dequantize(
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int split_k_iters,
    int thx,
    int thy);

torch::Tensor marlin_gemm(
    torch::Tensor& a, 
    torch::Tensor& b_q_weight,
    torch::Tensor& b_scales, 
    torch::Tensor& workspace,
    int64_t size_m, 
    int64_t size_n, 
    int64_t size_k);
#endif

void squeezellm_gemm(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table);

torch::Tensor gptq_gemm(
  torch::Tensor a,
  torch::Tensor b_q_weight,
  torch::Tensor b_gptq_qzeros,
  torch::Tensor b_gptq_scales,
  torch::Tensor b_g_idx,
  bool use_exllama,
  int bit);

void gptq_shuffle(
  torch::Tensor q_weight,
  torch::Tensor q_perm,
  int bit);

void moe_align_block_size(
  torch::Tensor topk_ids,
  int num_experts,
  int block_size,
  torch::Tensor sorted_token_ids,
  torch::Tensor experts_ids,
  torch::Tensor num_tokens_post_pad);

#ifndef USE_ROCM
using fptr_t = uint64_t;
fptr_t init_custom_ar(torch::Tensor &meta, torch::Tensor &rank_data,
                    const std::vector<std::string> &handles,
                    const std::vector<int64_t> &offsets, int rank,
                    bool full_nvlink);
bool should_custom_ar(torch::Tensor &inp, int max_size, int world_size,
                      bool full_nvlink);
void all_reduce_reg(fptr_t _fa, torch::Tensor &inp, torch::Tensor &out);
void all_reduce_unreg(fptr_t _fa, torch::Tensor &inp, torch::Tensor &reg_buffer,
                      torch::Tensor &out);
void dispose(fptr_t _fa);
int meta_size();
void register_buffer(fptr_t _fa, torch::Tensor &t,
                     const std::vector<std::string> &handles,
                     const std::vector<int64_t> &offsets);
std::pair<std::vector<uint8_t>, std::vector<int64_t>> get_graph_buffer_ipc_meta(fptr_t _fa);
void register_graph_buffers(fptr_t _fa, const std::vector<std::string> &handles,
                            const std::vector<std::vector<int64_t>> &offsets);
#endif
