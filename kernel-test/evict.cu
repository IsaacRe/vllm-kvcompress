#include <iostream>
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))
#define KERNEL_BLOCK_SIZE_VEC_SIZE_HEAD_SIZE(BLOCK_SIZE, VEC_SIZE, HEAD_SIZE) \
evict<float, HEAD_SIZE, VEC_SIZE, BLOCK_SIZE><<<grid,block>>>(\
  k_cache,\
  v_cache,\
  evicted_kv_indices,\
  evicted_kv_count,\
  block_tables,\
  context_lens,\
  layer_idx,\
  layer_offset,\
  num_layers,\
  num_kv_heads,\
  max_evicted_blocks,\
  max_num_blocks_per_seq);

#define KERNEL_BLOCK_SIZE_VEC_SIZE(BLOCK_SIZE, VEC_SIZE, HEAD_SIZE) \
  switch(HEAD_SIZE) { \
    case 1: \
      KERNEL_BLOCK_SIZE_VEC_SIZE_HEAD_SIZE(BLOCK_SIZE, VEC_SIZE, 1) \
      break; \
    case 2: \
      KERNEL_BLOCK_SIZE_VEC_SIZE_HEAD_SIZE(BLOCK_SIZE, VEC_SIZE, 2) \
      break; \
    case 4: \
      KERNEL_BLOCK_SIZE_VEC_SIZE_HEAD_SIZE(BLOCK_SIZE, VEC_SIZE, 4) \
      break; \
    case 8: \
      KERNEL_BLOCK_SIZE_VEC_SIZE_HEAD_SIZE(BLOCK_SIZE, VEC_SIZE, 8) \
      break; \
    default: \
      return 1; \
  }

#define KERNEL_BLOCK_SIZE(BLOCK_SIZE, VEC_SIZE, HEAD_SIZE) \
  switch(VEC_SIZE) { \
    case 1: \
      KERNEL_BLOCK_SIZE_VEC_SIZE(BLOCK_SIZE, 1, HEAD_SIZE) \
      break; \
    case 2: \
      KERNEL_BLOCK_SIZE_VEC_SIZE(BLOCK_SIZE, 2, HEAD_SIZE) \
      break; \
    case 4: \
      KERNEL_BLOCK_SIZE_VEC_SIZE(BLOCK_SIZE, 4, HEAD_SIZE) \
      break; \
    case 8: \
      KERNEL_BLOCK_SIZE_VEC_SIZE(BLOCK_SIZE, 8, HEAD_SIZE) \
      break; \
    default: \
      return 1; \
  }

#define KERNEL(BLOCK_SIZE, VEC_SIZE, HEAD_SIZE) \
  switch (BLOCK_SIZE) { \
    case 1: \
      KERNEL_BLOCK_SIZE(1, VEC_SIZE, HEAD_SIZE) \
      break; \
    case 2: \
      KERNEL_BLOCK_SIZE(2, VEC_SIZE, HEAD_SIZE) \
      break; \
    case 4: \
      KERNEL_BLOCK_SIZE(4, VEC_SIZE, HEAD_SIZE) \
      break; \
    case 8: \
      KERNEL_BLOCK_SIZE(8, VEC_SIZE, HEAD_SIZE) \
      break; \
    default: \
      return 1; \
  }

template<
  typename cache_t,
  int HEAD_SIZE,
  int VEC_SIZE,
  int BLOCK_SIZE>
__global__ void evict(
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

  printf("KERNEL: seq_layer_head_idx: %d, seq_head_idx: %d, seq_idx: %d\n", seq_layer_head_idx, seq_head_idx, seq_idx);
  // get range of src KVs that will be handled by this thread
  const int evicted_kv_cnt = evicted_kv_count[seq_layer_head_idx];
  const int evicted_kv_offset = blockIdx.x * blockDim.x + threadIdx.x;
  const int max_parallel_kv = gridDim.x * blockDim.x;

  printf("MAX_PARALLEL_KV: %d", max_parallel_kv);

  // printf("KERNEL: seq_head_idx: %d, evicted_kv_offset: %d\n", seq_head_idx, evicted_kv_offset);

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

  printf("evicted_kv_cnt: %d, layer_offset: %d\n", evicted_kv_cnt, layer_offset);
  printf("start_idx: %d, end_idx: %d\n", seq_layer_head_idx + evicted_kv_offset, seq_layer_head_idx + evicted_kv_cnt);
  for (
    int i = seq_layer_head_idx * max_evicted_blocks * BLOCK_SIZE + evicted_kv_offset;
    i < seq_layer_head_idx * max_evicted_blocks * BLOCK_SIZE + evicted_kv_cnt;
    i += max_parallel_kv, src_kv_idx += max_parallel_kv) {
    dst_kv_idx = evicted_kv_indices[i] - layer_offset;
    dst_block_start = dst_kv_idx / BLOCK_SIZE * BLOCK_STRIDE;
    dst_tok_offset = dst_kv_idx % BLOCK_SIZE;
    printf("seq_idx: %d, dst_tok_offset: %d, dst_kv_idx: %d, i: %d, next_i: %d, max_i: %d\n", seq_idx, dst_tok_offset, dst_kv_idx, i, i + max_parallel_kv, seq_layer_head_idx * max_evicted_blocks * BLOCK_SIZE + evicted_kv_cnt);
    dst_k_start = dst_block_start + dst_tok_offset * VEC_SIZE;  // key cache has vector size as last dim
    dst_v_start = dst_block_start + dst_tok_offset;
    
    src_block_table_block_idx = src_kv_idx / BLOCK_SIZE;
    src_physical_block_number = block_tables[seq_head_block_idx + src_block_table_block_idx];  // block number
    src_block_start = src_physical_block_number * BLOCK_STRIDE;  // ptr to first elem in block in cache
    src_tok_offset = src_kv_idx % BLOCK_SIZE;
    src_k_start = src_block_start + src_tok_offset * VEC_SIZE;  // key cache has vector size as last dim
    src_v_start = src_block_start + src_tok_offset;
    printf("SRC DEBUG: seq_head_idx=%d, src_kv_idx=%d, blk_tbl_idx=%d, phys_blk_num=%d, src_blk_start=%d, src_tok_offset=%d, src_k_start=%d\n",
      seq_head_idx, src_kv_idx, src_block_table_block_idx, src_physical_block_number, src_block_start, src_tok_offset, src_k_start);

#pragma unroll
    for (int j = 0; j < BLOCK_STRIDE; j += K_STRIDE) {
      dst_k_group_start = dst_k_start + j;
      src_k_group_start = src_k_start + j;
#pragma unroll
      for (int k = 0; k < VEC_SIZE; ++k) {
        printf("evicting k, phys_blk_num=%d, seq_idx=%d, kv_offset=%d, dst=%d, src=%d, o=%d\n", src_physical_block_number, seq_idx, evicted_kv_offset, dst_k_group_start, src_k_group_start, k);
        k_cache[dst_k_group_start + k] = k_cache[src_k_group_start + k];
      }
    }
  
#pragma unroll
    for (int j = 0; j < BLOCK_STRIDE; j += BLOCK_SIZE) {
      printf("evicting v, phys_blk_num=%d, seq_idx=%d, kv_offset=%d, dst=%d, src=%d, o=%d\n", src_physical_block_number, seq_idx, evicted_kv_offset, dst_v_start, src_v_start, j);
      v_cache[dst_v_start + j] = v_cache[src_v_start + j];
    }
  }

}

template<typename T>void print_output(
  T* t,
  int size,
  int stride,
  int substride1,
  int substride2
) {
  std::cout << "[";
  for (int i = 0; i < size; ++i) {
    if (i % stride == 0) {
      if (i == 0) {
        std::cout << "[ ";
      } else {
        std::cout << std::endl << " [ ";
      }
    }
    std::cout << t[i];
    if ((i + 1) % stride == 0) {
      std::cout << " ]";
    } else {
      if ((i + 1) % substride1 == 0) {
        std::cout << " | ";
      } else if ((i + 1) % substride2 == 0) {
        std::cout << " / ";
      } else {        
        std::cout << ",  ";
      }
    }
  }
  std::cout << "]" << std::endl;
}

void set_incr_int_data(int* ptr, int size, int incr, int stride, int wrap) {
  int data[size];
  if (stride < 1) {
    stride = 1;
  }
  if (wrap < 1) {
    wrap = size;
  }
  for (int i = 0; i < size; ++i) {
    data[i] = ((i % wrap) / stride) * incr;
  }
  print_output<int>(data, size, wrap, stride, -1);
  cudaMemcpy(ptr, data, size * sizeof(int), cudaMemcpyHostToDevice);
}

void set_incr_int_data_rev(int* ptr, int size, int incr, int stride, int wrap) {
  int data[size];
  if (stride < 1) {
    stride = 1;
  }
  if (wrap < 1) {
    wrap = size;
  }
  for (int i = 0; i < size; ++i) {
    data[i] = (wrap - 1) / stride * incr - ((i % wrap) / stride) * incr;
  }
  print_output<int>(data, size, size, 1, 1);
  cudaMemcpy(ptr, data, size * sizeof(int), cudaMemcpyHostToDevice);
}

void set_incr_float_data(float* ptr, int size, int incr, int stride, int wrap) {
  float data[size];
  if (stride < 1) {
    stride = 1;
  }
  if (wrap < 1) {
    wrap = size;
  }
  for (int i = 0; i < size; ++i) {
    data[i] = static_cast<float>(((i % wrap) / stride) * incr);
  }
  print_output<float>(data, size, wrap, stride, -1);
  cudaMemcpy(ptr, data, size * sizeof(float), cudaMemcpyHostToDevice);
}

void set_incr_int_data_multi(
  int* ptr, int size,
  int incr, int stride, int wrap,
  int incr2, int stride2, int wrap2,
  int incr3, int stride3, int wrap3,
  int incr4, int stride4, int wrap4) {
  int data[size];
  if (stride < 1) {
    stride = 1;
  }
  if (wrap < 1) {
    wrap = size;
  }
  if (stride2 < 1) {
    stride2 = 1;
  }
  if (wrap2 < 1) {
    wrap2 = size;
  }
  if (stride3 < 1) {
    stride3 = 1;
  }
  if (wrap3 < 1) {
    wrap3 = size;
  }
  if (stride4 < 1) {
    stride4 = 1;
  }
  if (wrap4 < 1) {
    wrap4 = size;
  }
  for (int i = 0; i < size; ++i) {
    data[i] = (((i % wrap) / stride) * incr);
  }
  for (int i = 0; i < size; ++i) {
    data[i] += (((i % wrap2) / stride2) * incr2);
  }
  for (int i = 0; i < size; ++i) {
    data[i] += (((i % wrap3) / stride3) * incr3);
  }
  for (int i = 0; i < size; ++i) {
    data[i] += (((i % wrap4) / stride4) * incr4);
  }
  print_output<int>(data, size, wrap, wrap2, wrap3);
  cudaMemcpy(ptr, data, size * sizeof(int), cudaMemcpyHostToDevice);
}

void set_incr_float_data_multi(
  float* ptr, int size,
  int incr, int stride, int wrap,
  int incr2, int stride2, int wrap2,
  int incr3, int stride3, int wrap3) {
  float data[size];
  if (stride < 1) {
    stride = 1;
  }
  if (wrap < 1) {
    wrap = size;
  }
  if (stride2 < 1) {
    stride2 = 1;
  }
  if (wrap2 < 1) {
    wrap2 = size;
  }
  if (stride3 < 1) {
    stride3 = 1;
  }
  if (wrap3 < 1) {
    wrap3 = size;
  }
  for (int i = 0; i < size; ++i) {
    data[i] = static_cast<float>(((i % wrap) / stride) * incr);
  }
  for (int i = 0; i < size; ++i) {
    data[i] += static_cast<float>(((i % wrap2) / stride2) * incr2);
  }
  for (int i = 0; i < size; ++i) {
    data[i] += static_cast<float>(((i % wrap3) / stride3) * incr3);
  }
  print_output<float>(data, size, wrap, wrap2, wrap3);
  cudaMemcpy(ptr, data, size * sizeof(float), cudaMemcpyHostToDevice);
}

void set_const_int_data(int* ptr, int size, int value) {
  int data[size];
  for (int i = 0; i < size; ++i) {
    data[i] = value;
  }
  print_output<int>(data, size, size, 1, 1);
  cudaMemcpy(ptr, data, size * sizeof(int), cudaMemcpyHostToDevice);
}

/*
cache_t* __restrict__ k_cache,                  // [num_blocks, head_size/x, block_size, x]
cache_t* __restrict__ v_cache,                  // [num_blocks, head_size, block_size]
const int* __restrict__ evicted_kv_indices,     // [num_seqs, num_layers, num_kv_heads, max_evicted_blocks, BLOCK_SIZE] indexes into [num_layers, num_blocks |, block_size]
const int* __restrict__ evicted_kv_count,       // [num_seqs, num_layers, num_kv_heads]
const int* __restrict__ context_lens,           // [num_seqs, num_kv_heads]
const int layer_idx,
const int layer_offset,
const int max_parallel_kv,
const int num_layers,
const int num_kv_heads
*/

void set_inputs(
  float* k_cache,
  float* v_cache,
  int* evicted_kv_indices,
  int* evicted_kv_count,
  int* block_tables,
  int* context_lens,
  int num_seqs,
  int num_blocks_per_seq,
  int num_layers,
  int num_kv_heads,
  int evicted_kvs_per_seq,
  int max_evicted_blocks,
  int head_size,
  int block_size,
  int vec_size
) {
  //TODO k cache not being set properly for large block size
  const int num_blocks = num_blocks_per_seq * num_seqs * num_kv_heads;
  printf("num_blocks: %d\n", num_blocks);
  std::cout << "k_cache" << std::endl;
  set_incr_float_data_multi(
    k_cache, num_blocks * head_size * block_size,
    vec_size * block_size, vec_size * block_size, -1, //head_size * block_size, head_size * block_size, -1,  // block offset
    1, vec_size, block_size * vec_size,  // vector offset
    block_size, 1, vec_size  // token offset
  );
  std::cout << "v_cache" << std::endl;
  set_incr_float_data(v_cache, num_blocks * head_size * block_size, 1, 1, -1);
  std::cout << "evicted_kv_indices" << std::endl;
  // [num_seqs, num_layers, num_kv_heads, max_evicted_blocks, BLOCK_SIZE]
  // must index into [num_layers, num_blocks |(ragged), block_size] and align with layer offsets
  set_incr_int_data_multi(
    evicted_kv_indices, num_seqs * num_layers * num_kv_heads * max_evicted_blocks * block_size,
    1, 1, max_evicted_blocks * block_size,  // block offset
    num_blocks * block_size, num_kv_heads * max_evicted_blocks * block_size, num_layers * num_kv_heads * max_evicted_blocks * block_size,  // layer offset
    num_blocks_per_seq * num_kv_heads * block_size, num_layers * num_kv_heads * max_evicted_blocks * block_size, -1,  // seq offset
    num_blocks_per_seq * block_size, max_evicted_blocks * block_size, num_kv_heads * max_evicted_blocks * block_size // head offset
  );
  std::cout << "evicted_kv_count" << std::endl;
  set_const_int_data(
    evicted_kv_count, num_seqs * num_layers * num_kv_heads, evicted_kvs_per_seq);
  std::cout << "block_tables" << std::endl;
  set_incr_int_data(
    block_tables, num_seqs * num_kv_heads * num_blocks_per_seq, 1, 1, -1);
  std::cout << "context_lens" << std::endl;
  set_const_int_data(
    context_lens, num_seqs * num_kv_heads, num_blocks_per_seq * block_size);
}

int main(int argc, char** argv) {
  if (argc != 12) {
    std::cerr << "wrong number of arguments"  << std::endl;
    return 1;
  }

  int num_seqs;
  int num_layers;
  int num_kv_heads;
  int evicted_kvs_per_seq;
  int block_size;
  int head_size;
  int vec_size;
  int num_blocks_per_seq;
  int layer_idx;
  int num_seq_blocks;
  int num_seq_threads;

  sscanf(argv[1], "%d", &layer_idx);  
  sscanf(argv[2], "%d", &num_layers);
  sscanf(argv[3], "%d", &num_seqs);
  sscanf(argv[4], "%d", &num_kv_heads);
  sscanf(argv[5], "%d", &evicted_kvs_per_seq);
  sscanf(argv[6], "%d", &block_size);
  sscanf(argv[7], "%d", &head_size);
  sscanf(argv[8], "%d", &vec_size);
  sscanf(argv[9], "%d", &num_blocks_per_seq);
  sscanf(argv[10], "%d", &num_seq_blocks);  
  sscanf(argv[11], "%d", &num_seq_threads);

  int max_evicted_blocks = DIVIDE_ROUND_UP(evicted_kvs_per_seq, block_size);

  if (layer_idx >= num_layers) {
    std::cerr << "not enough layers for layer idx" << std::endl;
  }

  printf("args:\nlayer_idx=%d num_layers=%d num_seqs=%d num_kv_heads=%d evicted_kvs_per_seq=%d block_size=%d head_size=%d\nvec_size=%d num_blocks_per_seq=%d num_seq_blocks=%d num_seq_threads=%d\n",
          layer_idx, num_layers, num_seqs, num_kv_heads, evicted_kvs_per_seq, block_size, head_size, vec_size, num_blocks_per_seq, num_seq_blocks, num_seq_threads);

  float* k_cache;
  float* v_cache;
  int* evicted_kv_indices;
  int* evicted_kv_count;
  int* block_tables;
  int* context_lens;
  const int cache_size = num_blocks_per_seq * num_seqs * num_kv_heads * head_size * block_size;
  cudaMalloc(&k_cache, sizeof(float)* cache_size);
  cudaMalloc(&v_cache, sizeof(float)* cache_size);
  cudaMalloc(&evicted_kv_indices, sizeof(int)* num_seqs * num_layers * num_kv_heads * max_evicted_blocks * block_size);
  cudaMalloc(&evicted_kv_count, sizeof(int)* num_seqs * num_layers * num_kv_heads);
  cudaMalloc(&block_tables, sizeof(int)* num_seqs * num_kv_heads * num_blocks_per_seq);
  cudaMalloc(&context_lens, sizeof(int)* num_seqs * num_kv_heads);
  set_inputs(
    k_cache,
    v_cache,
    evicted_kv_indices,
    evicted_kv_count,
    block_tables,
    context_lens,
    num_seqs,
    num_blocks_per_seq,
    num_layers,
    num_kv_heads,
    evicted_kvs_per_seq,
    max_evicted_blocks,
    head_size,
    block_size,
    vec_size
  );
  float k_cache_cpu[cache_size];
  cudaMemcpy(k_cache_cpu, k_cache, sizeof(float) * cache_size, cudaMemcpyDeviceToHost);
  std::cout << "K CACHE:" << std::endl;
  print_output<float>(
    k_cache_cpu,
    cache_size,
    head_size * block_size,
    block_size * vec_size,
    vec_size);
  float v_cache_cpu[cache_size];
  cudaMemcpy(v_cache_cpu, v_cache, sizeof(float) * cache_size, cudaMemcpyDeviceToHost);
  std::cout << "V CACHE:" << std::endl;
  print_output<float>(
    v_cache_cpu,
    cache_size,
    head_size * block_size,
    block_size * vec_size,
    vec_size);
  int evicted_kv_indices_cpu[num_seqs * num_layers * num_kv_heads * max_evicted_blocks * block_size];
  cudaMemcpy(evicted_kv_indices_cpu, evicted_kv_indices, sizeof(int) * num_seqs * num_layers * num_kv_heads * max_evicted_blocks * block_size, cudaMemcpyDeviceToHost);
  std::cout << "EVICTED_KV_INDICES:" << std::endl;
  print_output<int>(
    evicted_kv_indices_cpu,
    num_seqs * num_layers * num_kv_heads * max_evicted_blocks * block_size,
    num_layers * num_kv_heads * max_evicted_blocks * block_size,
    max_evicted_blocks * block_size,
    block_size);
  

  // index is into [num_layers, num_blocks, block_size]
  int layer_offset = layer_idx * num_blocks_per_seq * num_seqs * num_kv_heads * block_size;
  int max_num_blocks_per_seq = num_blocks_per_seq;

//   const int layer_idx,
// const int layer_offset,
// const int max_parallel_kv,
// const int num_layers,
// const int num_kv_heads

  dim3 grid(num_seq_blocks, num_seqs);
  dim3 block(num_seq_threads, num_kv_heads);
  printf("num_seqs: %d, num_seq_blocks: %d, num_kv_heads: %d, num_seq_threads: %d\n", num_seqs, num_seq_blocks, num_kv_heads, num_seq_threads);
  KERNEL(block_size, vec_size, head_size)

  float k_cache_out[cache_size];
  float v_cache_out[cache_size];

  cudaMemcpy(k_cache_out, k_cache, sizeof(float) * cache_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(v_cache_out, v_cache, sizeof(float) * cache_size, cudaMemcpyDeviceToHost);
  
  printf("k cache: [num_seqs=%d, num_kv_heads=%d, num_blocks_per_seq=%d, head_size/x=%d, block_size=%d, x=%d]\n",
    num_seqs, num_kv_heads, num_blocks_per_seq, head_size/vec_size, block_size, vec_size);
  print_output<float>(
    k_cache_out,
    cache_size,
    head_size * block_size,
    block_size * vec_size,
    vec_size);
  std::cout << "v cache: [num_blocks, head_size, block_size]" << std::endl;
  print_output<float>(
    v_cache_out,
    cache_size,
    head_size * block_size,
    block_size,
    1);

  return 0;
}