#include <iostream>
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))
#define KERNEL_BLOCK_SIZE_VEC_SIZE_HEAD_SIZE(BLOCK_SIZE, VEC_SIZE, HEAD_SIZE) \
evict<float, HEAD_SIZE, VEC_SIZE, BLOCK_SIZE><<<grid,block>>>(\
  k_cache,\
  v_cache,\
  cache_move_idx,\
  cache_move_count,\
  max_num_moves);

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

// WARNING: Will fail if cache moves of different sequences/heads specify same dst indices
template<
  typename cache_t,
  int HEAD_SIZE,
  int VEC_SIZE,
  int BLOCK_SIZE>
__global__ void evict(
  cache_t* __restrict__ k_cache,                  // [num_blocks, head_size/x, block_size, x]
  cache_t* __restrict__ v_cache,                  // [num_blocks, head_size, block_size]
  const int* __restrict__ cache_move_idx,         // [num_seqs, num_kv_heads, max_num_moves, 2] indexes into [num_blocks, block_size]
  const int* __restrict__ cache_move_count,       // [num_seqs, num_kv_heads]
  const int max_num_moves) {
  constexpr int BLOCK_STRIDE = BLOCK_SIZE * HEAD_SIZE;
  constexpr int K_STRIDE = BLOCK_SIZE * VEC_SIZE;
  const int seq_head_idx = blockIdx.y * blockDim.y + threadIdx.y;  // allow block-level or thread-level parallelization (or both)

  printf("KERNEL: seq_head_idx: %d\n", seq_head_idx);
  // get range of src KVs that will be handled by this thread
  const int moved_kv_count = cache_move_count[seq_head_idx];
  const int moved_kv_offset = blockIdx.x * blockDim.x + threadIdx.x;
  const int max_parallel_kv = gridDim.x * blockDim.x;
  const int move_table_offset = seq_head_idx * max_num_moves * 2;

  printf("MAX_PARALLEL_KV: %d", max_parallel_kv);

  for (
    int i = move_table_offset + moved_kv_offset;
    i < move_table_offset + moved_kv_count;
    i += max_parallel_kv) {
    const int src_idx = cache_move_idx[i+1];
    const int src_block_start = src_idx / BLOCK_SIZE * BLOCK_STRIDE;
    const int src_block_offset = src_idx % BLOCK_SIZE;
    const int dst_idx = cache_move_idx[i];
    const int dst_block_start = dst_idx / BLOCK_SIZE * BLOCK_STRIDE;
    const int dst_block_offset = dst_idx % BLOCK_SIZE;

    const int src_k_start = src_block_start + src_block_offset * VEC_SIZE;
    const int src_v_start = src_block_start + src_block_offset;
    const int dst_k_start = dst_block_start + dst_block_offset * VEC_SIZE;
    const int dst_v_start = dst_block_start + dst_block_offset;

    printf("SRC DEBUG: seq_head_idx=%d, src_kv_idx=%d, src_blk_start=%d, src_tok_offset=%d, src_k_start=%d\n",
      seq_head_idx, src_idx, src_block_start, src_block_offset, src_k_start);

#pragma unroll
    for (int j = 0; j < BLOCK_STRIDE; j += K_STRIDE) {
      const int dst_k_group_start = dst_k_start + j;
      const int src_k_group_start = src_k_start + j;
#pragma unroll
      for (int k = 0; k < VEC_SIZE; ++k) {
        printf("evicting k, dst_k_start=%d, dst_k_group_start=%d, src_k_start=%d, src_k_group_start=%d, kv_offset=%d,\n", dst_k_start, dst_k_group_start, src_k_start, src_k_group_start, k);
        k_cache[dst_k_group_start + k] = k_cache[src_k_group_start + k];
      }
    }
  
#pragma unroll
    for (int j = 0; j < BLOCK_STRIDE; j += BLOCK_SIZE) {
      printf("evicting v, dst_v_start=%d, src_v_start=%d, kv_offset=%d\n", dst_v_start, src_v_start, j);
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

void set_cache_moves_int_data(int* ptr, int size, int dst_incr, int dst_stride, int dst_wrap, int src_offset) {
  int data[size];
  if (dst_stride < 1) {
    dst_stride = 1;
  }
  if (dst_wrap < 1) {
    dst_wrap = size;
  }
  for (int i = 0; i < size; ++i) {
    data[i] = ((i % dst_wrap) / dst_stride) * dst_incr;
  }
  for (int i = 1; i < size; i += 2) {
    data[i] = data[i-1] + src_offset;
  }
  print_output<int>(data, size, dst_wrap, dst_stride, -1);
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
const int* __restrict__ cache_move_idx,         // [num_seqs, num_kv_heads, max_num_moves, 2] indexes into [num_blocks, block_size]
const int* __restrict__ cache_move_count,       // [num_seqs, num_kv_heads]
const int max_num_moves
*/

void set_inputs(
  float* k_cache,
  float* v_cache,
  int* cache_move_idx,
  int* cache_move_count,
  int num_seqs,
  int num_blocks_per_seq,
  int num_kv_heads,
  int num_moves_per_seq,
  int max_num_moves,
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
  std::cout << "cache_move_idx" << std::endl;
  // [num_seqs, num_kv_heads, max_num_moves, 2]
  // must index into [num_blocks, block_size]

  // move last kv of each head to first of each head
  set_cache_moves_int_data(
    cache_move_idx, num_seqs * num_kv_heads * max_num_moves * 2,
    num_blocks_per_seq * block_size, 2, -1, num_blocks_per_seq * block_size - 1  // dst offset (beginning of block)
  );

  std::cout << "cache_move_count" << std::endl;
  set_const_int_data(
    cache_move_count, num_seqs * num_kv_heads, num_moves_per_seq);
}

int main(int argc, char** argv) {
  if (argc != 10) {
    std::cerr << "wrong number of arguments"  << std::endl;
    return 1;
  }

  int num_seqs;
  int num_kv_heads;
  int num_moves_per_seq;
  int block_size;
  int head_size;
  int vec_size;
  int num_blocks_per_seq;
  int num_seq_blocks;
  int num_seq_threads;

  sscanf(argv[1], "%d", &num_seqs);
  sscanf(argv[2], "%d", &num_kv_heads);
  sscanf(argv[3], "%d", &num_moves_per_seq);  
  sscanf(argv[4], "%d", &block_size);
  sscanf(argv[5], "%d", &head_size);
  sscanf(argv[6], "%d", &vec_size);
  sscanf(argv[7], "%d", &num_blocks_per_seq);
  sscanf(argv[8], "%d", &num_seq_blocks);  
  sscanf(argv[9], "%d", &num_seq_threads);

  int max_num_moves = num_moves_per_seq;

  printf("args:\nnum_seqs=%d num_kv_heads=%d num_moves_per_seq=%d block_size=%d head_size=%d\nvec_size=%d num_blocks_per_seq=%d num_seq_blocks=%d num_seq_threads=%d\n",
          num_seqs, num_kv_heads, num_moves_per_seq, block_size, head_size, vec_size, num_blocks_per_seq, num_seq_blocks, num_seq_threads);

  float* k_cache;
  float* v_cache;
  int* cache_move_idx;
  int* cache_move_count;;
  const int cache_size = num_blocks_per_seq * num_seqs * num_kv_heads * head_size * block_size;
  cudaMalloc(&k_cache, sizeof(float)* cache_size);
  cudaMalloc(&v_cache, sizeof(float)* cache_size);
  cudaMalloc(&cache_move_idx, sizeof(int)* num_seqs * num_kv_heads * max_num_moves * 2);
  cudaMalloc(&cache_move_count, sizeof(int)* num_seqs * num_kv_heads);
  set_inputs(
    k_cache,
    v_cache,
    cache_move_idx,
    cache_move_count,
    num_seqs,
    num_blocks_per_seq,
    num_kv_heads,
    num_moves_per_seq,
    max_num_moves,
    head_size,
    block_size,
    vec_size);
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
  int cache_move_idx_cpu[num_seqs * num_kv_heads * max_num_moves * 2];
  cudaMemcpy(cache_move_idx_cpu, cache_move_idx, sizeof(int) * num_seqs * num_kv_heads * max_num_moves * 2, cudaMemcpyDeviceToHost);
  std::cout << "CACHE_MOVE_IDX [num_seqs, num_kv_heads, max_num_moves, 2]:" << std::endl;
  print_output<int>(
    cache_move_idx_cpu,
    num_seqs * num_kv_heads * max_num_moves * 2,
    num_kv_heads * max_num_moves * 2 ,
    max_num_moves * 2,
    2);
  
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