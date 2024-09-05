#include <iostream>
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))
#define KERNEL_BLOCK_SIZE(BLOCK_SIZE) \
schedule_moves_t2<BLOCK_SIZE><<<grid,block>>>(\
  cache_moves_idx,\
  cache_moves_count,\
  evicted_kv_indices,\
  evicted_kv_count,\
  t1_block_tables,\
  t2_block_tables,\
  context_lens,\
  layer_idx,\
  num_layers,\
  num_kv_heads,\
  max_evicted_tokens,\
  max_num_blocks_per_seq);


#define KERNEL(BLOCK_SIZE) \
  switch (BLOCK_SIZE) { \
    case 1: \
      KERNEL_BLOCK_SIZE(1) \
      break; \
    case 2: \
      KERNEL_BLOCK_SIZE(2) \
      break; \
    case 4: \
      KERNEL_BLOCK_SIZE(4) \
      break; \
    case 8: \
      KERNEL_BLOCK_SIZE(8) \
      break; \
    default: \
      return 1; \
  }

template<int BLOCK_SIZE> __global__ void schedule_moves_t2(
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
  const int cache_moves_offset = seq_head_idx * max_evicted_tokens;

  printf("KERNEL: seq_layer_head_idx: %d, seq_head_idx: %d, seq_idx: %d\n", seq_layer_head_idx, seq_head_idx, seq_idx);
  // get range of src KVs that will be handled by this thread
  const int evicted_kv_cnt = evicted_kv_count[seq_layer_head_idx];

  int move_count = 0;  // number of KVs not scheduled for eviction that we've moved into earlier slots of KVs that HAVE been scheduled for eviction
  int evict_count = 0;  // number of KVs scheduled for eviction that we've skipped over without moving into an earlier slot (and will therefore be lost)
  for (int i = 0; i < evicted_kv_cnt; ++i) {
    const int src_kv_idx = context_lens[seq_head_idx] - 1 - i;
    const int src_kv_stop_idx = evicted_kv_indices[seq_layer_head_offset + evicted_kv_cnt - 1 - evict_count];
    const int dst_kv_idx = evicted_kv_indices[seq_layer_head_offset + move_count];

    printf("src_kv_idx: %d, src_kv_stop_idx: %d, dst_kv_idx: %d", src_kv_idx, src_kv_stop_idx, dst_kv_idx);

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

    const int cache_moves_idx_idx = cache_moves_offset + (move_count++);
    cache_moves_idx[cache_moves_idx_idx] = dst_physical_kv_idx;
    cache_moves_idx[cache_moves_idx_idx + 1] = src_physical_kv_idx;
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
int* __restrict__ cache_moves_idx,               // [num_seqs, num_kv_heads, max_evicted_tokens, 2]  (virtual token indices)
int* __restrict__ cache_moves_count,             // [num_seqs, num_kv_heads]
const int* __restrict__ evicted_kv_indices,     // [num_seqs, num_layers, num_kv_heads, max_evicted_tokens] indexes into [num_layers, num_blocks |, block_size]
const int* __restrict__ evicted_kv_count,       // [num_seqs, num_layers, num_kv_heads]
const int* __restrict__ context_lens,           // [num_seqs, num_kv_heads]
const int layer_idx,
const int max_parallel_kv,
const int num_layers,
const int num_kv_heads
*/

void set_inputs(
  int* evicted_kv_indices,
  int* evicted_kv_count,
  int* t1_block_tables,
  int* t2_block_tables,
  int* context_lens,
  int num_seqs,
  int num_blocks_per_seq,
  int num_layers,
  int num_kv_heads,
  int evicted_kvs_per_seq,
  int max_evicted_blocks,
  int block_size
) {
  //TODO k cache not being set properly for large block size
  const int num_blocks = num_blocks_per_seq * num_seqs * num_kv_heads;
  printf("num_blocks: %d\n", num_blocks);
  std::cout << "evicted_kv_indices" << std::endl;
  // [num_seqs, num_layers, num_kv_heads, max_evicted_blocks, BLOCK_SIZE]
  // evict every other KV
  set_incr_int_data(
    evicted_kv_indices, num_seqs * num_layers * num_kv_heads * num_blocks_per_seq / 2 * block_size,
    2, 1, num_blocks_per_seq * block_size / 2 // block offset
  );
  std::cout << "evicted_kv_count" << std::endl;
  set_const_int_data(
    evicted_kv_count, num_seqs * num_layers * num_kv_heads, num_blocks_per_seq / 2 * block_size);
  std::cout << "t1_block_tables" << std::endl;
  set_incr_int_data(
    t1_block_tables, num_seqs * num_blocks_per_seq, 1, 1, -1);
  std::cout << "t2_block_tables" << std::endl;
  set_incr_int_data_multi(
    t2_block_tables, num_seqs * num_blocks_per_seq * num_kv_heads,
    num_blocks_per_seq, 1, num_kv_heads,  // head offset
    1, num_kv_heads, num_blocks_per_seq * num_kv_heads,  // token offset
    0, 1, -1,
    0, 1, -1);
  std::cout << "context_lens" << std::endl;
  set_const_int_data(
    context_lens, num_seqs * num_kv_heads, num_blocks_per_seq * block_size);
}

int main(int argc, char** argv) {
  if (argc != 10) {
    std::cerr << "wrong number of arguments"  << std::endl;
    return 1;
  }

  int num_seqs;
  int num_layers;
  int num_kv_heads;
  int evicted_kvs_per_seq;
  int block_size;
  int num_blocks_per_seq;
  int layer_idx;

  sscanf(argv[1], "%d", &layer_idx);  
  sscanf(argv[2], "%d", &num_layers);
  sscanf(argv[3], "%d", &num_seqs);
  sscanf(argv[4], "%d", &num_kv_heads);
  sscanf(argv[5], "%d", &evicted_kvs_per_seq);
  sscanf(argv[6], "%d", &block_size);
  sscanf(argv[7], "%d", &num_blocks_per_seq);

  int max_evicted_blocks = DIVIDE_ROUND_UP(evicted_kvs_per_seq, block_size);
  int max_evicted_tokens = max_evicted_blocks * block_size;

  if (layer_idx >= num_layers) {
    std::cerr << "not enough layers for layer idx" << std::endl;
  }

  printf("args:\nlayer_idx=%d num_layers=%d num_seqs=%d num_kv_heads=%d evicted_kvs_per_seq=%d block_size=%d\nnum_blocks_per_seq=%d\n",
          layer_idx, num_layers, num_seqs, num_kv_heads, evicted_kvs_per_seq, block_size, num_blocks_per_seq);

  int* cache_moves_idx;
  int* cache_moves_count;
  int* evicted_kv_indices;
  int* evicted_kv_count;
  int* t1_block_tables;
  int* t2_block_tables;
  int* context_lens;
  cudaMalloc(&cache_moves_idx, sizeof(int)* num_seqs * num_kv_heads * max_evicted_tokens);
  cudaMalloc(&cache_moves_count, sizeof(int)* num_seqs * num_kv_heads);
  cudaMalloc(&evicted_kv_indices, sizeof(int)* num_seqs * num_layers * num_kv_heads * max_evicted_tokens);
  cudaMalloc(&evicted_kv_count, sizeof(int)* num_seqs * num_layers * num_kv_heads);
  cudaMalloc(&t1_block_tables, sizeof(int)* num_seqs * num_blocks_per_seq);
  cudaMalloc(&t2_block_tables, sizeof(int)* num_seqs * num_blocks_per_seq * num_kv_heads);
  cudaMalloc(&context_lens, sizeof(int)* num_seqs * num_kv_heads);
  set_inputs(
    evicted_kv_indices,
    evicted_kv_count,
    t1_block_tables,
    t2_block_tables,
    context_lens,
    num_seqs,
    num_blocks_per_seq,
    num_layers,
    num_kv_heads,
    evicted_kvs_per_seq,
    max_evicted_blocks,
    block_size
  );

  // index is into [num_layers, num_blocks, block_size]
  int max_num_blocks_per_seq = num_blocks_per_seq;

//   const int layer_idx,
// const int max_parallel_kv,
// const int num_layers,
// const int num_kv_heads

  dim3 grid(num_seqs);
  dim3 block(num_kv_heads);
  printf("num_seqs: %d, num_kv_heads: %d\n", num_seqs, num_kv_heads);
  KERNEL(block_size)

  int cache_moves_idx_cpu[num_seqs * num_kv_heads * max_evicted_blocks * block_size];
  cudaMemcpy(cache_moves_idx_cpu, cache_moves_idx, sizeof(int) * num_seqs * num_kv_heads * max_evicted_blocks * block_size, cudaMemcpyDeviceToHost);
  std::cout << "CACHE MOVES IDX:" << std::endl;
  print_output<int>(
    cache_moves_idx_cpu,
    num_seqs * num_kv_heads * max_evicted_blocks * block_size,
    num_kv_heads * max_evicted_blocks * block_size,
    max_evicted_blocks * block_size,
    block_size);
  int cache_moves_count_cpu[num_seqs * num_kv_heads];
  cudaMemcpy(cache_moves_count_cpu, cache_moves_count, sizeof(int) * num_seqs * num_kv_heads, cudaMemcpyDeviceToHost);
  std::cout << "CACHE MOVES COUNT:" << std::endl;
  print_output<int>(
    cache_moves_count_cpu,
    num_seqs * num_kv_heads,
    num_kv_heads,
    -1,
    -1);
  int evicted_kv_indices_cpu[num_seqs * num_layers * num_kv_heads * max_evicted_blocks * block_size];
  cudaMemcpy(evicted_kv_indices_cpu, evicted_kv_indices, sizeof(int) * num_seqs * num_layers * num_kv_heads * max_evicted_blocks * block_size, cudaMemcpyDeviceToHost);
  std::cout << "EVICTED_KV_INDICES:" << std::endl;
  print_output<int>(
    evicted_kv_indices_cpu,
    num_seqs * num_layers * num_kv_heads * max_evicted_blocks * block_size,
    num_layers * num_kv_heads * max_evicted_blocks * block_size,
    max_evicted_blocks * block_size,
    block_size);

  return 0;
}