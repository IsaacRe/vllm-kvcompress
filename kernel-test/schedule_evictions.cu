#include <iostream>
#define KERNEL_BLOCK_SIZE_TOTAL_KV_HEADS(BLOCK_SIZE, TOTAL_KV_HEADS) \
schedule_evictions<BLOCK_SIZE, TOTAL_KV_HEADS><<<1,num_seqs,sizeof(int)*num_seqs*num_layers*num_kv_heads>>>(\
  kv_idx,\
  kv_cnt,\
  sort_idx,\
  seq_blk_offsets,\
  layer_by_blk,\
  head_by_blk,\
  virtual_blk_num_by_block,\
  evicted_blks_per_seq,\
  context_lens,\
  hanging_token_count,\
  num_layers,\
  num_kv_heads,\
  total_blocks,\
  max_evicted_tokens);

#define KERNEL_BLOCK_SIZE(BLOCK_SIZE, TOTAL_KV_HEADS) \
  if (TOTAL_KV_HEADS <= 1) { \
    KERNEL_BLOCK_SIZE_TOTAL_KV_HEADS(BLOCK_SIZE, 1) \
  } else if (TOTAL_KV_HEADS <= 2) { \
    KERNEL_BLOCK_SIZE_TOTAL_KV_HEADS(BLOCK_SIZE, 2) \
  } else if (TOTAL_KV_HEADS <= 4) { \
    KERNEL_BLOCK_SIZE_TOTAL_KV_HEADS(BLOCK_SIZE, 4) \
  } else if (TOTAL_KV_HEADS <= 8) { \
    KERNEL_BLOCK_SIZE_TOTAL_KV_HEADS(BLOCK_SIZE, 8) \
  } else { \
    return 1; \
  }

#define KERNEL(BLOCK_SIZE, TOTAL_KV_HEADS) \
  switch (BLOCK_SIZE) { \
    case 1: \
      KERNEL_BLOCK_SIZE(1, TOTAL_KV_HEADS) \
      break; \
    case 2: \
      KERNEL_BLOCK_SIZE(2, TOTAL_KV_HEADS) \
      break; \
    case 4: \
      KERNEL_BLOCK_SIZE(4, TOTAL_KV_HEADS) \
      break; \
    case 8: \
      KERNEL_BLOCK_SIZE(8, TOTAL_KV_HEADS) \
      break; \
    default: \
      return 1; \
  }

template<int BLOCK_SIZE, int MAX_TOTAL_KV_HEADS> __global__ void schedule_evictions(
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
  const int num_layers,
  const int num_kv_heads,
  const int total_blocks,     // Total number of blocks across all layers, seqs, heads
  const int max_evicted_tokens) {
  const int num_seqs = gridDim.x * blockDim.x;
  const int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;  // allow block-level or thread-level parallelization (or both)

  const int output_head_stride = max_evicted_tokens;
  const int output_seq_stride = num_layers * num_kv_heads;

  const int output_head_offset = seq_idx * output_seq_stride;

  const int seq_start_offset = seq_block_offsets[seq_idx];

  printf("seq %d blk offset: %d\n", seq_idx, seq_start_offset);

  const int seq_end_offset = (seq_idx + 1 >= num_seqs) ? total_blocks : seq_block_offsets[seq_idx + 1];
  const int blocks_to_evict = evicted_blocks_per_seq[seq_idx];

  if (blocks_to_evict == 0) {
    return;
  }

  printf("seq %d evictions: %d\n", seq_idx, blocks_to_evict);
  
  /*
  The unsorted metrics array is structured as [num_seqs (ragged), num_layers (ragged), num_heads (ragged), num_blocks, BLOCK_SIZE]
  so we can use index / BLOCK_SIZE to get the block index in the unsorted array,
  then lookup layer/head index using the block index.
  */

  const int* seq_sorted_indices = sorted_indices + seq_start_offset * BLOCK_SIZE;

  const int seq_total_tokens = (seq_end_offset - seq_start_offset) * BLOCK_SIZE;
  printf("seq %d total toks: %d\n", seq_idx, seq_total_tokens);

  int evicted_block_count = 0;  // track number of evicted blocks for this sequence
  printf("seq %d tot kv heads: %d/%d\n", seq_idx, output_seq_stride, MAX_TOTAL_KV_HEADS);

  // use shared memory since MAX_TOTAL_KV_HEADS will max out registers
  extern __shared__ char shmem[];
  int* remaining_kv_count = reinterpret_cast<int*>(shmem);  // [num_seqs_per_block, MAX_TOTAL_KV_HEADS]
  const int thread_offset = threadIdx.x * MAX_TOTAL_KV_HEADS;

  //int remaining_kv_count[MAX_TOTAL_KV_HEADS];  // track number of number of KVs remaining in current block for each head
  for (int i = 0; i < output_seq_stride; ++i) {
    remaining_kv_count[thread_offset + i] = hanging_token_count[output_head_offset + i];
  }

  // Iterate over the merged sorted list (KVs for all layers of all heads for current sequence)
  for (int i = 0; i < seq_total_tokens; ++i) {
    const int token_idx = seq_sorted_indices[i];
    const int block_idx = token_idx / BLOCK_SIZE;
    const int layer_idx = layer_by_block[block_idx];
    const int head_idx = head_by_block[block_idx];
    const int virtual_block_num = virtual_block_num_by_block[block_idx];
    const int block_offset = token_idx % BLOCK_SIZE;
    const int virtual_token_idx = virtual_block_num * BLOCK_SIZE + block_offset;

    const int layer_head_idx = layer_idx * num_kv_heads + head_idx;
    const int output_kv_idx = output_head_offset + layer_head_idx;

    // Skip empty token slots in fragmented cache blocks
    if (virtual_token_idx >= context_lens[output_kv_idx]) {
      continue;
    }

    // Add to evicted KVs, incrementing the evicted KV count for the current head
    // Note: only the first ( (evicted_kv_count / BLOCK_SIZE) * BLOCK_SIZE ) KVs for each head will be evicted
    evicted_kv_indices[output_kv_idx * output_head_stride + evicted_kv_count[output_kv_idx]++] = virtual_token_idx;
    printf("loop %d, seq %d, token_idx %d, output_kv_idx %d, remaining: %d\n", i, seq_idx, token_idx, output_kv_idx, remaining_kv_count[layer_head_idx]);

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

/*
int* __restrict__ evicted_kv_indices,             // [num_seqs, num_layers, num_kv_heads, max_evicted_tokens]
int* __restrict__ evicted_kv_count,               // [num_seqs, num_layers, num_kv_heads]
const int* __restrict__ sorted_indices,           // [total_blocks * BLOCK_SIZE] sorted indices of concat([metrics_0, ..., metrics_N]) where metrics_i[j] is eviction metric for kv j%BLOCK_SIZE of block j/BLOCK_SIZE in sequence i
const int* __restrict__ seq_block_offsets,        // [num_seqs]  (offset into indices post-sort)
const int* __restrict__ layer_by_block,           // [total_blocks]  TODO: could use uint8
const int* __restrict__ head_by_block,            // [total_blocks]  TODO: could use uint8
const int* __restrict__ virtual_block_num_by_block,  // [total_blocks]
const int* __restrict__ evicted_blocks_per_seq,   // [num_seqs]
const int num_layers,
const int num_kv_heads,
const int total_blocks,     // Total number of blocks across all layers, seqs, heads
const int max_evicted_tokens
*/

void print_output(
  int* t,
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
  print_output(data, size, wrap, stride, -1);
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
  print_output(data, size, wrap, stride, -1);
  cudaMemcpy(ptr, data, size * sizeof(int), cudaMemcpyHostToDevice);
}


// void set_incr_int_data_strided(int* ptr, int size, int incr) {
//   int data[size];
//   for (int i = 0; i < size; ++i) {
//     data[i] = i * incr;
//   }
//   cudaMemcpy(ptr, data, size * sizeof(int), cudaMemcpyHostToDevice);
// }

void set_const_int_data(int* ptr, int size, int value) {
  int data[size];
  for (int i = 0; i < size; ++i) {
    data[i] = value;
  }
  cudaMemcpy(ptr, data, size * sizeof(int), cudaMemcpyHostToDevice);
}


void set_inputs(
  int* kv_idx,
  int* kv_cnt,
  int* sort_idx,
  int* seq_blk_offsets,
  int* layer_by_blk,
  int* head_by_blk,
  int* virtual_blk_num_by_block,
  int* evicted_blks_per_seq,
  int* context_lens,
  int* hanging_token_count,
  int num_seqs,
  int num_layers,
  int num_kv_heads,
  int max_evicted_blocks,
  int blocks_per_head,
  int block_size,
  int hanging_tokens
) {
  const int total_blocks = num_seqs * num_layers * num_kv_heads * blocks_per_head;
  std::cout << "sort_idx" << std::endl;
  set_incr_int_data_rev(sort_idx, total_blocks * block_size, 1, -1, total_blocks * block_size);
  std::cout << "seq_blk_offsets" << std::endl;
  set_incr_int_data(seq_blk_offsets, num_seqs, blocks_per_head * num_layers * num_kv_heads, -1, -1);
  std::cout << "layer_idxs" << std::endl;
  set_incr_int_data(layer_by_blk, total_blocks, 1, num_kv_heads * blocks_per_head, num_layers * num_kv_heads * blocks_per_head);
  std::cout << "head_idxs" << std::endl;
  set_incr_int_data(head_by_blk, total_blocks, 1, blocks_per_head, blocks_per_head * num_kv_heads);
  std::cout << "virtual_block_nums" << std::endl;
  set_incr_int_data(virtual_blk_num_by_block, total_blocks, 1, 1, blocks_per_head);
  std::cout << "tgt_evictions" << std::endl;
  set_const_int_data(evicted_blks_per_seq, num_seqs, max_evicted_blocks);
  std::cout << "context_lens" << std::endl;
  set_const_int_data(context_lens, num_seqs * num_layers * num_kv_heads, blocks_per_head * block_size - (block_size - hanging_tokens));
  std::cout << "hanging_token_count" << std::endl;
  set_const_int_data(hanging_token_count, num_seqs * num_layers * num_kv_heads, hanging_tokens);
}

// num_seqs, num_layers, num_kv_heads, max_evicted_blocks, blocks_per_head, block_size
int main(int argc, char** argv) {
  if (argc != 8) {
    std::cerr << "wrong number of arguments";
    return 1;
  }

  int num_seqs; // = 2; //2;
  int num_layers; // = 1; //8;
  int num_kv_heads; // = 1; //8;
  int max_evicted_blocks; // = 1; //50;
  int blocks_per_head; // = 1; //321;
  int block_size; // = 1; //32;
  int hanging_tokens;

  sscanf(argv[1], "%d", &num_seqs);
  sscanf(argv[2], "%d", &num_layers);
  sscanf(argv[3], "%d", &num_kv_heads);
  sscanf(argv[4], "%d", &max_evicted_blocks);
  sscanf(argv[5], "%d", &blocks_per_head);
  sscanf(argv[6], "%d", &block_size);
  sscanf(argv[7], "%d", &hanging_tokens);  

  printf("num_seqs=%d, num_layers=%d, num_kv_heads=%d, max_evicted_blocks=%d, blocks_per_head=%d, block_size=%d",
    num_seqs, num_layers, num_kv_heads, max_evicted_blocks, blocks_per_head, block_size);

  int total_kv_heads = num_layers * num_kv_heads;
  int total_blocks = num_seqs * num_layers * num_kv_heads * blocks_per_head;
  int max_evicted_tokens = max_evicted_blocks * block_size;

  int* kv_idx;
  int* kv_cnt;
  int* sort_idx;
  int* seq_blk_offsets;
  int* layer_by_blk;
  int* head_by_blk;
  int* virtual_blk_num_by_block;
  int* evicted_blks_per_seq;
  int* context_lens;
  int* hanging_token_count;
  cudaMalloc(&kv_idx, sizeof(int)* num_seqs * num_layers * num_kv_heads * max_evicted_blocks * block_size);
  cudaMalloc(&kv_cnt, sizeof(int)* num_seqs * num_layers * num_kv_heads);
  cudaMalloc(&sort_idx, sizeof(int)* total_blocks * block_size);
  cudaMalloc(&seq_blk_offsets, sizeof(int)* num_seqs);
  cudaMalloc(&layer_by_blk, sizeof(int)* total_blocks);
  cudaMalloc(&head_by_blk, sizeof(int)* total_blocks);
  cudaMalloc(&virtual_blk_num_by_block, sizeof(int) * total_blocks);
  cudaMalloc(&evicted_blks_per_seq, sizeof(int)* num_seqs);
  cudaMalloc(&context_lens, sizeof(int)* num_seqs * num_layers * num_kv_heads);
  cudaMalloc(&hanging_token_count, sizeof(int)* num_seqs * num_layers * num_kv_heads);
  set_inputs(
    kv_idx,
    kv_cnt,
    sort_idx,
    seq_blk_offsets,
    layer_by_blk,
    head_by_blk,
    virtual_blk_num_by_block,
    evicted_blks_per_seq,
    context_lens,
    hanging_token_count,
    num_seqs,
    num_layers,
    num_kv_heads,
    max_evicted_blocks,
    blocks_per_head,
    block_size,
    hanging_tokens
  );

  KERNEL(block_size, total_kv_heads)

  int idx_out_numel = num_seqs * num_layers * num_kv_heads * max_evicted_blocks * block_size;
  int cnt_out_numel = num_seqs * num_layers * num_kv_heads;
  int kv_idx_out[idx_out_numel];
  int kv_cnt_out[cnt_out_numel];

  cudaMemcpy(kv_idx_out, kv_idx, sizeof(int) * idx_out_numel, cudaMemcpyDeviceToHost);  //[num_seqs, num_layers, num_kv_heads, max_evicted_tokens]
  cudaMemcpy(kv_cnt_out, kv_cnt, sizeof(int) * cnt_out_numel, cudaMemcpyDeviceToHost);  //[num_seqs, num_layers, num_kv_heads]
  
  std::cout << "Evict idxs: [num_seqs, num_layers, num_kv_heads, max_evicted_blocks, BLOCK_SIZE]" << std::endl;
  print_output(
    kv_idx_out,
    idx_out_numel,
    num_layers * num_kv_heads * max_evicted_blocks * block_size,
    num_kv_heads * max_evicted_blocks * block_size,
    max_evicted_blocks * block_size
  );
  std::cout << "Evict cnts: [num_seqs, num_layers, num_kv_heads]" << std::endl;
  print_output(kv_cnt_out, cnt_out_numel, num_layers * num_kv_heads, num_kv_heads, -1);

  return 0;
}