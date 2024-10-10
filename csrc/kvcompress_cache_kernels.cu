#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"

#ifdef USE_ROCM
  #include "quantization/fp8/amd/quant_utils.cuh"
#else
  #include "quantization/fp8/nvidia/quant_utils.cuh"
#endif

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

#ifdef USE_ROCM
  #include <hip/hip_bf16.h>
  typedef __hip_bfloat16 __nv_bfloat16;
#endif


namespace vllm {

template<typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void single_tier_reshape_and_cache_kernel(
  const scalar_t* __restrict__ key,           // [num_tokens, num_heads, head_size]
  const scalar_t* __restrict__ value,         // [num_tokens, num_heads, head_size]
  cache_t* __restrict__ key_cache,            // [num_blocks, head_size/x, block_size, x]
  cache_t* __restrict__ value_cache,          // [num_blocks, head_size, block_size]
  float* __restrict__ kv_metrics,             // [num_blocks, block_size]
  const int64_t* __restrict__ slot_mapping,   // [num_tokens, num_heads]
  const float* __restrict__ kv_metric_head_bias,  // [num_heads]
  const int key_stride,
  const int value_stride,
  const int num_heads,
  const int head_size,
  const int block_size,
  const int x,
  const float k_scale,
  const float v_scale) {
  const int64_t token_idx = blockIdx.x;

  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int head_idx = i / head_size;
    const int64_t slot_idx = slot_mapping[token_idx * num_heads + head_idx];
    if (slot_idx < 0) {
      // Padding token that should be ignored.
      continue;
    }

    // Initialize metric for current KV to the bias for its head
    if (i % head_size == 0) {
      kv_metrics[slot_idx] = kv_metric_head_bias[head_idx];
    }

    const int64_t block_idx = slot_idx / block_size;
    const int64_t block_offset = slot_idx % block_size;

    const int64_t src_key_idx = token_idx * key_stride + i;
    const int64_t src_value_idx = token_idx * value_stride + i;

    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    const int64_t tgt_key_idx = block_idx * (head_size / x) * block_size * x
                                + x_idx * block_size * x
                                + block_offset * x
                                + x_offset;
    const int64_t tgt_value_idx = block_idx * head_size * block_size
                                  + head_offset * block_size
                                  + block_offset;
    scalar_t tgt_key = key[src_key_idx];
    scalar_t tgt_value = value[src_value_idx];
    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      key_cache[tgt_key_idx] = tgt_key;
      value_cache[tgt_value_idx] = tgt_value;
    } else {
      key_cache[tgt_key_idx] =
          fp8::scaled_convert<cache_t, scalar_t, kv_dt>(tgt_key, k_scale);
      value_cache[tgt_value_idx] =
          fp8::scaled_convert<cache_t, scalar_t, kv_dt>(tgt_value, v_scale);
    }
  }
}

} // namespace vllm

#define CALL_KVC_RESHAPE_AND_CACHE(KV_T, CACHE_T, KV_DTYPE)                                     \
  vllm::single_tier_reshape_and_cache_kernel<KV_T, CACHE_T, KV_DTYPE><<<grid, block, 0, stream>>>(      \
    reinterpret_cast<KV_T*>(key.data_ptr()),                                                       \
    reinterpret_cast<KV_T*>(value.data_ptr()),                                                     \
    reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),                                              \
    reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),                                            \
    kv_metrics.data_ptr<float>(),                                                                  \
    slot_mapping.data_ptr<int64_t>(),                                                              \
    kv_metric_head_bias.data_ptr<float>(),                                                         \
    key_stride,                                                                                    \
    value_stride,                                                                                  \
    num_heads,                                                                                     \
    head_size,                                                                                     \
    block_size,                                                                                    \
    x,                                                                                             \
    k_scale,                                                                                       \
    v_scale);

void kvcompress_reshape_and_cache(
  torch::Tensor& key,           // [num_tokens, num_heads, head_size]
  torch::Tensor& value,         // [num_tokens, num_heads, head_size]
  torch::Tensor& key_cache,     // [num_blocks, head_size/x, block_size, x]
  torch::Tensor& value_cache,   // [num_blocks, head_size, block_size]
  torch::Tensor& kv_metrics,    // [num_blocks, block_size]
  torch::Tensor& slot_mapping,  // [num_tokens, num_heads]
  torch::Tensor& kv_metric_head_bias, // [num_heads]
  const std::string& kv_cache_dtype,
  const double k_scale,
  const double v_scale)
{
  int num_tokens = key.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(2);
  int x = key_cache.size(3);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype,
                             CALL_KVC_RESHAPE_AND_CACHE)
}
