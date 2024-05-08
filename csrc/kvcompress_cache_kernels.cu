#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"
#if defined(ENABLE_FP8_E5M2)
#include "quantization/fp8_e5m2_kvcache/quant_utils.cuh"
#elif defined(ENABLE_FP8_E4M3)
#include "quantization/fp8/amd_detail/quant_utils.cuh"
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

template<typename scalar_t, typename cache_t, bool is_fp8_kv_cache>
__global__ void reshape_and_cache_kernel(
  const scalar_t* __restrict__ key,           // [num_tokens, num_heads, head_size]
  const scalar_t* __restrict__ value,         // [num_tokens, num_heads, head_size]
  cache_t* __restrict__ key_cache,            // [num_blocks, num_heads, head_size/x, block_size, x]
  cache_t* __restrict__ value_cache,          // [num_blocks, num_heads, head_size, block_size]
  const int64_t* __restrict__ slot_mapping,   // [num_tokens]
  const int key_stride,
  const int value_stride,
  const int num_heads,
  const int head_size,
  const int block_size,
  const int x,
  const float kv_scale) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    // Padding token that should be ignored.
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t src_key_idx = token_idx * key_stride + i;
    const int64_t src_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    const int64_t tgt_key_idx = block_idx * num_heads * (head_size / x) * block_size * x
                                + head_idx * (head_size / x) * block_size * x
                                + x_idx * block_size * x
                                + block_offset * x
                                + x_offset;
    const int64_t tgt_value_idx = block_idx * num_heads * head_size * block_size
                                  + head_idx * head_size * block_size
                                  + head_offset * block_size
                                  + block_offset;
    scalar_t tgt_key = key[src_key_idx];
    scalar_t tgt_value = value[src_value_idx];
    if constexpr (is_fp8_kv_cache) {
#if defined(ENABLE_FP8_E5M2)
      key_cache[tgt_key_idx] = fp8_e5m2_unscaled::vec_conversion<uint8_t, scalar_t>(tgt_key);
      value_cache[tgt_value_idx] = fp8_e5m2_unscaled::vec_conversion<uint8_t, scalar_t>(tgt_value);
#elif defined(ENABLE_FP8_E4M3)
      key_cache[tgt_key_idx] = fp8_e4m3::scaled_vec_conversion<uint8_t, scalar_t>(tgt_key, kv_scale);
      value_cache[tgt_value_idx] = fp8_e4m3::scaled_vec_conversion<uint8_t, scalar_t>(tgt_value, kv_scale);
#else
      assert(false);
#endif
    } else {
      key_cache[tgt_key_idx] = tgt_key;
      value_cache[tgt_value_idx] = tgt_value;
    }
  }
}

} // namespace vllm

#define CALL_RESHAPE_AND_CACHE(KV_T, CACHE_T, IS_FP8_KV_CACHE)                                     \
  vllm::reshape_and_cache_kernel<KV_T, CACHE_T, IS_FP8_KV_CACHE><<<grid, block, 0, stream>>>(      \
    reinterpret_cast<KV_T*>(key.data_ptr()),                                                       \
    reinterpret_cast<KV_T*>(value.data_ptr()),                                                     \
    reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),                                              \
    reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),                                            \
    slot_mapping.data_ptr<int64_t>(),                                                              \
    key_stride,                                                                                    \
    value_stride,                                                                                  \
    num_heads,                                                                                     \
    head_size,                                                                                     \
    block_size,                                                                                    \
    x,                                                                                             \
    kv_scale);

void reshape_and_cache(
  torch::Tensor& key,           // [num_tokens, num_heads, head_size]
  torch::Tensor& value,         // [num_tokens, num_heads, head_size]
  torch::Tensor& key_cache,     // [num_blocks, num_heads, head_size/x, block_size, x]
  torch::Tensor& value_cache,   // [num_blocks, num_heads, head_size, block_size]
  torch::Tensor& slot_mapping,  // [num_tokens]
  const std::string& kv_cache_dtype,
  const float kv_scale)
{
  int num_tokens = key.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (kv_cache_dtype == "auto") {
    if (key.dtype() == at::ScalarType::Float) {
      CALL_RESHAPE_AND_CACHE(float, float, false);
    } else if (key.dtype() == at::ScalarType::Half) {
      CALL_RESHAPE_AND_CACHE(uint16_t, uint16_t, false);
    } else if (key.dtype() == at::ScalarType::BFloat16) {
      CALL_RESHAPE_AND_CACHE(__nv_bfloat16, __nv_bfloat16, false);
    }
  } else if (kv_cache_dtype == "fp8") {
    if (key.dtype() == at::ScalarType::Float) {
      CALL_RESHAPE_AND_CACHE(float, uint8_t, true);
    } else if (key.dtype() == at::ScalarType::Half) {
      CALL_RESHAPE_AND_CACHE(uint16_t, uint8_t, true);
    } else if (key.dtype() == at::ScalarType::BFloat16) {
      CALL_RESHAPE_AND_CACHE(__nv_bfloat16, uint8_t, true);
    }
  } else {
    TORCH_CHECK(false, "Unsupported data type of kv cache: ", kv_cache_dtype);
  }
}

namespace vllm {

template<typename Tout, typename Tin>
__global__ void convert_fp8_kernel(
  const Tin* __restrict__ src_cache,
  Tout* __restrict__ dst_cache,
  const int64_t block_stride) {
  const int64_t block_idx = blockIdx.x;
  for (int i = threadIdx.x; i < block_stride; i += blockDim.x) {
    int64_t idx = block_idx * block_stride + i;
#if defined(ENABLE_FP8_E5M2)
    dst_cache[idx] = fp8_e5m2_unscaled::vec_conversion<Tout, Tin>(src_cache[idx]);
#elif defined(ENABLE_FP8_E4M3)
    dst_cache[idx] = fp8_e4m3::vec_conversion<Tout, Tin>(src_cache[idx]);
#else
    assert(false);
#endif
  }
}

} // namespace vllm

#define CALL_CONVERT_FP8(Tout, Tin)                                 \
  vllm::convert_fp8_kernel<Tout, Tin><<<grid, block, 0, stream>>>(  \
    reinterpret_cast<Tin*>(src_cache.data_ptr()),                   \
    reinterpret_cast<Tout*>(dst_cache.data_ptr()),                  \
    block_stride);

void convert_fp8(
  torch::Tensor& src_cache,
  torch::Tensor& dst_cache)
{
  torch::Device src_device = src_cache.device();
  torch::Device dst_device = dst_cache.device();
  TORCH_CHECK(src_device.is_cuda(), "src must be on a GPU")
  TORCH_CHECK(dst_device.is_cuda(), "dst must be on a GPU")
  TORCH_CHECK(
    src_device.index() == dst_device.index(),
    "src and dst must be on the same GPU");
  at::cuda::OptionalCUDAGuard device_guard(src_device);

  int64_t num_blocks = src_cache.size(0);
  int64_t block_stride = src_cache.stride(0);

  dim3 grid(num_blocks);
  dim3 block(std::min(block_stride, int64_t(512)));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (src_cache.dtype() == at::ScalarType::Float) {
    CALL_CONVERT_FP8(uint8_t, float);
  } else if (src_cache.dtype() == at::ScalarType::Half) {
    CALL_CONVERT_FP8(uint8_t, uint16_t);
  } else if (src_cache.dtype() == at::ScalarType::BFloat16) {
    CALL_CONVERT_FP8(uint8_t, __nv_bfloat16);
  } else if (dst_cache.dtype() == at::ScalarType::Float) {
    CALL_CONVERT_FP8(float, uint8_t);
  } else if (dst_cache.dtype() == at::ScalarType::Half) {
    CALL_CONVERT_FP8(uint16_t, uint8_t);
  } else if (dst_cache.dtype() == at::ScalarType::BFloat16) {
    CALL_CONVERT_FP8(__nv_bfloat16, uint8_t);
  }
}
