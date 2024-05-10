#pragma once

#include <torch/extension.h>

void kvcompress_reshape_and_cache(
  torch::Tensor& key,
  torch::Tensor& value,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  torch::Tensor& kv_metrics,
  torch::Tensor& slot_mapping,
  torch::Tensor& kv_metric_head_bias,
  const std::string& kv_cache_dtype,
  const float kv_scale);
