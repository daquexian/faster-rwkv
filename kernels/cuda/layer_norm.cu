/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <assert.h>
#include <cub/cub.cuh>
#include <math_constants.h>

#include <kernels/cuda/layer_norm.cuh>
#include <kernels/registry.h>
#include <tensor.h>

namespace rwkv {

namespace cuda {

namespace layer_norm {

namespace {

template <typename SRC, typename DST, bool do_scale, bool do_center>
struct AffineStore {
  AffineStore(DST *y, int64_t row_size, const DST *gamma, const DST *beta)
      : y(y), row_size(row_size), gamma(gamma), beta(beta) {}
  template <int N>
  __device__ void store(const SRC *src, int64_t row, int64_t col) {
    cuda::layer_norm::Pack<DST, N> y_pack;
    cuda::layer_norm::Pack<DST, N> gamma_pack;
    cuda::layer_norm::Pack<DST, N> beta_pack;
    const int64_t offset = (row * row_size + col) / N;
    const int64_t gamma_offset = col / N;
    if (do_scale) {
      gamma_pack.storage = *(
          reinterpret_cast<const cuda::layer_norm::PackType<DST, N> *>(gamma) +
          gamma_offset);
    } else {
#pragma unroll
      for (int i = 0; i < N; ++i) {
        gamma_pack.elem[i] = static_cast<DST>(1.f);
      }
    }
    if (do_center) {
      beta_pack.storage =
          *(reinterpret_cast<const cuda::layer_norm::PackType<DST, N> *>(beta) +
            gamma_offset);
    } else {
#pragma unroll
      for (int i = 0; i < N; ++i) {
        beta_pack.elem[i] = static_cast<DST>(0.f);
      }
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
      DST normalized_i = static_cast<DST>(src[i]);
      if (do_scale || do_center) {
        y_pack.elem[i] = normalized_i * gamma_pack.elem[i] + beta_pack.elem[i];
      } else {
        y_pack.elem[i] = normalized_i;
      }
    }
    *(reinterpret_cast<cuda::layer_norm::PackType<DST, N> *>(y) + offset) =
        y_pack.storage;
  }
  DST *y;
  int64_t row_size;
  const DST *gamma;
  const DST *beta;
};
} // namespace

template <typename T>
void LayerNormForwardGpu(const int64_t num_instances, const int64_t norm_size,
                         const double epsilon, const T *x_ptr,
                         const T *gamma_ptr, const T *beta_ptr, T *y_ptr) {
  using ComputeType = typename cuda::layer_norm::DefaultComputeType<T>::type;
  cuda::layer_norm::DirectLoad<T, T> load(x_ptr, norm_size);
  AffineStore<ComputeType, T, true, true> store(y_ptr, norm_size, gamma_ptr,
                                                beta_ptr);
  // we have updated the OneFlow implementation for inference so that mean and
  // inv_variance are not needed
  cuda::layer_norm::DispatchLayerNorm<decltype(load), decltype(store),
                                      ComputeType>(nullptr, load, store,
                                                   num_instances, norm_size,
                                                   epsilon, nullptr, nullptr);
}

} // namespace layer_norm

Tensor layer_norm_op(const Tensor &x, const Tensor &weight,
                     const Tensor &bias) {
  Tensor out = Tensor::Empty(x.sizes(), x.dtype(), x.device());
  RV_CHECK(x.sizes().size() <= 2);
  int m, k;
  if (x.sizes().size() == 1) {
    m = 1;
    k = x.size(0);
  } else {
    m = x.size(0);
    k = x.size(1);
  }
  double epsilon = 1e-5;
  if (x.dtype() == weight.dtype() && x.dtype() == bias.dtype()) {
    if (x.dtype() == DType::kFloat32) {
      layer_norm::LayerNormForwardGpu(
          m, k, epsilon, x.data_ptr<float>(), weight.data_ptr<float>(),
          bias.data_ptr<float>(), out.data_ptr<float>());
    } else if (x.dtype() == DType::kFloat16) {
      layer_norm::LayerNormForwardGpu(
          m, k, epsilon, x.data_ptr<half>(), weight.data_ptr<half>(),
          bias.data_ptr<half>(), out.data_ptr<half>());
    } else {
      RV_CHECK(false);
    }
  } else {
    RV_CHECK(false);
  }
  return out;
}

KernelRegister layer_norm_reg("layernorm", Device::kCUDA, layer_norm_op);
} // namespace cuda

} // namespace rwkv
