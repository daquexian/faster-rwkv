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

#include <tensor.h>
#include <kernels/registry.h>
#include <kernels/cuda/layer_norm.cuh>

namespace rwkv {

namespace cuda {

namespace group_norm {

template<typename SRC, typename DST, bool affine>
struct AffineStore {
  AffineStore(DST* y, int64_t row_size, int64_t channel_size, int64_t spatial_size,
              const DST* gamma, const DST* beta)
      : y(y),
        row_size(row_size),
        channel_size(channel_size),
        spatial_size(spatial_size),
        gamma(gamma),
        beta(beta){}

  template<int PackSize>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    cuda::layer_norm::Pack<DST, PackSize> y_pack;
    const int64_t offset = row * row_size + col;
    const int64_t packed_offset = offset / PackSize;
    const int64_t gamma_beta_offset = (offset / spatial_size) % channel_size;
    DST gamma_val = 1.0;
    DST beta_val = 0.0;
    if (affine) {
      gamma_val = gamma[gamma_beta_offset];
      beta_val = beta[gamma_beta_offset];
    }

#pragma unroll
    for (int i = 0; i < PackSize; ++i) {
      DST normalized_i = static_cast<DST>(src[i]);
      if (affine) {
        y_pack.elem[i] = normalized_i * gamma_val + beta_val;
      } else {
        // Direct Store.
        y_pack.elem[i] = normalized_i;
      }
    }
    *(reinterpret_cast<cuda::layer_norm::PackType<DST, PackSize>*>(y) + packed_offset) =
        y_pack.storage;
  }
  bool CanPackAs(size_t pack_size) { return (spatial_size % pack_size) == 0; }
  DST* y;
  int64_t row_size;
  int64_t channel_size;
  int64_t spatial_size;
  const DST* gamma;
  const DST* beta;
};

template<typename SRC, typename DST, bool affine>
struct ScaleLoad {
  using LoadType = DST;
  ScaleLoad(const SRC* src, const SRC* gamma, int64_t row_size, int64_t channel_size,
            int64_t spatial_size)
      : src(src),
        gamma(gamma),
        row_size(row_size),
        channel_size(channel_size),
        spatial_size(spatial_size) {}
  template<int PackSize>
  __device__ void load(DST* dst, int64_t row, int64_t col) const {
    cuda::layer_norm::Pack<SRC, PackSize> src_pack;
    cuda::layer_norm::Pack<SRC, PackSize> gamma_pack;

    const int64_t offset = row * row_size + col;
    const int64_t packed_offset = offset / PackSize;
    const int64_t gamma_offset = (offset / spatial_size) % channel_size;

    src_pack.storage =
        *(reinterpret_cast<const cuda::layer_norm::PackType<SRC, PackSize>*>(src) + packed_offset);
    SRC gamma_val = static_cast<SRC>(1.0);
    if (affine) { gamma_val = gamma[gamma_offset]; }
#pragma unroll
    for (int i = 0; i < PackSize; ++i) { dst[i] = static_cast<DST>(src_pack.elem[i] * gamma_val); }
  }
  bool CanPackAs(size_t pack_size) { return (spatial_size % pack_size) == 0; }
  const SRC* src;
  const SRC* gamma;
  int64_t row_size;
  int64_t channel_size;
  int64_t spatial_size;
};

#ifdef WITH_CUTLASS

template<typename SRC, typename DST, bool affine>
struct ChannelsLastStore {
  ChannelsLastStore(DST* y, const DST* gamma, const DST* beta, int64_t spatial_size,
                    int64_t channel_size, int64_t num_groups)
      : y(y),
        gamma(gamma),
        beta(beta),
        spatial_size(spatial_size),
        c0(num_groups),
        c1(channel_size / num_groups) {}

  template<int PackSize>
  __device__ void store(const SRC* src, int32_t row, int32_t col) {
    cuda::layer_norm::Pack<DST, PackSize> y_pack;
    cuda::layer_norm::Pack<DST, PackSize> gamma_pack;
    cuda::layer_norm::Pack<DST, PackSize> beta_pack;
    int32_t spatial_idx;
    int32_t c1_idx;
    c1(spatial_idx, c1_idx, col);
    int32_t batch_idx;
    int32_t c0_idx;
    c0(batch_idx, c0_idx, row);
    const int32_t y_offset =
        (batch_idx * c0.divisor * c1.divisor * spatial_size + spatial_idx * c0.divisor * c1.divisor
         + c0_idx * c1.divisor + c1_idx)
        / PackSize;
    const int32_t gamma_beta_offset = (c0_idx * c1.divisor + c1_idx) / PackSize;
    if (affine) {
      gamma_pack.storage =
          *(reinterpret_cast<const cuda::layer_norm::PackType<DST, PackSize>*>(gamma)
            + gamma_beta_offset);
      beta_pack.storage = *(reinterpret_cast<const cuda::layer_norm::PackType<DST, PackSize>*>(beta)
                            + gamma_beta_offset);
    }

#pragma unroll
    for (int i = 0; i < PackSize; ++i) {
      DST normalized_i = static_cast<DST>(src[i]);
      if (affine) {
        y_pack.elem[i] = normalized_i * gamma_pack.elem[i] + beta_pack.elem[i];
      } else {
        // Direct Store.
        y_pack.elem[i] = normalized_i;
      }
    }
    *(reinterpret_cast<cuda::layer_norm::PackType<DST, PackSize>*>(y) + y_offset) = y_pack.storage;
  }
  bool CanPackAs(size_t pack_size) { return (c1.divisor % pack_size) == 0; }
  DST* y;
  const DST* gamma;
  const DST* beta;
  int32_t spatial_size;
  cutlass::FastDivmod c0;
  cutlass::FastDivmod c1;
};

template<typename SRC, typename DST>
struct ChannelsLastLoad {
  using LoadType = DST;
  ChannelsLastLoad(const SRC* src, int64_t spatial_size, int64_t channel_size, int64_t num_groups)
      : src(src), spatial_size(spatial_size), c0(num_groups), c1(channel_size / num_groups) {}
  template<int N>
  __device__ void load(DST* dst, int32_t row, int32_t col) const {
    int32_t spatial_idx;
    int32_t c1_idx;
    c1(spatial_idx, c1_idx, col);
    int32_t batch_idx;
    int32_t c0_idx;
    c0(batch_idx, c0_idx, row);
    cuda::layer_norm::Pack<SRC, N> pack;
    const int32_t offset = (batch_idx * c0.divisor * c1.divisor * spatial_size
                            + spatial_idx * c0.divisor * c1.divisor + c0_idx * c1.divisor + c1_idx)
                           / N;

    pack.storage = *(reinterpret_cast<const cuda::layer_norm::PackType<SRC, N>*>(src) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) { dst[i] = static_cast<DST>(pack.elem[i]); }
  }
  bool CanPackAs(size_t pack_size) { return (c1.divisor % pack_size) == 0; }
  const SRC* src;
  int32_t spatial_size;
  cutlass::FastDivmod c0;
  cutlass::FastDivmod c1;
};

#else

template<typename SRC, typename DST, bool affine>
struct ChannelsLastStore {
  ChannelsLastStore(DST* y, const DST* gamma, const DST* beta, int64_t spatial_size,
                    int64_t channel_size, int64_t num_groups)
      : y(y),
        gamma(gamma),
        beta(beta),
        spatial_size(spatial_size),
        c0(num_groups),
        c1(channel_size / num_groups) {}

  template<int PackSize>
  __device__ void store(const SRC* src, int32_t row, int32_t col) {
    cuda::layer_norm::Pack<DST, PackSize> y_pack;
    cuda::layer_norm::Pack<DST, PackSize> gamma_pack;
    cuda::layer_norm::Pack<DST, PackSize> beta_pack;
    int32_t spatial_idx = col / c1;
    int32_t c1_idx = col - spatial_idx * c1;
    int32_t batch_idx = row / c0;
    int32_t c0_idx = row - batch_idx * c0;
    const int32_t y_offset =
        (batch_idx * c0 * c1 * spatial_size + spatial_idx * c0 * c1 + c0_idx * c1 + c1_idx)
        / PackSize;
    const int32_t gamma_beta_offset = (c0_idx * c1 + c1_idx) / PackSize;
    if (affine) {
      gamma_pack.storage =
          *(reinterpret_cast<const cuda::layer_norm::PackType<DST, PackSize>*>(gamma)
            + gamma_beta_offset);
      beta_pack.storage = *(reinterpret_cast<const cuda::layer_norm::PackType<DST, PackSize>*>(beta)
                            + gamma_beta_offset);
    }

#pragma unroll
    for (int i = 0; i < PackSize; ++i) {
      DST normalized_i = static_cast<DST>(src[i]);
      if (affine) {
        y_pack.elem[i] = normalized_i * gamma_pack.elem[i] + beta_pack.elem[i];
      } else {
        // Direct Store.
        y_pack.elem[i] = normalized_i;
      }
    }
    *(reinterpret_cast<cuda::layer_norm::PackType<DST, PackSize>*>(y) + y_offset) = y_pack.storage;
  }
  bool CanPackAs(size_t pack_size) { return (c1 % pack_size) == 0; }
  DST* y;
  const DST* gamma;
  const DST* beta;
  int32_t spatial_size;
  int32_t c0;
  int32_t c1;
};

template<typename SRC, typename DST>
struct ChannelsLastLoad {
  using LoadType = DST;
  ChannelsLastLoad(const SRC* src, int64_t spatial_size, int64_t channel_size, int64_t num_groups)
      : src(src), spatial_size(spatial_size), c0(num_groups), c1(channel_size / num_groups) {}
  template<int N>
  __device__ void load(DST* dst, int32_t row, int32_t col) const {
    int32_t spatial_idx = col / c1;
    int32_t c1_idx = col - spatial_idx * c1;
    int32_t batch_idx = row / c0;
    int32_t c0_idx = row - batch_idx * c0;
    cuda::layer_norm::Pack<SRC, N> pack;
    const int32_t offset =
        (batch_idx * c0 * c1 * spatial_size + spatial_idx * c0 * c1 + c0_idx * c1 + c1_idx) / N;

    pack.storage = *(reinterpret_cast<const cuda::layer_norm::PackType<SRC, N>*>(src) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) { dst[i] = static_cast<DST>(pack.elem[i]); }
  }
  bool CanPackAs(size_t pack_size) { return (c1 % pack_size) == 0; }
  const SRC* src;
  int32_t spatial_size;
  int32_t c0;
  int32_t c1;
};

#endif  // WITH_CUTLASS

template<typename T>
void GroupNormForwardGpu(const int64_t num_instances, const int64_t norm_size,
                         const int64_t channel_size, const int64_t spatial_size,
                         const double epsilon, const T* x_ptr, const T* gamma_ptr,
                         const T* beta_ptr, T* y_ptr, bool channels_first) {
  using ComputeType = typename cuda::layer_norm::DefaultComputeType<T>::type;
  if (channels_first) {
    cuda::layer_norm::DirectLoad<T, T> load(x_ptr, norm_size);
    AffineStore<ComputeType, T, true> store(y_ptr, norm_size, channel_size,
                                                          spatial_size, gamma_ptr, beta_ptr);

    cuda::layer_norm::DispatchLayerNorm<decltype(load), decltype(store), ComputeType>(
         nullptr, load, store, num_instances, norm_size, epsilon, nullptr, nullptr);
  } else {
    ChannelsLastLoad<T, T> load(x_ptr, spatial_size, channel_size,
                                channel_size / (norm_size / spatial_size));
    ChannelsLastStore<ComputeType, T, true> store(
        y_ptr, gamma_ptr, beta_ptr, spatial_size, channel_size,
        channel_size / (norm_size / spatial_size));

    cuda::layer_norm::DispatchLayerNorm<decltype(load), decltype(store), ComputeType>(
        nullptr, load, store, num_instances, norm_size, epsilon,
        nullptr, nullptr);
  }
}


} // namespace group_norm

Tensor group_norm_op(const Tensor& x, int num_groups, const Tensor& weight, const Tensor& bias) {
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
  auto instance_num = num_groups;
  auto norm_size = k / num_groups;
  auto channel_size = k;
  auto spatial_size = m;
  double epsilon = 1e-5;
  if (x.dtype() == weight.dtype() && x.dtype() == bias.dtype()) {
    if (x.dtype() == DType::kFloat32) {
      group_norm::GroupNormForwardGpu(
          instance_num, norm_size, channel_size, spatial_size, epsilon, x.data_ptr<float>(), weight.data_ptr<float>(),
          bias.data_ptr<float>(), out.data_ptr<float>(), true);
    } else if (x.dtype() == DType::kFloat16) {
      group_norm::GroupNormForwardGpu(
          instance_num, norm_size, channel_size, spatial_size, epsilon, x.data_ptr<half>(), weight.data_ptr<half>(),
          bias.data_ptr<half>(), out.data_ptr<half>(), true);
    } else {
      RV_CHECK(false);
    }
  } else {
    RV_CHECK(false);
  }
  return out;
}

KernelRegister group_norm_reg("groupnorm", Device::kCUDA, group_norm_op);

} // namespace cuda
} // namespace rwkv