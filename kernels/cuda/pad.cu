
#include "check.h"
#include "element_wise.cuh"
#include "util.cuh"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <kernels/macro.h>
#include <kernels/registry.h>
#include <numeric>
#include <tensor.h>
#include <vector>

namespace rwkv {
namespace cuda {

template <typename T, int ndim>
__global__ void _pad(LengthType total_elems, LengthType offset, const T *src,
                     T *dst, const LengthType *src_shape,
                     const LengthType *dst_shape, const LengthType *paddings,
                     T value) {
  LengthType src_idx[ndim];
  LengthType dst_idx[ndim];

  for (LengthType i = blockIdx.x * blockDim.x + threadIdx.x + offset;
       i < total_elems; i += blockDim.x * gridDim.x) {
    ::cuda::offset_to_indices(i, dst_shape, dst_idx, total_elems, ndim);
    bool to_pad = false;
    for (int j = 0; j < ndim; j++) {
      auto before = paddings[(ndim - j - 1) * 2];
      if (dst_idx[j] < before || dst_idx[j] >= src_shape[j] + before) {
        to_pad = true;
      } else {
        src_idx[j] = dst_idx[j] - before;
      }
    }

    if (!to_pad) {
      dst[i] = src[::cuda::indices_to_offset(src_shape, src_idx, ndim)];
    } else {
      dst[i] = value;
    }
  }
}

template <int ndim>
Tensor pad_internal(const Tensor &x, const std::vector<LengthType> &paddings,
                    const std::string &mode) {
  RV_CHECK(mode == "constant");
  RV_CHECK(paddings.size() % 2 == 0);
  auto src_shape = x.shape();
  std::vector<LengthType> paddings_vec;
  auto deduce_shape = [&src_shape, &paddings, &paddings_vec]() {
    Shape res(src_shape);
    for (int i = paddings.size(); i < res.size() * 2; i++) {
      paddings_vec.push_back(0);
    }
    for (int i = paddings.size() - 1; i > 0; i -= 2) {
      paddings_vec.push_back(paddings[i - 1]);
      paddings_vec.push_back(paddings[i]);
    }
    RV_CHECK(paddings_vec.size() == 2 * res.size());
    for (int i = 0; i < res.size(); i++) {
      RV_CHECK(paddings_vec[i * 2] >= 0 && paddings_vec[i * 2 + 1] >= 0);
      res[i] += paddings_vec[i * 2] + paddings_vec[i * 2 + 1];
    }
    return res;
  };

  Shape dst_shape = deduce_shape();
  Tensor dst = Tensor::Empty(dst_shape, x.dtype(), x.device());
  auto total_elems = dst.numel();

  LengthType *src_shape_gpu;
  LengthType *dst_shape_gpu;
  LengthType *paddings_gpu;
  cudaMalloc(&src_shape_gpu, ndim * sizeof(LengthType));
  cudaMalloc(&dst_shape_gpu, ndim * sizeof(LengthType));
  cudaMalloc(&paddings_gpu, paddings_vec.size() * sizeof(LengthType));
  cudaMemcpy(src_shape_gpu, src_shape.data(), ndim * sizeof(LengthType),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dst_shape_gpu, dst_shape.data(), ndim * sizeof(LengthType),
             cudaMemcpyHostToDevice);
  cudaMemcpy(paddings_gpu, paddings_vec.data(),
             paddings_vec.size() * sizeof(LengthType), cudaMemcpyHostToDevice);

#define LAUNCH_PAD_KERNEL(type, value)                                         \
  FR_LAUNCH_CUDA_KERNEL_BASE_256(                                              \
      _pad, type, ndim, total_elems, x.data_ptr<type>(), dst.data_ptr<type>(), \
      src_shape_gpu, dst_shape_gpu, paddings_gpu, value);

  if (x.dtype() == DType::kFloat32) {
    LAUNCH_PAD_KERNEL(float, .0f)
  } else if (x.dtype() == DType::kFloat16) {
    LAUNCH_PAD_KERNEL(half, static_cast<half>(.0f))
  } else if (x.dtype() == DType::kInt8) {
    LAUNCH_PAD_KERNEL(int8_t, static_cast<int8_t>(0))
  } else {
    RV_UNIMPLEMENTED();
  }

  cudaFree(src_shape_gpu);
  cudaFree(dst_shape_gpu);
  cudaFree(paddings_gpu);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("pad CUDA error: %s\n", cudaGetErrorString(error));
  }

  return dst;
}

Tensor pad(const Tensor &x, const std::vector<LengthType> &paddings,
           const std::string &mode) {
  int ndim = x.sizes().size();
  if (ndim == 1) {
    return pad_internal<1>(x, paddings, mode);
  } else if (ndim == 2) {
    return pad_internal<2>(x, paddings, mode);
  } else if (ndim == 3) {
    return pad_internal<3>(x, paddings, mode);
  } else if (ndim == 4) {
    return pad_internal<4>(x, paddings, mode);
  } else {
    RV_UNIMPLEMENTED();
  }
}

KernelRegister pad_reg_cuda("pad", Device::kCUDA, pad);

} // namespace cuda
} // namespace rwkv
