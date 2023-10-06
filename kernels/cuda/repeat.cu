
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
__global__ void _repeat(LengthType dst_total, LengthType offset, const T *src,
                        T *dst, LengthType *dst_shape, LengthType *src_shape) {
  LengthType indices[ndim];
  for (LengthType i = blockIdx.x * blockDim.x + threadIdx.x + offset;
       i < dst_total; i += blockDim.x * gridDim.x) {
    ::cuda::offset_to_indices(i, dst_shape, indices, dst_total, ndim);
    for (int j = 0; j < ndim; j++) {
      indices[j] = indices[j] % src_shape[j];
    }
    LengthType src_offset = ::cuda::indices_to_offset(src_shape, indices, ndim);
    dst[i] = src[src_offset];
  }
}

template <int ndim>
Tensor repeat_internal(const Tensor &x,
                       const std::vector<LengthType> &repeats) {
  auto deduce_shape = [&x, &repeats]() {
    Shape x_shape = x.shape();
    RV_CHECK(repeats.size() >= x_shape.size());
    auto max_dims = std::max(x_shape.size(), repeats.size());
    Shape res(max_dims, 1);
    for (int i = 0; i < max_dims; i++) {
      auto x_dim = x_shape.size() > i ? x_shape[x_shape.size() - i - 1] : 1;
      auto r_dim = repeats.size() > i ? repeats[repeats.size() - i - 1] : 1;
      res[max_dims - i - 1] = x_dim * r_dim;
    }
    return res;
  };

  std::vector<LengthType> src_shape(x.shape());
  for (int i = src_shape.size(); i < repeats.size(); i++) {
    src_shape.insert(src_shape.begin(), 1);
  }
  auto dst_shape = deduce_shape();
  Tensor dst = Tensor::Empty(dst_shape, x.dtype(), x.device());
  LengthType dst_total = dst.numel();

  LengthType *src_shape_gpu;
  LengthType *dst_shape_gpu;
  cudaMalloc(&src_shape_gpu, ndim * sizeof(LengthType));
  cudaMalloc(&dst_shape_gpu, ndim * sizeof(LengthType));
  cudaMemcpy(src_shape_gpu, src_shape.data(), ndim * sizeof(LengthType),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dst_shape_gpu, dst_shape.data(), ndim * sizeof(LengthType),
             cudaMemcpyHostToDevice);

#define LAUNCH_REPEAT_KERNEL(type)                                             \
  FR_LAUNCH_CUDA_KERNEL_BASE_256(_repeat, type, ndim, dst_total,               \
                                 x.data_ptr<type>(), dst.data_ptr<type>(),     \
                                 dst_shape_gpu, src_shape_gpu);

  if (x.dtype() == DType::kFloat32) {
    LAUNCH_REPEAT_KERNEL(float);
  } else if (x.dtype() == DType::kFloat16) {
    LAUNCH_REPEAT_KERNEL(half);
  } else if (x.dtype() == DType::kInt8) {
    LAUNCH_REPEAT_KERNEL(int8_t);
  } else {
    RV_UNIMPLEMENTED();
  }

#undef LAUNCH_REPEAT_KERNEL

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("repeat CUDA error: %s\n", cudaGetErrorString(error));
  }

  cudaFree(src_shape_gpu);
  cudaFree(dst_shape_gpu);

  return dst;
} // namespace cuda

Tensor repeat(const Tensor &x, const std::vector<LengthType> &repeats) {
  int ndim = repeats.size();
  if (ndim == 1) {
    return repeat_internal<1>(x, repeats);
  } else if (ndim == 2) {
    return repeat_internal<2>(x, repeats);
  } else if (ndim == 3) {
    return repeat_internal<3>(x, repeats);
  } else if (ndim == 4) {
    return repeat_internal<4>(x, repeats);
  } else {
    RV_UNIMPLEMENTED();
  }
}

KernelRegister repeat_reg_gpu("repeat", Device::kCUDA, repeat);

} // namespace cuda
} // namespace rwkv
