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
__global__ void _flip(LengthType total_elems, LengthType offset, const T *src,
                      T *dst, const LengthType *shape, int flipped_ndim,
                      const LengthType *dims) {
  LengthType src_idx[ndim];

  for (LengthType i = blockIdx.x * blockDim.x + threadIdx.x + offset;
       i < total_elems; i += blockDim.x * gridDim.x) {
    ::cuda::offset_to_indices(i, shape, src_idx, total_elems, ndim);
    for (LengthType j = 0; j < flipped_ndim; j++) {
      auto dim = dims[j];
      src_idx[dim] = shape[dim] - src_idx[dim] - 1;
    }
    LengthType src_pos = ::cuda::indices_to_offset(shape, src_idx, ndim);
    dst[i] = src[src_pos];
  }
}

template <int ndim>
Tensor flip_internal(const Tensor &x, const std::vector<LengthType> &dims) {
  Shape shape = Shape(x.shape());
  Tensor dst = Tensor::Empty(shape, x.dtype(), x.device());
  auto total_elems = x.numel();
  auto flipped_ndim = dims.size();

  LengthType *src_shape_gpu;
  LengthType *dims_gpu;
  cudaMalloc(&src_shape_gpu, ndim * sizeof(LengthType));
  cudaMalloc(&dims_gpu, flipped_ndim * sizeof(LengthType));
  cudaMemcpy(src_shape_gpu, shape.data(), ndim * sizeof(LengthType),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dims_gpu, dims.data(), flipped_ndim * sizeof(LengthType),
             cudaMemcpyHostToDevice);

#define LAUNCH_FLIP_KERNEL(type)                                               \
  FR_LAUNCH_CUDA_KERNEL_BASE_256(_flip, type, ndim, total_elems,               \
                                 x.data_ptr<type>(), dst.data_ptr<type>(),     \
                                 src_shape_gpu, flipped_ndim, dims_gpu);

  if (x.dtype() == DType::kFloat32) {
    LAUNCH_FLIP_KERNEL(float)
  } else if (x.dtype() == DType::kFloat16) {
    LAUNCH_FLIP_KERNEL(half)
  } else if (x.dtype() == DType::kInt8) {
    LAUNCH_FLIP_KERNEL(int8_t)
  } else {
    RV_CHECK(false);
  }

  cudaFree(src_shape_gpu);
  cudaFree(dims_gpu);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("flip CUDA error: %s\n", cudaGetErrorString(error));
  }

  return dst;
}

Tensor flip(const Tensor &x, const std::vector<LengthType> &dims) {
  int ndim = x.sizes().size();
  if (ndim == 1) {
    return flip_internal<1>(x, dims);
  } else if (ndim == 2) {
    return flip_internal<2>(x, dims);
  } else if (ndim == 3) {
    return flip_internal<3>(x, dims);
  } else if (ndim == 4) {
    return flip_internal<4>(x, dims);
  } else {
    RV_UNIMPLEMENTED();
  }
}

KernelRegister flip_reg_cuda("flip", Device::kCUDA, flip);

} // namespace cuda
} // namespace rwkv
