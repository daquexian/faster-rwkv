
#include "check.h"
#include "element_wise.cuh"
#include "util.cuh"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <kernels/registry.h>
#include <numeric>
#include <tensor.h>
#include <vector>

namespace rwkv {
namespace cuda {
template <typename T, int ndim>
__global__ void _transpose(const T *src, T *dst, int dim_a, int dim_b,
                           const LengthType *src_shape,
                           const LengthType *dst_shape,
                           LengthType total_elems) {
  LengthType src_idx[ndim];
  LengthType dst_idx[ndim];
  for (LengthType i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elems;
       i += blockDim.x * gridDim.x) {
    ::cuda::offset_to_indices(i, dst_shape, dst_idx, total_elems, ndim);
    for (int j = 0; j < ndim; j++) {
      src_idx[j] = dst_idx[j];
    }
    src_idx[dim_b] = dst_idx[dim_a];
    src_idx[dim_a] = dst_idx[dim_b];
    LengthType src_pos = ::cuda::indices_to_offset(src_shape, src_idx, ndim);
    dst[i] = src[src_pos];
  }
}

template <int ndim>
Tensor transpose_internal(const Tensor &x, int dim_a, int dim_b) {
  if (dim_a < 0)
    dim_a += ndim;
  if (dim_b < 0)
    dim_b += ndim;
  RV_CHECK(dim_a >= 0 && dim_b >= 0)
  RV_CHECK(dim_a < ndim && dim_b < ndim)
  auto deduce_shape = [&x, &dim_a, &dim_b]() {
    auto shape = Shape(x.shape());
    std::swap(shape[dim_a], shape[dim_b]);
    return shape;
  };
  Shape src_shape = x.shape();
  Shape dst_shape = deduce_shape();
  Tensor dst = Tensor::Empty(dst_shape, x.dtype(), x.device());
  auto total_elems = dst.numel();

  LengthType *src_shape_gpu;
  LengthType *dst_shape_gpu;
  cudaMalloc(&src_shape_gpu, ndim * sizeof(LengthType));
  cudaMalloc(&dst_shape_gpu, ndim * sizeof(LengthType));
  cudaMemcpy(src_shape_gpu, src_shape.data(), ndim * sizeof(LengthType),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dst_shape_gpu, dst_shape.data(), ndim * sizeof(LengthType),
             cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  cudaError_t error_a = cudaGetLastError();
  if (error_a != cudaSuccess) {
    printf("transpose allocation CUDA error: %s\n",
           cudaGetErrorString(error_a));
    RV_UNIMPLEMENTED();
  }

#define LAUNCH_TRANSPOSE_KERNEL(type)                                          \
  if (total_elems % 256 == 0) {                                                \
    _transpose<type, ndim><<<total_elems / 256, 256>>>(                        \
        x.data_ptr<type>(), dst.data_ptr<type>(), dim_a, dim_b, src_shape_gpu, \
        dst_shape_gpu, total_elems);                                           \
  } else {                                                                     \
    _transpose<type, ndim>                                                     \
        <<<1, 256>>>(x.data_ptr<type>(), dst.data_ptr<type>(), dim_a, dim_b,   \
                     src_shape_gpu, dst_shape_gpu, total_elems);               \
  }

  if (x.dtype() == DType::kFloat32) {
    LAUNCH_TRANSPOSE_KERNEL(float)
  } else if (x.dtype() == DType::kFloat16) {
    LAUNCH_TRANSPOSE_KERNEL(half)
  } else if (x.dtype() == DType::kInt8) {
    LAUNCH_TRANSPOSE_KERNEL(int8_t)
  } else {
    RV_UNIMPLEMENTED();
  }

  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("transpose CUDA error: %s\n", cudaGetErrorString(error));
  }

  cudaFree(src_shape_gpu);
  cudaFree(dst_shape_gpu);

  return dst;
}

Tensor transpose(const Tensor &x, int dim_a, int dim_b) {
  int ndim = x.sizes().size();
  RV_CHECK(ndim >= 2)
  if (ndim == 2) {
    return transpose_internal<2>(x, dim_a, dim_b);
  } else if (ndim == 3) {
    return transpose_internal<3>(x, dim_a, dim_b);
  } else if (ndim == 4) {
    return transpose_internal<4>(x, dim_a, dim_b);
  } else {
    RV_UNIMPLEMENTED();
  }
}

KernelRegister transpose_reg_cuda("transpose", Device::kCUDA, transpose);

} // namespace cuda
} // namespace rwkv
