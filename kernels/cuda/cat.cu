
#include "check.h"
#include "element_wise.cuh"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <kernels/registry.h>
#include <numeric>
#include <tensor.h>
#include <vector>

namespace rwkv {
namespace cuda {
/*
a: input a
b: input b
out: output
dim: the dim to cat
left: the element number at the left of the dim
right: the element number at the right of the dim
a_size: the size of the dim to cat of tensor a
b_size: the size of the dim to cat of tensor b
out_total: the total element number of the output tensor

TODO(Rinne): optimize this kernel.
*/
template <typename T>
__global__ void _cat(const T *a, const T *b, T *out, int dim, int left,
                     int right, int a_size, int b_size, int out_total) {
  int a_except_left = a_size * right;
  int b_except_left = b_size * right;
  int except_left =
      a_except_left + b_except_left; // elem count except the left part
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < out_total;
       i += blockDim.x * gridDim.x) {
    int left_index = i / except_left;
    int out_dim_index = (i / right) % except_left;
    int right_index = i % right;
    if (out_dim_index < a_size) {
      int a_dim_index = out_dim_index;
      int in_idx =
          left_index * a_except_left + a_dim_index * right + right_index;
      out[i] = a[in_idx];
    } else {
      int b_dim_index = out_dim_index - a_size;
      int in_idx =
          left_index * b_except_left + b_dim_index * right + right_index;
      out[i] = b[in_idx];
    }
  }
}

// NOTE: packed data type (e.g. float4) is a overkill for current sizes
// (4096 in 7B model and 768 in 0.1B model),
// and is not faster than the plain float version.
Tensor cat(const Tensor &a, const Tensor &b, int dim) {
  RV_CHECK(a.sizes().size() == b.sizes().size());
  RV_CHECK(a.dtype() == b.dtype());
  RV_CHECK(a.device() == b.device());
  int ndim = static_cast<int>(a.sizes().size());

  if (dim < 0) {
    dim += ndim;
  }
  RV_CHECK(dim >= 0 && dim < ndim);

  int left_size = 1;
  int right_size = 1;
  int a_dim_size = a.size(dim);
  int b_dim_size = b.size(dim);
  std::vector<LengthType> out_shape;
  for (int i = 0; i < ndim; i++) {
    if (i < dim) {
      RV_CHECK(a.size(i) == b.size(i));
      left_size *= static_cast<int>(a.size(i));
      out_shape.push_back(a.size(i));
    } else if (i > dim) {
      RV_CHECK(a.size(i) == b.size(i));
      right_size *= static_cast<int>(a.size(i));
      out_shape.push_back(a.size(i));
    } else {
      out_shape.push_back(a.size(i) + b.size(i));
    }
  }
  Tensor out = Tensor::Empty(out_shape, a.dtype(), a.device());
  int total = out.numel();

#define LAUNCH_KERNEL(type)                                                    \
  if (total % 256 == 0) {                                                      \
    _cat<<<total / 256, 256>>>(a.data_ptr<type>(), b.data_ptr<type>(),         \
                               out.data_ptr<type>(), dim, left_size,           \
                               right_size, a_dim_size, b_dim_size, total);     \
  } else {                                                                     \
    _cat<<<1, 256>>>(a.data_ptr<type>(), b.data_ptr<type>(),                   \
                     out.data_ptr<type>(), dim, left_size, right_size,         \
                     a_dim_size, b_dim_size, total);                           \
  }

  if (out.dtype() == DType::kFloat32) {
    LAUNCH_KERNEL(float)
  } else if (out.dtype() == DType::kFloat16) {
    LAUNCH_KERNEL(half)
  } else if (out.dtype() == DType::kInt8) {
    LAUNCH_KERNEL(int8_t);
  }

#undef LAUNCH_KERNEL

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("cat CUDA error: %s\n", cudaGetErrorString(error));
  }

  return out;
}

KernelRegister inplace_cat_reg("cat", Device::kCUDA, cat);

} // namespace cuda
} // namespace rwkv
