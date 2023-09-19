#include "check.h"
#include "utils.h"
#include <bits/stdint-intn.h>
#include <bits/stdint-uintn.h>
#include <functional>
#include <kernels/kernels.h>
#include <kernels/registry.h>
#include <numeric>
#include <tensor.h>
#include <vector>

namespace rwkv {
namespace def {

template <typename T>
void transpose_internal(const T *src, T *dst, const Shape &src_shape,
                        const Shape &dst_shape, int dim_a, int dim_b,
                        LengthType total_elems) {
  auto ndim = src_shape.size();
  std::vector<LengthType> src_idx(ndim);
  std::vector<LengthType> dst_idx(ndim);
  for (LengthType i = 0; i < total_elems; i++) {
    utils::offset_to_indices(i, dst_shape, dst_idx);
    for (LengthType j = 0; j < ndim; j++) {
      src_idx[j] = dst_idx[j];
    }
    src_idx[dim_a] = dst_idx[dim_b];
    src_idx[dim_b] = dst_idx[dim_a];
    LengthType src_pos = utils::indices_to_offset(src_shape, src_idx);
    dst[i] = src[src_pos];
  }
}

Tensor transpose(const Tensor &x, int dim_a, int dim_b) {
  auto deduce_shape = [&x, &dim_a, &dim_b]() {
    auto n_dim = x.sizes().size();
    RV_CHECK(n_dim >= 2)
    if (dim_a < 0)
      dim_a += n_dim;
    if (dim_b < 0)
      dim_b += n_dim;
    RV_CHECK(dim_a < n_dim && dim_b < n_dim)
    auto shape = x.shape();
    std::swap(shape[dim_a], shape[dim_b]);
    return shape;
  };
  Shape src_shape = x.shape();
  Shape dst_shape = deduce_shape();
  Tensor dst = Tensor::Empty(dst_shape, x.dtype(), x.device());
  auto total_elems = x.numel();

#define LAUNCH_TRANSPOSE_KERNEL(type)                                          \
  transpose_internal(x.data_ptr<type>(), dst.data_ptr<type>(), src_shape,      \
                     dst_shape, dim_a, dim_b, total_elems);

  if (x.dtype() == DType::kFloat32) {
    LAUNCH_TRANSPOSE_KERNEL(float)
  } else if (x.dtype() == DType::kFloat16) {
    LAUNCH_TRANSPOSE_KERNEL(half)
  } else if (x.dtype() == DType::kInt8) {
    LAUNCH_TRANSPOSE_KERNEL(int8_t)
  } else {
    RV_CHECK(false)
  }

  return dst;
}

KernelRegister transpose_reg_cpu("transpose", Device::kCPU, transpose);
KernelRegister transpose_reg_cuda("transpose", Device::kCUDA, transpose);

} // namespace def
} // namespace rwkv