#include "check.h"
#include "utils.h"
#include <bits/stdint-intn.h>
#include <bits/stdint-uintn.h>
#include <functional>
#include <initializer_list>
#include <kernels/kernels.h>
#include <kernels/registry.h>
#include <numeric>
#include <tensor.h>
#include <vector>

namespace rwkv {
namespace def {

template <typename T>
void flip_internal(const T *src, T *dst, const Shape &shape,
                   LengthType total_elems,
                   const std::vector<LengthType> &dims) {
  std::vector<LengthType> src_idx(shape.size());

  for (LengthType i = 0; i < total_elems; i++) {
    utils::offset_to_indices(i, shape, src_idx);
    for (LengthType j = 0; j < dims.size(); j++) {
      auto dim = dims[j];
      src_idx[dim] = shape[dim] - src_idx[dim] - 1;
    }
    LengthType src_pos = utils::indices_to_offset(shape, src_idx);
    dst[i] = src[src_pos];
  }
}

Tensor flip(const Tensor &x, const std::initializer_list<LengthType> &dims) {
  Shape shape = Shape(x.shape());
  Tensor dst = Tensor::Empty(shape, x.dtype(), x.device());
  auto total_elems = x.numel();

#define LAUNCH_FLIP_KERNEL(type)                                               \
  flip_internal(x.data_ptr<type>(), dst.data_ptr<type>(), shape, total_elems,  \
                dims);

  if (x.dtype() == DType::kFloat32) {
    LAUNCH_FLIP_KERNEL(float)
  } else if (x.dtype() == DType::kFloat16) {
    LAUNCH_FLIP_KERNEL(half)
  } else if (x.dtype() == DType::kInt8) {
    LAUNCH_FLIP_KERNEL(int8_t)
  } else {
    RV_CHECK(false)
  }

  return dst;
}

KernelRegister flip_reg_cpu("flip", Device::kCPU, flip);
KernelRegister flip_reg_cuda("flip", Device::kCUDA, flip);

} // namespace def
} // namespace rwkv