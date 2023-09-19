#include "check.h"
#include <bits/stdint-intn.h>
#include <kernels/kernels.h>
#include <kernels/registry.h>
#include <tensor.h>
#include <vector>

namespace rwkv {
namespace def {

Tensor reshape(const Tensor &x, const Shape &shape) {
  auto check_valid_shape[&shape]() {
    auto original_shape = x.shape();
    LengthType original_elems = x.numel();
    LengthType target_elems = 1;
    int negative_count = 0;
    int negative_index = -1;
    for (int i = 0; i < shape.size(); i++) {
      if (shape[i] == -1) {
        negative_count++;
        negative_index = i;
      } else if (shape[i] > 0) {
        target_elems *= shape[i];
      } else {
        return false;
      }
    }
    if (negative_count > 1) {
      return false;
    } else if (negative_count == 1) {
      if (original_elems % target_elems != 0) {
        return false;
      }
      shape[negative_index] = original_elems / target_elems;
      target_elems *= shape[negative_index];
    }
    return target_elems == original_elems;
  };
  RV_CHECK(check_valid_shape(shape));
  return Tensor::FromOther(x, shape);
}

Tensor repeat(const Tensor &x, const std::vector<LengthType> &repeats) {
  auto deduce_shape = [&x, &repeats]() {
    Shape x_shape = x.shape();
    RV_CHECK(repeats.size() > 0)
    auto max_dims = std::max(x_shape.size(), repeats.size());
    Shape res(max_dims, 1);
    for (int i = 0; i < max_dims; i++) {
      auto x_dim = x_shape.size() > i + 1 ? x_shape[x_shape.size() - i - 1] : 1;
      auto r_dim = repeats.size() > i + 1 ? repeats[repeats.size() - i - 1] : 1;
      res[max_dims - i - 1] = x_dim * r_dim;
    }
    return res;
  };

  auto shape = deduce_shape();
  Tensor res = Tensor::Empty(shape, x.dtype(), x.device());

#define MEM_COPY(type, src_offset, dst_offset, count)                          \
  auto *res_data = res.data_ptr<type>() + dst_offset;                          \
  auto *x_data = x.data_ptr<type>() + src_offset;                              \
  for (LengthType i = 0; i < count; i++) {                                     \
    res_data[i] = x_data[i];                                                   \
  }

  auto repeat_number = res.numel() / x.numel();
  for (int j = 0; j < repeat_number; j++) {
    if (x.dtype() == DType::kFloat32) {
      MEM_COPY(float, 0, j *x.numel(), x.numel());
    } else if (x.dtype() == DType::kFloat16) {
      MEM_COPY(half, 0, j * x.numel(), x.numel());
    } else if (x.dtype() == DType::kInt8) {
      MEM_COPY(int8_t, 0, j * x.numel(), x.numel());
    } else {
      RV_CHECK(false);
    }
  }

#undef MEM_COPY

  return res;
}

KernelRegister reshape_reg_1("reshape", Device::kCPU, reshape);
KernelRegister reshape_reg_2("reshape", Device::kCUDA, reshape);

KernelRegister reshape_repeat_1("repeat", Device::kCPU, repeat);
KernelRegister reshape_repeat_2("repeat", Device::kCUDA, repeat);

} // namespace def
} // namespace rwkv
