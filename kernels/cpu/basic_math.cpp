
#include "check.h"
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

template <typename T>
void naive_sub(const T *x, const T *y, T *z, LengthType numel) {
  for (LengthType i = 0; i < numel; i++) {
    z[i] = x[i] - y[i];
  }
}

template <typename T>
void naive_div(const T *x, const T *y, T *z, LengthType numel,
               LengthType base_x, LengthType base_y) {
  for (LengthType i = 0; i < numel; i++) {
    z[i] = x[i % base_x] / y[i % base_y];
  }
}

Tensor sub(const Tensor &x, const Tensor &y) {
  RV_CHECK(x.sizes().size() == y.sizes().size() &&
           std::equal(x.sizes().begin(), x.sizes().end(), y.sizes().begin()));
  auto z = Tensor::Empty(x.sizes(), x.dtype(), x.device());
  LengthType total = z.numel();
  if (x.dtype() == DType::kFloat32) {
    naive_sub(x.data_ptr<float>(), y.data_ptr<float>(), z.data_ptr<float>(),
              total);
  } else if (x.dtype() == DType::kFloat16) {
    naive_sub(x.data_ptr<float16>(), y.data_ptr<float16>(),
              z.data_ptr<float16>(), total);
  } else {
    RV_UNIMPLEMENTED();
  }
  return z;
}

Tensor div(const Tensor &x, const Tensor &y) {
  RV_CHECK(x.sizes().size() == y.sizes().size() &&
           (std::equal(x.sizes().begin(), x.sizes().end(), y.sizes().begin()) ||
            x.numel() == 1 || y.numel() == 1));
  auto z = Tensor::Empty(x.numel() == 1 ? y.sizes() : x.sizes(), x.dtype(),
                         x.device());
  LengthType total = z.numel();
  if (x.dtype() == DType::kFloat32) {
    naive_div(x.data_ptr<float>(), y.data_ptr<float>(), z.data_ptr<float>(),
              total, x.numel(), y.numel());
  } else if (x.dtype() == DType::kFloat16) {
    naive_div(x.data_ptr<float16>(), y.data_ptr<float16>(),
              z.data_ptr<float16>(), total, x.numel(), y.numel());
  } else {
    RV_UNIMPLEMENTED();
  }
  return z;
}

KernelRegister sub_reg_cpu("sub", Device::kCPU, sub);
KernelRegister div_reg_cpu("div", Device::kCPU, div);

} // namespace cuda
} // namespace rwkv
