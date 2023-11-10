
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

template <typename T> void naive_relu(const T *x, T *y, LengthType numel) {
  for (LengthType i = 0; i < numel; i++) {
    y[i] = x[i] > T(.0f) ? x[i] : T(.0f);
  }
}

Tensor relu(const Tensor &x) {
  LengthType total = x.numel();
  Tensor y = Tensor::Empty(x.sizes(), x.dtype(), x.device());
  if (x.dtype() == DType::kFloat32) {
    naive_relu(x.data_ptr<float>(), y.data_ptr<float>(), total);
  } else if (x.dtype() == DType::kFloat16) {
    naive_relu(x.data_ptr<float16>(), y.data_ptr<float16>(), total);
  } else {
    RV_UNIMPLEMENTED();
  }
  return y;
}

KernelRegister relu_reg_cpu("relu", Device::kCPU, relu);

} // namespace cuda
} // namespace rwkv
