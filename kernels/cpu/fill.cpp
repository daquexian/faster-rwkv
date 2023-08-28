#include <iostream>
#include <kernels/registry.h>
#include <tensor.h>

namespace rwkv {
namespace cpu {

Tensor &fill_(Tensor &x, float value) {
  auto numel = x.numel();
  if (x.dtype() == DType::kFloat32) {
    auto dptr = x.data_ptr<float>();
    for (int i = 0; i < numel; i++) {
      dptr[i] = value;
    }
  } else if (x.dtype() == DType::kFloat16) {
    auto dptr = x.data_ptr<float16>();
    auto fp16_value = static_cast<float16>(value);
    for (int i = 0; i < numel; i++) {
      dptr[i] = fp16_value;
    }
  } else {
    RV_UNIMPLEMENTED();
  }
  return x;
}

KernelRegister inplace_fill_reg("fill_", Device::kCPU, fill_);

} // namespace cpu
} // namespace rwkv
