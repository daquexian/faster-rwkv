#include <iostream>
#include <kernels/registry.h>
#include <tensor.h>

namespace rwkv {
namespace cpu {

Tensor cast_dtype(const Tensor& x, DType dtype) {
  if (x.dtype() == dtype) {
    return x;
  }
  RV_CHECK(dtype == DType::kFloat32 || dtype == DType::kFloat16);
  RV_CHECK(x.dtype() == DType::kFloat32 || x.dtype() == DType::kFloat16);
  auto y = Tensor::Empty(x.shape(), dtype, x.device());
  if (dtype == DType::kFloat16) {
    for (int i = 0; i < x.numel(); ++i) {
      y.data_ptr<float16>()[i] = static_cast<float16>(x.data_ptr<float>()[i]);
    }
  } else if (dtype == DType::kFloat32) {
    for (int i = 0; i < x.numel(); ++i) {
      y.data_ptr<float>()[i] = static_cast<float>(x.data_ptr<float16>()[i]);
    }
  } else {
    RV_UNIMPLEMENTED();
  }
  return y;
}

KernelRegister cast_dtype_reg("cast_dtype", Device::kCPU, cast_dtype);

} // namespace cpu
} // namespace rwkv

