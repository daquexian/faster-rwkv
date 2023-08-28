#include "element_wise.cuh"
#include <kernels/registry.h>
#include <iostream>
#include <tensor.h>

namespace rwkv {
namespace cuda {
namespace {
template<typename T>
struct InplaceFill {
  T *ptr;
  T value;
  __device__ void operator()(int i) { ptr[i] = value; }
};
} // namespace

Tensor& fill_(Tensor& x, float value) {
  if (x.dtype() == DType::kFloat32) {
    element_wise(InplaceFill<float>{x.data_ptr<float>(), value}, x.numel());
  } else if (x.dtype() == DType::kFloat16) {
    element_wise(InplaceFill<half>{x.data_ptr<half>(), __float2half(value)}, x.numel());
  } else {
    RV_UNIMPLEMENTED();
  }
  return x;
}

KernelRegister inplace_fill_reg("fill_", Device::kCUDA, fill_);

} // namespace cuda
} // namespace rwkv
