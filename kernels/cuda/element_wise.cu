#include "element_wise.cuh"
#include <kernels/registry.h>
#include <tensor.h>

namespace rwkv {
namespace cuda {
namespace {
struct InplaceScalarDiv {
  half *ptr;
  half divisor;
  __device__ void operator()(int i) { ptr[i] /= divisor; }
};

struct Multiply {
  const half* a;
  const half* b;
  half* c;
  __device__ void operator()(int i) { c[i] = a[i] * b[i]; }
};

struct Add {
  const half* a;
  const half* b;
  half* c;
  __device__ void operator()(int i) { c[i] = a[i] + b[i]; }
};
} // namespace

Tensor scalar_div_(Tensor& x, float divisor) {
  half divisor_half = __float2half(divisor);
  element_wise(InplaceScalarDiv{x.data_ptr<half>(), divisor_half}, x.numel());
  return x;
}

Tensor multiply(const Tensor& a, const Tensor& b) {
  Tensor c = Tensor::Empty(a.sizes(), a.dtype(), a.device());
  element_wise(Multiply{a.data_ptr<half>(), b.data_ptr<half>(), c.data_ptr<half>()}, c.numel());
  return c;
}

Tensor add(const Tensor& a, const Tensor& b) {
  Tensor c = Tensor::Empty(a.sizes(), a.dtype(), a.device());
  element_wise(Add{a.data_ptr<half>(), b.data_ptr<half>(), c.data_ptr<half>()}, c.numel());
  return c;
}

KernelRegister inplace_scalar_div_reg("scalar_div_", Device::kCUDA, scalar_div_);
KernelRegister multiply_reg("mul", Device::kCUDA, multiply);
KernelRegister add_reg("add", Device::kCUDA, add);

} // namespace cuda
} // namespace rwkv

