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
} // namespace

Tensor& scalar_div_(Tensor& x, float divisor) {
  half divisor_half = __float2half(divisor);
  element_wise(InplaceScalarDiv{x.data_ptr<half>(), divisor_half}, x.numel());
  return x;
}

KernelRegister inplace_scalar_div_reg("scalar_div_", Device::kCUDA, scalar_div_);

} // namespace cuda
} // namespace rwkv

