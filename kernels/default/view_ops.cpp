#include <kernels/kernels.h>
#include <kernels/registry.h>
#include <tensor.h>

namespace rwkv {
namespace def {

Tensor reshape(const Tensor &x, const Shape &shape) {
  return Tensor::FromOther(x, shape);
}

KernelRegister reshape_reg_1("reshape", Device::kCPU, reshape);
KernelRegister reshape_reg_2("reshape", Device::kCUDA, reshape);

} // namespace def
} // namespace rwkv
