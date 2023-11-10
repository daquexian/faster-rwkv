
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

template <typename T> T naive_sum(const T *x, LengthType numel) {
  T res = T(.0f);
  for (LengthType i = 0; i < numel; i++) {
    res += x[i];
  }
  return res;
}

Tensor reduce(const Tensor &x, const std::string &mode) {
  if (mode == "sum") {
    // compute
    auto res = Tensor::Empty({1}, x.dtype(), x.device());
    if (x.dtype() == DType::kFloat32) {
      res.set_item({0}, naive_sum(x.data_ptr<float>(), x.numel()));
    } else if (x.dtype() == DType::kFloat16) {
      res.set_item({0}, naive_sum(x.data_ptr<float16>(), x.numel()));
    } else {
      RV_UNIMPLEMENTED();
    }
    return res;
  } else {
    RV_UNIMPLEMENTED();
  }
}

KernelRegister reduce_reg_cpu("reduce", Device::kCPU, reduce);

} // namespace cuda
} // namespace rwkv
