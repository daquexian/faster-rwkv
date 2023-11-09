
#include "check.h"
#include "element_wise.cuh"
#include "util.cuh"
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
    auto x_cpu = Copy(x, Device::kCPU);
    auto res = Tensor::Empty({1}, x.dtype(), x.device());
    if (x_cpu.dtype() == DType::kFloat32) {
      res.set_item({0}, naive_sum(x_cpu.data_ptr<float>(), x_cpu.numel()));
    } else if (x_cpu.dtype() == DType::kFloat16) {
      res.set_item({0}, naive_sum(x_cpu.data_ptr<float16>(), x_cpu.numel()));
    } else {
      RV_UNIMPLEMENTED();
    }
    return res;
  } else {
    RV_UNIMPLEMENTED();
  }
}

KernelRegister reduce_reg_gpu("reduce", Device::kCUDA, reduce);

} // namespace cuda
} // namespace rwkv
