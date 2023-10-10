#include "check.h"
#include <bits/stdint-intn.h>
#include <kernels/kernels.h>
#include <kernels/registry.h>
#include <tensor.h>
#include <vector>

namespace rwkv {
namespace def {

Tensor reshape(const Tensor &x, const Shape &shape) {
  auto check_and_normalize_shape = [&x](Shape &output_shape) {
    auto original_shape = x.shape();
    LengthType original_elems = x.numel();
    LengthType target_elems = 1;
    int negative_count = 0;
    int negative_index = -1;
    for (int i = 0; i < output_shape.size(); i++) {
      if (output_shape[i] == -1) {
        negative_count++;
        negative_index = i;
      } else if (output_shape[i] > 0) {
        target_elems *= output_shape[i];
      } else {
        return false;
      }
    }
    if (negative_count > 1) {
      return false;
    } else if (negative_count == 1) {
      if (original_elems % target_elems != 0) {
        return false;
      }
      output_shape[negative_index] = original_elems / target_elems;
      target_elems *= output_shape[negative_index];
    }
    return target_elems == original_elems;
  };
  Shape output_shape(shape);
  RV_CHECK(check_and_normalize_shape(output_shape));
  return Tensor::FromOther(x, output_shape);
}

KernelRegister reshape_reg_1("reshape", Device::kCPU, reshape);
KernelRegister reshape_reg_2("reshape", Device::kCUDA, reshape);

} // namespace def
} // namespace rwkv
