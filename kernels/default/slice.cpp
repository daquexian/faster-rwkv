#include "check.h"
#include <bits/stdint-intn.h>
#include <bits/stdint-uintn.h>
#include <functional>
#include <kernels/kernels.h>
#include <kernels/registry.h>
#include <numeric>
#include <tensor.h>
#include <vector>

namespace rwkv {
namespace def {

LengthType indices_to_offset(const Shape &shape,
                             const std::vector<LengthType> &indices) {
  LengthType offset = 0;
  LengthType base = 1;
  for (LengthType i = 0; i < shape.size(); i++) {
    auto reversed_index = shape.size() - i - 1;
    offset += indices[reversed_index] * base;
    base *= shape[reversed_index];
  }
  return offset;
}

void offset_to_indices(LengthType offset, const Shape &shape,
                       std::vector<LengthType> &indices) {
  for (LengthType i = 0; i < shape.size(); i++) {
    auto reversed_index = shape.size() - i - 1;
    indices[i] = offset % shape[reversed_index];
    offset /= shape[reversed_index];
  }
}

Shape slice_deduce_shape(const Shape &input_shape,
                         const std::vector<Range> &ranges) {
  Shape output_shape;
  for (int i = 0; i < input_shape.size(); i++) {
    LengthType input_dim = input_shape[i];
    auto [start, interval, end] = ranges[i];
    if (start == 0 && interval == 0 && end == 0) {
      end = input_dim;
      interval = 1;
    }
    if (start < 0)
      start += input_dim;
    if (end < 0)
      end += input_dim;
    RV_CHECK(interval > 0);
    RV_CHECK(start <= end && end <= input_dim);
    output_shape.push_back((end - start) / interval);
  }
  return output_shape;
}

// TODO(Rinne): write a more effective kernel.
template <typename T>
void slice_copy_value(const T *input, T *output, const Shape &input_shape,
                      const Shape &output_shape,
                      const std::vector<Range> &ranges) {
  auto output_total = std::accumulate(output_shape.begin(), output_shape.end(),
                                      1, std::multiplies<LengthType>());
  std::vector<LengthType> indices(input_shape.size());
  for (LengthType i = 0; i < output_total; i++) {
    offset_to_indices(i, output_shape, indices);
    for (int j = 0; j < indices.size(); j++) {
      auto [start, interval, end] = ranges[j];
      indices[j] = start + interval * indices[j];
    }
    auto input_offset = indices_to_offset(input_shape, indices);
    output[i] = input[input_offset];
  }
}

Tensor slice(const Tensor &x, const std::vector<Range> &ranges) {
  // currently slice with dim reduction has not been supported.
  RV_CHECK(x.sizes().size() == ranges.size());

  auto output_shape = slice_deduce_shape(x.shape(), ranges);
  Tensor output = Tensor::Empty(output_shape, x.dtype(), x.device());

#define LAUNCH_SLICE_VALUE_COPY(type)                                          \
  slice_copy_value(x.data_ptr<type>(), output.data_ptr<type>(), x.shape(),     \
                   output_shape, ranges);                                      \
  return output;

  if (x.dtype() == DType::kFloat32) {
    LAUNCH_SLICE_VALUE_COPY(float)
  } else if (x.dtype() == DType::kFloat16) {
    LAUNCH_SLICE_VALUE_COPY(half)
  } else if (x.dtype() == DType::kInt8) {
    LAUNCH_SLICE_VALUE_COPY(uint8_t)
  }

  RV_UNIMPLEMENTED()

#undef LAUNCH_SLICE_VALUE_COPY
}

KernelRegister slice_reg_cpu("slice", Device::kCPU, slice);
KernelRegister slice_reg_cuda("slice", Device::kCUDA, slice);

} // namespace def
} // namespace rwkv