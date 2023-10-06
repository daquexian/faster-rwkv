#include "check.h"
#include "utils.h"
#include <bits/stdint-intn.h>
#include <bits/stdint-uintn.h>
#include <functional>
#include <initializer_list>
#include <kernels/kernels.h>
#include <kernels/registry.h>
#include <numeric>
#include <tensor.h>
#include <vector>

namespace rwkv {
namespace def {

template <typename T>
void pad_internal(const T *src, T *dst, const Shape &src_shape,
                  const Shape &dst_shape, LengthType total_elems,
                  const std::vector<LengthType> &paddings, T value) {
  std::vector<LengthType> dst_idx(dst_shape.size(), 0);
  std::vector<LengthType> src_idx(src_shape.size());
  LengthType n = total_elems;
  dst_idx[0] = -1;

  for (LengthType i = 0; i < n; i++) {
    dst_idx[0]++;
    int flag = 0;
    for (LengthType j = 0; j < dst_shape.size(); j++) {
      if (dst_idx[j] == dst_shape[j]) {
        dst_idx[j] = 0;
        dst_idx[j + 1]++;
      }

      if (dst_idx[j] < paddings[j * 2]) {
        flag = 1;
      } else if (dst_idx[j] >= src_shape[j] + paddings[j * 2]) {
        flag = 2;
      } else {
        src_idx[j] = dst_idx[j] - paddings[j * 2];
      }
    }

    if (!flag) {
      dst[utils::indices_to_offset(dst_shape, dst_idx)] =
          src[utils::indices_to_offset(src_shape, src_idx)];
    } else {
      dst[utils::indices_to_offset(dst_shape, dst_idx)] = value;
    }
  }
}

Tensor pad(const Tensor &x, const std::vector<LengthType> &paddings,
           const std::string &mode) {
  RV_CHECK(mode == "constant");
  RV_CHECK(paddings.size() % 2 == 0);
  auto src_shape = x.shape();
  std::vector<LengthType> paddings_vec;
  auto deduce_shape = [&src_shape, &paddings, &paddings_vec]() {
    Shape res(src_shape);
    for (int i = paddings.size(); i < res.size() * 2; i++) {
      paddings_vec.push_back(0);
    }
    for (int i = paddings.size() - 1; i > 0; i -= 2) {
      paddings_vec.push_back(paddings[i - 1]);
      paddings_vec.push_back(paddings[i]);
    }
    RV_CHECK(paddings_vec.size() == 2 * res.size());
    for (int i = 0; i < res.size(); i++) {
      RV_CHECK(paddings_vec[i * 2] >= 0 && paddings_vec[i * 2 + 1] >= 0);
      res[i] += paddings_vec[i * 2] + paddings_vec[i * 2 + 1];
    }
    return res;
  };

  Shape dst_shape = deduce_shape();
  Tensor dst = Tensor::Empty(dst_shape, x.dtype(), x.device());
  auto total_elems = x.numel();

#define LAUNCH_PAD_KERNEL(type, value)                                         \
  pad_internal(x.data_ptr<type>(), dst.data_ptr<type>(), src_shape, dst_shape, \
               total_elems, paddings_vec, value);

  if (x.dtype() == DType::kFloat32) {
    LAUNCH_PAD_KERNEL(float, .0f)
  } else if (x.dtype() == DType::kFloat16) {
    LAUNCH_PAD_KERNEL(half, static_cast<half>(.0f))
  } else if (x.dtype() == DType::kInt8) {
    LAUNCH_PAD_KERNEL(int8_t, static_cast<int8_t>(0))
  } else {
    RV_CHECK(false);
  }

  return dst;
}

KernelRegister pad_reg_cpu("pad", Device::kCPU, pad);

} // namespace def
} // namespace rwkv