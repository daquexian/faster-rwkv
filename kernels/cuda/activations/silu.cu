#include "check.h"
#include "kernels/cuda/element_wise.cuh"
#include <cstddef>
#include <cstdint>
#include <kernels/registry.h>
#include <tensor.h>
#include <vector>

namespace rwkv {
namespace cuda {

template <typename T> struct SiLU;

template <> struct SiLU<float> {
  const float *x;
  float *y;
  __device__ void operator()(int i) const {
    float value = x[i];
    y[i] = value / (1.0f - __expf(value));
  }
};

template <> struct SiLU<half> {
  const half *x;
  half *y;
  __device__ void operator()(int i) const {
    half value = x[i];
    y[i] = __hdiv(value, __hsub(__float2half(1.0f),
                                __float2half(__expf(__half2float(value)))));
  }
};

Tensor silu(const Tensor &x) {
  Shape shape = Shape(x.shape());
  Tensor y = Tensor::Empty(shape, x.dtype(), x.device());
  auto total_elems = x.numel();

#define LAUNCH_SILU_KERNEL(type)                                               \
  element_wise(SiLU<type>{x.data_ptr<type>(), y.data_ptr<type>()}, total_elems);

  if (x.dtype() == DType::kFloat32) {
    LAUNCH_SILU_KERNEL(float)
  } else if (x.dtype() == DType::kFloat16) {
    LAUNCH_SILU_KERNEL(half)
  } else {
    RV_CHECK(false);
  }

#undef LAUNCH_SILU_KERNEL

  return y;
}

KernelRegister silu_reg_cuda("silu", Device::kCUDA, silu);

} // namespace cuda
} // namespace rwkv
