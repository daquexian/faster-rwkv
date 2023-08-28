#include "element_wise.cuh"
#include <kernels/registry.h>
#include <tensor.h>

namespace rwkv {
namespace cuda {
namespace {
struct Fp32ToHalf {
  __device__ __forceinline__ void operator()(int i) const {
    out[i] = __float2half(in[i]);
  }
  half *out;
  const float *in;
};
struct HalfToFp32 {
  __device__ __forceinline__ void operator()(int i) const {
    out[i] = __half2float(in[i]);
  }
  float *out;
  const half *in;
};
} // namespace

Tensor cast_dtype(const Tensor& x, DType dtype) {
  if (x.dtype() == dtype) {
    return x;
  }
  RV_CHECK(dtype == DType::kFloat32 || dtype == DType::kFloat16);
  RV_CHECK(x.dtype() == DType::kFloat32 || x.dtype() == DType::kFloat16);
  auto y = Tensor::Empty(x.shape(), dtype, x.device());
  if (dtype == DType::kFloat16) {
    element_wise(Fp32ToHalf{y.data_ptr<half>(), x.data_ptr<float>()},
                 x.numel());
  } else if (dtype == DType::kFloat32) {
    element_wise(HalfToFp32{y.data_ptr<float>(), x.data_ptr<half>()},
                 x.numel());
  } else {
    RV_UNIMPLEMENTED();
  }
  return y;
}

KernelRegister cast_dtype_reg("cast_dtype", Device::kCUDA, cast_dtype);

} // namespace cuda
} // namespace rwkv
