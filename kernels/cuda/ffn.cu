#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "element_wise.cuh"
#include "matmul.h"
#include <iostream>
#include <kernels/registry.h>
#include <tensor.h>

namespace rwkv {
namespace cuda {

Tensor layer_norm_op(const Tensor& x, const Tensor& weight, const Tensor& bias);

struct FfnOneMix {
  __device__ __forceinline__ void operator()(int idx) {
    half k_mix_ = k_mix[idx];
    half r_mix_ = r_mix[idx];
    half xx_ = xx[idx];
    half sx_ = sx[idx];
    kx[idx] = __hadd(__hmul(xx_, k_mix_),
                     __hmul(sx_, __hsub(__float2half(1), k_mix_)));
    rx[idx] = __hadd(__hmul(xx_, r_mix_),
                     __hmul(sx_, __hsub(__float2half(1), r_mix_)));
  }
  const half *k_mix;
  const half *r_mix;
  const half *xx;
  const half *sx;
  half *kx;
  half *rx;
};

struct InplaceSigmoid {
  __device__ __forceinline__ void operator()(int i) const {
    ptr[i] = __float2half(1.0 / (1.0 + exp(-__half2float(ptr[i]))));
  }
  half *ptr;
};

struct InplaceReLUAndSquare {
  __device__ __forceinline__ void operator()(int i) const {
    // __hmax is not defined in old cuda
    if (__hgt(ptr[i], __float2half(0))) {
      ptr[i] = __hmul(ptr[i], ptr[i]);
    } else {
      ptr[i] = __float2half(0);
    }
  }
  half *ptr;
};

struct InplaceFma {
  __device__ __forceinline__ void operator()(int i) const {
    a[i] = __hfma(a[i], b[i], c[i]);
  }
  half *a;
  const half *b;
  const half *c;
};

Tensor _FFN(const Tensor &x, const Tensor &sx, const Tensor &ln_w,
            const Tensor &ln_b, const Tensor &k_mix, const Tensor &r_mix,
            const Tensor &kw, const Tensor &vw, const Tensor &rw,
            /* imm */ Tensor &buf,
            /* out */ Tensor &x_plus_out) {
  Tensor xx = cuda::layer_norm_op(x, ln_w, ln_b);
  char *buf_ptr = (char *)buf.data_ptr();
  half *kx = (half *)buf_ptr;
  half *rx = kx + x.numel();
  half *vx = rx + x.numel();
  half *r = vx + x.size(0) * kw.size(1);
  element_wise(FfnOneMix{k_mix.data_ptr<half>(), r_mix.data_ptr<half>(),
                         xx.data_ptr<half>(), sx.data_ptr<half>(), kx, rx},
               x.numel());
  // vector * matrix, so m = 1
  // TODO: batch matmul
  gemm_cublas(rx, rw.data_ptr<half>(), r, 1, 1, rw.size(1), rw.size(0));
  element_wise(InplaceSigmoid{r}, rw.size(1));
  gemm_cublas(kx, kw.data_ptr<half>(), vx, 1, 1, kw.size(1), kw.size(0));
  element_wise(InplaceReLUAndSquare{vx}, kw.size(1));
  gemm_cublas(vx, vw.data_ptr<half>(), x_plus_out.data_ptr<half>(), 1, 1,
              vw.size(1), vw.size(0));
  Tensor r_t = Tensor::FromPtr(r, x.sizes(), DType::kFloat16, x.device());
  // hfma loses precision
  x_plus_out = x_plus_out * r_t + x;
  return xx;
}

std::tuple<Tensor, Tensor> ffn(const Tensor &x, const Tensor &sx,
                               const Tensor &ln_w, const Tensor &ln_b,
                               const Tensor &k_mix, const Tensor &r_mix,
                               const Tensor &kw, const Tensor &vw,
                               const Tensor &rw) {
  // krx_bytes = x.numel() * x.element_size()
  // vx_bytes = x.shape[0] * kw.shape[1] * x.element_size()
  // r_bytes = x.shape[0] * rw.shape[1] * x.element_size()
  // buf = torch.empty((krx_bytes * 2 + vx_bytes + r_bytes,), device=x.device,
  // dtype=torch.int8) x_plus_out = torch.empty_like(x) xx =
  // torch.ops.rwkv.ffn_one(x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw, rw, buf,
  // x_plus_out) return x_plus_out, xx
  int krx_bytes = x.numel() * x.elem_size();
  int vx_bytes = x.size(0) * kw.size(1) * x.elem_size();
  int r_bytes = x.size(0) * rw.size(1) * x.elem_size();
  Tensor buf = Tensor::Empty({krx_bytes * 2 + vx_bytes + r_bytes}, DType::kInt8,
                             x.device());
  Tensor x_plus_out = Tensor::Empty(x.sizes(), x.dtype(), x.device());
  Tensor xx =
      _FFN(x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw, rw, buf, x_plus_out);
  return std::make_tuple(x_plus_out, xx);
}

KernelRegister ffn_reg("ffn", Device::kCUDA, ffn);

} // namespace cuda
} // namespace rwkv
