#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "element_wise.cuh"
#include "matmul.h"
#include <iostream>
#include <kernels/registry.h>
#include <tensor.h>

namespace rwkv {
namespace cuda {

Tensor layer_norm_op(const Tensor &x, const Tensor &weight, const Tensor &bias);
void gemm_cublas_tensor(const Tensor &a, const Tensor &b, Tensor &c);

struct FfnSeqMix {
  __device__ __forceinline__ void operator()(int idx) const {
    half k_mix_ = k_mix[idx % mix_numel];
    half r_mix_ = r_mix[idx % mix_numel];
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
  int mix_numel;
};

struct InplaceSigmoid {
  __device__ __forceinline__ void operator()(int i) const {
    auto temp = __float2half(1.0 / (1.0 + exp(-__half2float(ptr[i]))));
    ptr[i] = temp;
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

Tensor _FFN_SEQ(const Tensor &x, const Tensor &sx, const Tensor &ln_w,
                const Tensor &ln_b, const Tensor &k_mix, const Tensor &r_mix,
                const Tensor &kw, const Tensor &vw, const Tensor &rw,
                /* imm */ Tensor &buf, Tensor &kx, Tensor &rx, Tensor &vx,
                Tensor &r, /* out */ Tensor &x_plus_out) {
  Tensor xx = cuda::layer_norm_op(x, ln_w, ln_b);
  // print_n(xx, "xx", xx.numel());
  Tensor sx_cat =
      sx.unsqueeze(0).cat(xx.slice({Range(0, 1, -1), Range::All}), 0);
  // char *buf_ptr = (char *)buf.data_ptr();
  // half *kx = (half *)buf_ptr;
  // half *rx = kx + x.numel();
  // half *vx = rx + x.numel();
  // half *r = vx + x.size(0) * kw.size(1);
  // RV_UNIMPLEMENTED();
  // element_wise(FfnOneMix{k_mix.data_ptr<half>(), r_mix.data_ptr<half>(),
  //                        xx.data_ptr<half>(), sx_cat.data_ptr<half>(),
  //                        kx.data_ptr<half>(), rx.data_ptr<half>(),
  //                        static_cast<int>(k_mix.numel())},
  //              xx.numel());
  //
  element_wise2<int>(xx.numel(), k_mix.data_ptr<half>(), r_mix.data_ptr<half>(),
                     xx.data_ptr<half>(), sx_cat.data_ptr<half>(),
                     kx.data_ptr<half>(), rx.data_ptr<half>(),
                     static_cast<int>(k_mix.numel()));
  // print_n(rx, "rx", rx.numel());
  gemm_cublas_tensor(rx, rw, r);
  element_wise(InplaceSigmoid{r.data_ptr<half>()}, r.numel());
  gemm_cublas_tensor(kx, kw, vx);
  element_wise(InplaceReLUAndSquare{vx.data_ptr<half>()}, vx.numel());
  gemm_cublas_tensor(vx, vw, x_plus_out);
  element_wise(InplaceFma{x_plus_out.data_ptr<half>(), r.data_ptr<half>(),
                          x.data_ptr<half>()},
               x_plus_out.numel());
  // print_n(x_plus_out, "x_plus_out", x_plus_out.numel());
  // RV_UNIMPLEMENTED();
  return xx.slice({Range(-1, 1, xx.size(0)), Range::All}).squeeze(0);
}

std::tuple<Tensor, Tensor> ffn_seq(const Tensor &x, const Tensor &sx,
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
  Tensor kx = Tensor::Empty(x.sizes(), DType::kFloat16, x.device());
  Tensor rx = Tensor::Empty(x.sizes(), DType::kFloat16, x.device());
  Tensor vx =
      Tensor::Empty({x.size(0), kw.size(1)}, DType::kFloat16, x.device());
  Tensor r =
      Tensor::Empty({x.size(0), vw.size(1)}, DType::kFloat16, x.device());
  Tensor x_plus_out = Tensor::Empty(x.sizes(), x.dtype(), x.device());
  Tensor xx = _FFN_SEQ(x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw, rw, buf, kx, rx,
                       vx, r, x_plus_out);
  return std::make_tuple(x_plus_out, xx);
}

KernelRegister ffn_seq_reg("ffn_seq", Device::kCUDA, ffn_seq);

} // namespace cuda
} // namespace rwkv
