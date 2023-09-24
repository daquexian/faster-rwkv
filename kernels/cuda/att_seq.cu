#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "element_wise.cuh"
#include "layer_norm.cuh"
#include <iostream>
#include <kernels/kernels.h>
#include <kernels/registry.h>
#include <math.h>
#include <tensor.h>

namespace rwkv {
namespace cuda {

Tensor layer_norm_op(const Tensor &x, const Tensor &weight, const Tensor &bias);
Tensor group_norm_op(const Tensor &x, int num_groups, const Tensor &weight,
                     const Tensor &bias);
void gemm_cublas_tensor(const Tensor &a, const Tensor &b, Tensor &c);

namespace {
struct Mix {
  const half *xx;
  const half *sx;
  const half *k_mix;
  const half *v_mix;
  const half *r_mix;
  /* out */ half *kx;
  /* out */ half *vx;
  /* out */ half *rx;
  LengthType mix_numel;

  __device__ void operator()(int i) const {
    half xx_ = xx[i];
    half sx_ = sx[i];
    half k_mix_ = k_mix[i % mix_numel];
    half v_mix_ = v_mix[i % mix_numel];
    half r_mix_ = r_mix[i % mix_numel];
    kx[i] = __hadd(__hmul(xx_, k_mix_),
                   __hmul(sx_, __hsub(__float2half(1), k_mix_)));
    vx[i] = __hadd(__hmul(xx_, v_mix_),
                   __hmul(sx_, __hsub(__float2half(1), v_mix_)));
    rx[i] = __hadd(__hmul(xx_, r_mix_),
                   __hmul(sx_, __hsub(__float2half(1), r_mix_)));
  }
};

struct OnePairAdd {
  const float *a;
  const float *b;
  /* out */ float *c;

  __device__ void operator()(int i) const { c[i] = a[i] + b[i]; }
};

struct OnePairAddHalfInplace {
  const half *a;
  half *b;

  __device__ void operator()(int i) const { b[i] = __hadd(a[i], b[i]); }
};

struct TwoPairMul {
  const float *a;
  const float *b;
  const float *c;
  const float *d;
  /* out */ float *e;
  /* out */ float *f;

  __device__ void operator()(int i) const {
    e[i] = a[i] * b[i];
    f[i] = c[i] * d[i];
  }
};

struct PowScalar {
  const float *w;
  const float T;
  /* out */ float *ws;

  __device__ void operator()(int i) const { ws[i] = __powf(w[i], T); }
};

struct PowTensorInplace {
  float *w;
  const float *ind;

  __device__ void operator()(int i) const { w[i] = __powf(w[i], ind[i]); }
};
} // namespace

Tensor _ATT_SEQ_V5(const Tensor &x, const Tensor &s, const Tensor &ln_w,
                   const Tensor &ln_b, const Tensor &lx_w, const Tensor &lx_b,
                   const Tensor &sx, const Tensor &k_mix, const Tensor &v_mix,
                   const Tensor &r_mix, const Tensor &kw, Tensor &kx,
                   const Tensor &vw, Tensor &vx, const Tensor &rw, Tensor &rx,
                   const Tensor &ow, const Tensor &t_first, Tensor &k,
                   const Tensor &t_decay, Tensor &v, Tensor &r,
                   Tensor &decayed_s, Tensor &x_plus_out, Tensor &s_out,
                   Tensor &rk_gemm, Tensor &rkw_mul, Tensor &rkwv_gemm,
                   Tensor &rs_gemm, Tensor &rswb_mul, Tensor &kwk_mul,
                   Tensor &wss_mul, LengthType H, LengthType S, LengthType T) {
  Tensor xx = cuda::layer_norm_op(x, ln_w, ln_b);
  Tensor temp = xx.slice({Range(0, 1, -1), Range::All});
  Tensor converted_sx = cat(unsqueeze(sx, -1), temp, 0);

  // print_n(x, "x");
  // print_n(xx, "xx");
  // print_n(sx, "sx");

  // RV_UNIMPLEMENTED();

  element_wise(Mix{xx.data_ptr<half>(), converted_sx.data_ptr<half>(),
                   k_mix.data_ptr<half>(), v_mix.data_ptr<half>(),
                   r_mix.data_ptr<half>(), kx.data_ptr<half>(),
                   vx.data_ptr<half>(), rx.data_ptr<half>(), k_mix.numel()},
               x.numel());

  Tensor w = rwkv::reshape(t_decay, Shape({-1, 1})); // [H, 1]
  Tensor u = rwkv::reshape(t_first, Shape({-1, 1}));
  Tensor ws = Tensor::Empty(w.shape(), DType::kFloat32, w.device());
  element_wise(PowScalar{w.data_ptr<float>(), static_cast<float>(T),
                         ws.data_ptr<float>()},
               w.numel());
  ws = rwkv::reshape(ws, Shape({H, 1, 1}));
  Tensor ind = Tensor::Arange(static_cast<float>(T - 1), -1.0f, -1.0f,
                              w.dtype(), w.device());
  ind = ind.unsqueeze(0).repeat({H, 1}); // [H, T]
  w = w.repeat({1, T});
  element_wise(PowTensorInplace{w.data_ptr<float>(), ind.data_ptr<float>()},
               w.numel());

  Tensor wk = rwkv::reshape(w, Shape({H, 1, T}));
  Tensor wb = wk.transpose(-2, -1).flip(1);
  w = cat(w.slice({Range::All, Range(1, 1, w.size(1))}), u, 1);
  w = pad(w, {0, T}, "constant");
  w = w.repeat({1, T});
  w = w.slice({Range::All, Range(0, 1, -T)}).reshape(Shape({-1, T, 2 * T - 1}));
  w = w.slice({Range::All, Range::All, Range(T - 1, 1, w.size(2))})
          .reshape(Shape({H, T, T}));

  gemm_cublas_tensor(rx, rw, r);
  r = r.view({T, H, S}).transpose(0, 1); // [H, T, S]
  gemm_cublas_tensor(kx, kw, k);
  k = k.view({T, H, S}).transpose(0, 1);
  k = k.transpose(-2, -1); // [H, S, T]
  gemm_cublas_tensor(vx, vw, v);
  v = v.view({T, H, S}).transpose(0, 1); // [H, T, S]

  /* rkwv_gemm: `out` in python */
  // ((r @ k) * w) @ v + (r @ s) * wb
  gemm_cublas_tensor(r, k, rk_gemm); // [H, T, T]
  gemm_cublas_tensor(r, s, rs_gemm); // [H, T, S]
  element_wise(TwoPairMul{rk_gemm.data_ptr<float>(), w.data_ptr<float>(),
                          rs_gemm.data_ptr<float>(), wb.data_ptr<float>(),
                          rkw_mul.data_ptr<float>(),
                          rswb_mul.data_ptr<float>()},
               w.numel());
  element_wise(OnePairAdd{rkwv_gemm.data_ptr<float>(),
                          rswb_mul.data_ptr<float>(),
                          rkwv_gemm.data_ptr<float>()},
               rkwv_gemm.numel());
  gemm_cublas_tensor(rkw_mul, v, rkwv_gemm); // [H, T, S]
  print_n(rkw_mul, "rkw_mul");
  print_n(rkwv_gemm, "rkwv_gemm");

  // ws * s + (k * wk) @ v
  element_wise(TwoPairMul{k.data_ptr<float>(), wk.data_ptr<float>(),
                          ws.data_ptr<float>(), s.data_ptr<float>(),
                          kwk_mul.data_ptr<float>(), wss_mul.data_ptr<float>()},
               k.numel());
  kwk_mul = kwk_mul.unsqueeze(0).repeat({v.size(0), 1, 1});
  gemm_cublas_tensor(kwk_mul, v, rswb_mul); // [H, S, S]
  print_n(rkwv_gemm, "rkwv_gemm-0");
  element_wise(OnePairAdd{wss_mul.data_ptr<float>(), rswb_mul.data_ptr<float>(),
                          s_out.data_ptr<float>()},
               s_out.numel());
  print_n(s_out, "s_out");

  rkwv_gemm = rkwv_gemm.transpose(0, 1).reshape({T, H * S});
  print_n(rkwv_gemm, "rkwv_gemm-1");
  rkwv_gemm = group_norm_op(rkwv_gemm, H, lx_w, lx_b);
  print_n(lx_w, "lx_w");
  print_n(lx_b, "lx_b");
  print_n(rkwv_gemm, "rkwv_gemm-2");
  rkwv_gemm = cast_dtype(rkwv_gemm, x.dtype());
  gemm_cublas_tensor(rkwv_gemm, ow, x_plus_out);
  print_n(x_plus_out, "x_plus_out_gemm");
  element_wise(
      OnePairAddHalfInplace{x.data_ptr<half>(), x_plus_out.data_ptr<half>()},
      x.numel());

  print_n(x_plus_out, "x_plus_out");

  // RV_UNIMPLEMENTED();
  return xx;
}

inline std::tuple<Tensor, Tensor, Tensor>
att_seq_v5(const Tensor &x, const Tensor &sx, const Tensor &s,
           const Tensor &ln_w, const Tensor &ln_b, const Tensor &lx_w,
           const Tensor &lx_b, const Tensor &k_mix, const Tensor &v_mix,
           const Tensor &r_mix, const Tensor &t_decay, const Tensor &t_first,
           const Tensor &kw, const Tensor &vw, const Tensor &rw,
           const Tensor &ow) {
  auto kx = Tensor::Empty(x.sizes(), x.dtype(), x.device());
  auto vx = Tensor::Empty(x.sizes(), x.dtype(), x.device());
  auto rx = Tensor::Empty(x.sizes(), x.dtype(), x.device());
  auto k = Tensor::Empty({x.size(0), kw.size(0)}, DType::kFloat32, x.device());
  auto v = Tensor::Empty({x.size(0), vw.size(0)}, DType::kFloat32, x.device());
  auto r = Tensor::Empty({x.size(0), rw.size(0)}, DType::kFloat32, x.device());
  auto x_plus_out = Tensor::Empty(x.sizes(), x.dtype(), x.device());

  auto H = t_decay.size(0);
  auto S = x.size(x.shape().size() - 1) / H;
  auto T = x.size(0);

  auto decayed_s = Tensor::Empty(s.sizes(), DType::kFloat32, s.device());
  auto s_out = Tensor::Empty(s.sizes(), DType::kFloat32, s.device());

  auto rk_gemm = Tensor::Empty({H, T, T}, DType::kFloat32, s.device());
  auto rkw_mul = Tensor::Empty({H, T, T}, DType::kFloat32, s.device());
  auto rs_gemm = Tensor::Empty({H, T, S}, DType::kFloat32, s.device());
  auto rswb_mul = Tensor::Empty({H, S, S}, DType::kFloat32, s.device());
  auto rkwv_gemm = Tensor::Empty({H, T, S}, DType::kFloat32, s.device());
  auto kwk_mul = Tensor::Empty(k.shape(), DType::kFloat32, s.device());
  auto wss_mul = Tensor::Empty(s.shape(), DType::kFloat32, s.device());

  Tensor xx = _ATT_SEQ_V5(
      x, s, ln_w, ln_b, lx_w, lx_b, sx, k_mix, v_mix, r_mix, kw, kx, vw, vx, rw,
      rx, ow, t_first, k, t_decay, v, r, decayed_s, x_plus_out, s_out, rk_gemm,
      rkw_mul, rkwv_gemm, rs_gemm, rswb_mul, kwk_mul, wss_mul, H, S, T);
  std::cout << std::endl
            << "================================================" << std::endl
            << std::endl;
  return std::make_tuple(x_plus_out, xx, s_out);
}

KernelRegister att_seq_v5_reg("att_seq_v5", Device::kCUDA, att_seq_v5);

} // namespace cuda
} // namespace rwkv