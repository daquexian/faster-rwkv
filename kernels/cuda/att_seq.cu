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
struct AttSeqMix {
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

struct AttSeqMixWithG {
  const half *xx;
  const half *sx;
  const half *k_mix;
  const half *v_mix;
  const half *r_mix;
  const half *g_mix;
  /* out */ half *kx;
  /* out */ half *vx;
  /* out */ half *rx;
  /* out */ half *gx;
  LengthType mix_numel;

  __device__ void operator()(int i) const {
    half xx_ = xx[i];
    half sx_ = sx[i];
    half k_mix_ = k_mix[i % mix_numel];
    half v_mix_ = v_mix[i % mix_numel];
    half r_mix_ = r_mix[i % mix_numel];
    half g_mix_ = g_mix[i % mix_numel];
    kx[i] = __hadd(__hmul(xx_, k_mix_),
                   __hmul(sx_, __hsub(__float2half(1), k_mix_)));
    vx[i] = __hadd(__hmul(xx_, v_mix_),
                   __hmul(sx_, __hsub(__float2half(1), v_mix_)));
    rx[i] = __hadd(__hmul(xx_, r_mix_),
                   __hmul(sx_, __hsub(__float2half(1), r_mix_)));
    gx[i] = __hadd(__hmul(xx_, g_mix_),
                   __hmul(sx_, __hsub(__float2half(1), g_mix_)));
  }
};

struct InplaceSiLU {
  half *x;
  __device__ void operator()(int i) const {
    half value = x[i];
    x[i] = __hdiv(value, __hadd(__float2half(1.0f),
                                __float2half(__expf(-__half2float(value)))));
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

struct NoBroadcastMul {
  const float *a;
  const float *b;
  /* out */ float *c;

  __device__ void operator()(int i) const { c[i] = a[i] * b[i]; }
};

struct NoBroadcastMulHalf {
  const half *a;
  const half *b;
  /* out */ half *c;

  __device__ void operator()(int i) const { c[i] = __hmul(a[i], b[i]); }
};

struct SingleSideBroadcastMul {
  const float *ws;
  const float *s;
  /* out */ float *wss;
  const int broad_cast_base;

  __device__ void operator()(int i) const {
    wss[i] = s[i] * ws[i / broad_cast_base];
  }
};

struct KwkMul {
  const float *k;
  const float *wk;
  /* out */ float *kwk;
  const int broad_cast_base;
  const int last_dim;

  __device__ void operator()(int i) const {
    kwk[i] =
        k[i] * wk[i / last_dim / broad_cast_base * last_dim + i % last_dim];
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
                   Tensor &kwkv_gemm, Tensor &wss_mul, LengthType H,
                   LengthType S, LengthType T) {
  Tensor xx = cuda::layer_norm_op(x, ln_w, ln_b);
  Tensor converted_sx =
      cat(unsqueeze(sx, -1), xx.slice({Range(0, 1, -1), Range::All}), 0);

  element_wise(AttSeqMix{xx.data_ptr<half>(), converted_sx.data_ptr<half>(),
                         k_mix.data_ptr<half>(), v_mix.data_ptr<half>(),
                         r_mix.data_ptr<half>(), kx.data_ptr<half>(),
                         vx.data_ptr<half>(), rx.data_ptr<half>(),
                         k_mix.numel()},
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

  element_wise(NoBroadcastMul{rk_gemm.data_ptr<float>(), w.data_ptr<float>(),
                              rkw_mul.data_ptr<float>()},
               rkw_mul.numel());
  element_wise(
      SingleSideBroadcastMul{wb.data_ptr<float>(), rs_gemm.data_ptr<float>(),
                             rswb_mul.data_ptr<float>(), static_cast<int>(S)},
      rswb_mul.numel());

  gemm_cublas_tensor(rkw_mul, v, rkwv_gemm); // [H, T, S]

  element_wise(OnePairAdd{rkwv_gemm.data_ptr<float>(),
                          rswb_mul.data_ptr<float>(),
                          rkwv_gemm.data_ptr<float>()},
               rkwv_gemm.numel());
  element_wise(SingleSideBroadcastMul{ws.data_ptr<float>(), s.data_ptr<float>(),
                                      wss_mul.data_ptr<float>(),
                                      static_cast<int>(s.size(1) * s.size(2))},
               wss_mul.numel());
  element_wise(KwkMul{k.data_ptr<float>(), wk.data_ptr<float>(),
                      kwk_mul.data_ptr<float>(), static_cast<int>(k.size(1)),
                      static_cast<int>(k.size(2))},
               kwk_mul.numel());
  gemm_cublas_tensor(kwk_mul, v, kwkv_gemm); // [H, S, S]
  element_wise(OnePairAdd{wss_mul.data_ptr<float>(),
                          kwkv_gemm.data_ptr<float>(), s_out.data_ptr<float>()},
               s_out.numel());

  rkwv_gemm = rkwv_gemm.transpose(0, 1).reshape({T, H * S});
  rkwv_gemm = group_norm_op(rkwv_gemm, H, lx_w, lx_b);
  rkwv_gemm = cast_dtype(rkwv_gemm, x.dtype());
  gemm_cublas_tensor(rkwv_gemm, ow, x_plus_out);
  element_wise(
      OnePairAddHalfInplace{x.data_ptr<half>(), x_plus_out.data_ptr<half>()},
      x.numel());

  xx = xx.slice({Range(-1, 1, xx.size(0)), Range::All}).squeeze(0);

  return xx;
}

template <typename F, int N>
__global__ void
rwkv5_2_kernel_forward(const int B, const int T, const int C, const int H,
                       float *__restrict__ _state, const F *__restrict__ _r,
                       const F *__restrict__ _k, const F *__restrict__ _v,
                       const float *__restrict__ _w, const F *__restrict__ _u,
                       F *__restrict__ _y) {
  const int b = blockIdx.x / H;
  const int h = blockIdx.x % H;
  const int i = threadIdx.x;
  _w += h * N;
  _u += h * N;
  _state += h * N * N + i * N; // wrong if B > 1 !!!

  __shared__ float r[N], k[N], u[N], w[N];

  float state[N];
#pragma unroll
  for (int j = 0; j < N; j++)
    state[j] = _state[j];

  __syncthreads();
  u[i] = float(_u[i]);
  w[i] = _w[i];
  __syncthreads();

  for (int t = b * T * C + h * N + i; t < (b + 1) * T * C + h * N + i; t += C) {
    __syncthreads();
    r[i] = float(_r[t]);
    k[i] = float(_k[t]);
    __syncthreads();

    const float v = float(_v[t]);
    float y = 0;

#pragma unroll
    for (int j = 0; j < N; j += 4) {
      const float4 &r_ = (float4 &)(r[j]);
      const float4 &k_ = (float4 &)(k[j]);
      const float4 &w_ = (float4 &)(w[j]);
      const float4 &u_ = (float4 &)(u[j]);
      float4 &s = (float4 &)(state[j]);
      float4 x;

      x.x = k_.x * v;
      x.y = k_.y * v;
      x.z = k_.z * v;
      x.w = k_.w * v;

      y += r_.x * (u_.x * x.x + s.x);
      y += r_.y * (u_.y * x.y + s.y);
      y += r_.z * (u_.z * x.z + s.z);
      y += r_.w * (u_.w * x.w + s.w);

      s.x = s.x * w_.x + x.x;
      s.y = s.y * w_.y + x.y;
      s.z = s.z * w_.z + x.z;
      s.w = s.w * w_.w + x.w;
    }
    _y[t] = F(y);
  }
#pragma unroll
  for (int j = 0; j < N; j++)
    _state[j] = state[j];
}

void cuda_forward_fp16(int B, int T, int C, int H, float *state, const half *r,
                       const half *k, const half *v, const float *w,
                       const half *u, half *y) {
  int N = C / H;
  if (N == 64) {
    rwkv5_2_kernel_forward<half, 64>
        <<<dim3(B * H), dim3(N)>>>(B, T, C, H, state, r, k, v, w, u, y);
  } else {
    RV_UNIMPLEMENTED();
  }
}
void cuda_forward_fp32(int B, int T, int C, int H, float *state, const float *r,
                       const float *k, const float *v, const float *w,
                       const float *u, float *y) {
  int N = C / H;
  if (N == 64) {
    rwkv5_2_kernel_forward<float, 64>
        <<<dim3(B * H), dim3(N)>>>(B, T, C, H, state, r, k, v, w, u, y);
  } else {
    RV_UNIMPLEMENTED();
  }
}

inline Tensor rwkv5_2_internal(int B, int T, int C, int H, Tensor &state,
                               const Tensor &r, const Tensor &k,
                               const Tensor &v, const Tensor &w,
                               const Tensor &u) {
  auto y = Tensor::Empty({B, T, C}, r.dtype(), w.device());
  if (r.dtype() == DType::kFloat32) {
    cuda_forward_fp32(B, T, C, H, state.data_ptr<float>(), r.data_ptr<float>(),
                      k.data_ptr<float>(), v.data_ptr<float>(),
                      w.data_ptr<float>(), u.data_ptr<float>(),
                      y.data_ptr<float>());
  } else if (r.dtype() == DType::kFloat16) {
    cuda_forward_fp16(B, T, C, H, state.data_ptr<float>(), r.data_ptr<half>(),
                      k.data_ptr<half>(), v.data_ptr<half>(),
                      w.data_ptr<float>(), u.data_ptr<half>(),
                      y.data_ptr<half>());
  } else {
    RV_UNIMPLEMENTED();
  }
  return y;
}

Tensor _ATT_SEQ_V5_2(const Tensor &x, const Tensor &s, const Tensor &ln_w,
                     const Tensor &ln_b, const Tensor &lx_w, const Tensor &lx_b,
                     const Tensor &sx, const Tensor &k_mix, const Tensor &v_mix,
                     const Tensor &r_mix, const Tensor &g_mix, const Tensor &kw,
                     Tensor &kx, const Tensor &vw, Tensor &vx, const Tensor &rw,
                     Tensor &rx, const Tensor &gw, Tensor &gx, const Tensor &ow,
                     const Tensor &t_first, Tensor &k, const Tensor &t_decay,
                     Tensor &v, Tensor &r, Tensor &g, Tensor &decayed_s,
                     Tensor &x_plus_out, LengthType H, LengthType N,
                     LengthType T, int n_att) {
  decayed_s = Copy(s, s.device());
  Tensor xx = cuda::layer_norm_op(x, ln_w, ln_b);
  Tensor converted_sx =
      cat(unsqueeze(sx, 0), xx.slice({Range(0, 1, -1), Range::All}), 0);

  element_wise(
      AttSeqMixWithG{xx.data_ptr<half>(), converted_sx.data_ptr<half>(),
                     k_mix.data_ptr<half>(), v_mix.data_ptr<half>(),
                     r_mix.data_ptr<half>(), g_mix.data_ptr<half>(),
                     kx.data_ptr<half>(), vx.data_ptr<half>(),
                     rx.data_ptr<half>(), gx.data_ptr<half>(), k_mix.numel()},
      x.numel());

  gemm_cublas_tensor(rx, rw, r);
  gemm_cublas_tensor(kx, kw, k);
  gemm_cublas_tensor(vx, vw, v);
  gemm_cublas_tensor(gx, gw, g);
  element_wise(InplaceSiLU{g.data_ptr<half>()}, g.numel());

  decayed_s = decayed_s.transpose(-1, -2);
  // print_n(r, "r", 0, 30);
  // print_n(k, "k", 0, 30);
  // print_n(v, "v", 0, 30);
  Tensor out =
      rwkv5_2_internal(1, T, n_att, H, decayed_s, r, k, v, t_decay, t_first);
  // print_n(out, "out", 0, 30);
  decayed_s = decayed_s.transpose(-1, -2);

  out = out.reshape({T, H * N});
  out = group_norm_op(out, H, lx_w, lx_b);
  out = cast_dtype(out, x.dtype());
  element_wise(NoBroadcastMulHalf{out.data_ptr<half>(), g.data_ptr<half>(),
                                  out.data_ptr<half>()},
               out.numel());
  gemm_cublas_tensor(out, ow, x_plus_out);
  element_wise(
      OnePairAddHalfInplace{x.data_ptr<half>(), x_plus_out.data_ptr<half>()},
      x.numel());

  // print_n(x_plus_out, "x_plus_out", 0, 30);
  // exit(0);

  xx = xx.slice({Range(-1, 1, xx.size(0)), Range::All}).squeeze(0);

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
  auto rswb_mul = Tensor::Empty({H, T, S}, DType::kFloat32, s.device());
  auto rkwv_gemm = Tensor::Empty({H, T, S}, DType::kFloat32, s.device());
  auto kwk_mul = Tensor::Empty({H, S, T}, DType::kFloat32, s.device());
  auto kwkv_gemm = Tensor::Empty({H, S, S}, DType::kFloat32, s.device());
  auto wss_mul = Tensor::Empty(s.shape(), DType::kFloat32, s.device());

  Tensor xx =
      _ATT_SEQ_V5(x, s, ln_w, ln_b, lx_w, lx_b, sx, k_mix, v_mix, r_mix, kw, kx,
                  vw, vx, rw, rx, ow, t_first, k, t_decay, v, r, decayed_s,
                  x_plus_out, s_out, rk_gemm, rkw_mul, rkwv_gemm, rs_gemm,
                  rswb_mul, kwk_mul, kwkv_gemm, wss_mul, H, S, T);
  return std::make_tuple(x_plus_out, xx, s_out);
}

std::tuple<Tensor, Tensor, Tensor>
att_seq_v5_2(const Tensor &x, const Tensor &sx, const Tensor &s,
             const Tensor &ln_w, const Tensor &ln_b, const Tensor &lx_w,
             const Tensor &lx_b, const Tensor &k_mix, const Tensor &v_mix,
             const Tensor &r_mix, const Tensor &g_mix, const Tensor &t_decay,
             const Tensor &t_first, const Tensor &kw, const Tensor &vw,
             const Tensor &rw, const Tensor &gw, const Tensor &ow, int n_att) {

  auto kx = Tensor::Empty(x.sizes(), x.dtype(), x.device());
  auto vx = Tensor::Empty(x.sizes(), x.dtype(), x.device());
  auto rx = Tensor::Empty(x.sizes(), x.dtype(), x.device());
  auto gx = Tensor::Empty(x.sizes(), x.dtype(), x.device());
  auto k = Tensor::Empty({kx.size(0), kw.size(1)}, DType::kFloat32, x.device());
  auto v = Tensor::Empty({vx.size(0), vw.size(1)}, DType::kFloat32, x.device());
  auto r = Tensor::Empty({rx.size(0), rw.size(1)}, DType::kFloat32, x.device());
  auto g = Tensor::Empty({gx.size(0), gw.size(1)}, gx.dtype(), x.device());
  auto x_plus_out = Tensor::Empty(x.sizes(), x.dtype(), x.device());

  auto H = t_decay.size(0);
  auto N = x.size(x.shape().size() - 1) / H;
  auto T = x.size(0);

  auto decayed_s = Tensor::Empty(s.sizes(), DType::kFloat32, s.device());

  Tensor xx =
      _ATT_SEQ_V5_2(x, s, ln_w, ln_b, lx_w, lx_b, sx, k_mix, v_mix, r_mix,
                    g_mix, kw, kx, vw, vx, rw, rx, gw, gx, ow, t_first, k,
                    t_decay, v, r, g, decayed_s, x_plus_out, H, N, T, n_att);
  return std::make_tuple(x_plus_out, xx, decayed_s);
}

KernelRegister att_seq_v5_reg("att_seq_v5", Device::kCUDA, att_seq_v5);
KernelRegister att_seq_v5_2_reg("att_seq_v5_2", Device::kCUDA, att_seq_v5_2);

} // namespace cuda
} // namespace rwkv