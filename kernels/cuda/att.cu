#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "element_wise.cuh"
#include <tensor.h>
#include <kernels/registry.h>

namespace rwkv {
namespace cuda {

Tensor layer_norm_op(const Tensor &x, const Tensor &weight, const Tensor &bias);
Tensor group_norm_op(const Tensor& x, int num_groups, const Tensor& weight, const Tensor& bias);

namespace {
// Equivalent Python code:
// ww = t_first + k
// p = torch.maximum(pp, ww)
// e1 = torch.exp(pp - p)
// e2 = torch.exp(ww - p)
// wkv = ((e1 * aa + e2 * v) / (e1 * bb + e2)).to(dtype=x.dtype)
// ww = t_decay + pp
// p = torch.maximum(ww, k)
// e1 = torch.exp(ww - p)
// e2 = torch.exp(k - p)
// t1 = e1 * aa + e2 * v
// t2 = e1 * bb + e2
// r = r * wkv
// return t1, t2, p, r
struct WkvForwardOne {
  const float *t_first;
  const float *k;
  const float *pp;
  const float *aa;
  const float *bb;
  const float *t_decay;
  const float *v;
  /* out */ float *t1;
  /* out */ float *t2;
  /* out */ float *p;
  /* in & out */ half *r;

  __device__ void operator()(int i) const {
    float ww = t_first[i] + k[i];
    float pp_ = pp[i];
    float p_ = (pp_ > ww) ? pp_ : ww;
    float e1 = expf(pp_ - p_);
    float e2 = expf(ww - p_);
    float aa_ = aa[i];
    float bb_ = bb[i];
    float v_ = v[i];
    r[i] = __hmul(r[i], __float2half(((e1 * aa_ + e2 * v_) / (e1 * bb_ + e2))));
    ww = t_decay[i] + pp_;
    float k_ = k[i];
    p_ = (ww > k_) ? ww : k_;
    e1 = expf(ww - p_);
    e2 = expf(k_ - p_);
    t1[i] = e1 * aa_ + e2 * v_;
    t2[i] = e1 * bb_ + e2;
    p[i] = p_;
  }
};

struct OneV5MulAdd {
  const int broadcasted_length;
  const float *t_first;
  const float *a;
  const float *s;
  const float *t_decay;
  /* out */ float *out_temp2;
  /* out */ float *decayed_s;

  __device__ void operator()(int i) const {
    float t_first_elem = t_first[i / broadcasted_length];
    float a_elem = a[i];
    float s_elem = s[i];
    float t_decay_elem = t_decay[i / broadcasted_length];
    out_temp2[i] = __fadd_rn(__fmul_rn(t_first_elem, a_elem), s_elem);
    decayed_s[i] = __fadd_rn(__fmul_rn(t_decay_elem, s_elem), a_elem);
  }
};

/*
   Equivalent Python code:
   kx = xx * k_mix + sx * (1 - k_mix)
   vx = xx * v_mix + sx * (1 - v_mix)
   rx = xx * r_mix + sx * (1 - r_mix)
*/

struct Mix {
  const half *xx;
  const half *sx;
  const half *k_mix;
  const half *v_mix;
  const half *r_mix;
  /* out */ half *kx;
  /* out */ half *vx;
  /* out */ half *rx;

  __device__ void operator()(int i) const {
    half xx_ = xx[i];
    half sx_ = sx[i];
    half k_mix_ = k_mix[i];
    half v_mix_ = v_mix[i];
    half r_mix_ = r_mix[i];
    kx[i] = __hadd(__hmul(xx_, k_mix_),
                   __hmul(sx_, __hsub(__float2half(1), k_mix_)));
    vx[i] = __hadd(__hmul(xx_, v_mix_),
                   __hmul(sx_, __hsub(__float2half(1), v_mix_)));
    rx[i] = __hadd(__hmul(xx_, r_mix_),
                   __hmul(sx_, __hsub(__float2half(1), r_mix_)));
  }
};

struct InplaceSigmoid {
  __device__ __forceinline__ void operator()(int i) const {
    ptr[i] = __float2half(1.0 / (1.0 + exp(-__half2float(ptr[i]))));
  }
  half *ptr;
};

struct InplaceAdd {
  __device__ __forceinline__ void operator()(int i) const {
    y[i] = __hadd(x[i], y[i]);
  }
  half *y;
  const half *x;
};

} // namespace

void gemm_cublas_tensor(const Tensor &a, const Tensor &b, Tensor &c);
Tensor cast_dtype(const Tensor& x, DType dtype);

Tensor _ATT(const Tensor& x, const Tensor& ln_w, const Tensor& ln_b, const Tensor& sx, const Tensor& k_mix,
            const Tensor& v_mix, const Tensor& r_mix, const Tensor& kw,
            /* imm */ Tensor& kx, const Tensor& vw, /* imm */ Tensor& vx, const Tensor& rw,
            /* imm */ Tensor& rx, const Tensor& ow, const Tensor& t_first,
            /* imm */ Tensor& k, const Tensor& pp, const Tensor& ww, const Tensor& aa, const Tensor& bb,
            const Tensor& t_decay, /* imm */ Tensor& v, /* in & out */ Tensor& r,
            /* out */ Tensor& x_plus_out, /* out */ Tensor& t1,
            /* out */ Tensor& t2, /* out */ Tensor& p) {
  Tensor xx = cuda::layer_norm_op(x, ln_w, ln_b);
  element_wise(Mix{xx.data_ptr<half>(), sx.data_ptr<half>(),
                   k_mix.data_ptr<half>(), v_mix.data_ptr<half>(),
                   r_mix.data_ptr<half>(), kx.data_ptr<half>(),
                   vx.data_ptr<half>(), rx.data_ptr<half>()},
               x.numel());

  gemm_cublas_tensor(kx, kw, k);
  gemm_cublas_tensor(vx, vw, v);
  gemm_cublas_tensor(rx, rw, r);
  element_wise(InplaceSigmoid{r.data_ptr<half>()}, r.numel());

  element_wise(WkvForwardOne{t_first.data_ptr<float>(), k.data_ptr<float>(),
                             pp.data_ptr<float>(), aa.data_ptr<float>(),
                             bb.data_ptr<float>(), t_decay.data_ptr<float>(),
                             v.data_ptr<float>(), t1.data_ptr<float>(),
                             t2.data_ptr<float>(), p.data_ptr<float>(),
                             r.data_ptr<half>()},
               x.numel());

  gemm_cublas_tensor(r, ow, x_plus_out);
  element_wise(InplaceAdd{x_plus_out.data_ptr<half>(), x.data_ptr<half>()},
               x.numel());
  return xx;
}

Tensor _ATT_ONE_V5(const Tensor &x, const Tensor &s, const Tensor &ln_w,
                   const Tensor &ln_b,
                   const Tensor& lx_w, const Tensor& lx_b, const Tensor& sx, const Tensor& k_mix,
            const Tensor& v_mix, const Tensor& r_mix, const Tensor& kw,
            Tensor& kx, const Tensor& vw, Tensor& vx, const Tensor& rw,
            Tensor& rx, const Tensor& ow, const Tensor& t_first,
            Tensor& k,
            const Tensor& t_decay, Tensor& v, Tensor& r, Tensor& decayed_s,
            Tensor& x_plus_out, Tensor& a, Tensor& out_temp1, Tensor& out_temp2,
            LengthType H, LengthType S) {
  Tensor xx = cuda::layer_norm_op(x, ln_w, ln_b);

  element_wise(Mix{xx.data_ptr<half>(), sx.data_ptr<half>(),
                   k_mix.data_ptr<half>(), v_mix.data_ptr<half>(),
                   r_mix.data_ptr<half>(), kx.data_ptr<half>(),
                   vx.data_ptr<half>(), rx.data_ptr<half>()},
               x.numel());

  gemm_cublas_tensor(kx, kw, k);
  gemm_cublas_tensor(vx, vw, v);
  gemm_cublas_tensor(rx, rw, r);

  r = r.view({H, 1, S});
  k = k.view({H, S, 1});
  v = v.view({H, 1, S});

  gemm_cublas_tensor(k, v, a);

  element_wise(OneV5MulAdd{static_cast<int>(s.size(1) * s.size(2)), t_first.data_ptr<float>(), a.data_ptr<float>(), s.data_ptr<float>(), t_decay.data_ptr<float>(), out_temp2.data_ptr<float>(), decayed_s.data_ptr<float>()}, s.numel());

  gemm_cublas_tensor(r, out_temp2, out_temp1);
  out_temp1 = out_temp1.flatten().unsqueeze(0);

  Tensor out_temp3 = cuda::group_norm_op(out_temp1, H, lx_w, lx_b).flatten();
  
  Tensor out_temp4 = cast_dtype(out_temp3, DType::kFloat16);
  gemm_cublas_tensor(out_temp4, ow, x_plus_out);
  element_wise(InplaceAdd{x_plus_out.data_ptr<half>(), x.data_ptr<half>()},
               x.numel());

  return xx;
}

inline std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
att(const Tensor& x, const Tensor& sx, const Tensor& aa, const Tensor& bb, const Tensor& pp, const Tensor& ln_w,
    const Tensor& ln_b, const Tensor& k_mix, const Tensor& v_mix, const Tensor& r_mix, const Tensor& t_decay,
    const Tensor& t_first, const Tensor& kw, const Tensor& vw, const Tensor& rw, const Tensor& ow) {

  // kx = torch.empty_like(x)
  // vx = torch.empty_like(x)
  // rx = torch.empty_like(x)
  //
  // k_t = torch.empty((kw.shape[0],), dtype=torch.float32, device=x.device)
  // v_t = torch.empty((vw.shape[0],), dtype=torch.float32, device=x.device)
  // r_t = torch.empty((rw.shape[0],), dtype=torch.float16, device=x.device)
  // x_plus_out_t = torch.empty_like(x)
  // t1_t = torch.empty_like(x, dtype=torch.float32)
  // t2_t = torch.empty_like(x, dtype=torch.float32)
  // p_t = torch.empty_like(x, dtype=torch.float32)
  //             xx = torch.ops.rwkv.att_one(x, ln_w, ln_b, sx, k_mix, v_mix,
  //             r_mix, kw, kx, vw, vx, rw, rx, ow, t_first, k_t, pp, ow, aa,
  //             bb, t_decay, v_t, r_t, x_plus_out_t, t1_t, t2_t, p_t)
  // return x_plus_out_t, xx, t1_t, t2_t, p_t

  auto kx = Tensor::Empty(x.sizes(), x.dtype(), x.device());
  auto vx = Tensor::Empty(x.sizes(), x.dtype(), x.device());
  auto rx = Tensor::Empty(x.sizes(), x.dtype(), x.device());
  auto k = Tensor::Empty({kw.size(0)}, DType::kFloat32, x.device());
  auto v = Tensor::Empty({vw.size(0)}, DType::kFloat32, x.device());
  auto r = Tensor::Empty({rw.size(0)}, DType::kFloat16, x.device());
  auto x_plus_out = Tensor::Empty(x.sizes(), x.dtype(), x.device());
  auto t1 = Tensor::Empty(x.sizes(), DType::kFloat32, x.device());
  auto t2 = Tensor::Empty(x.sizes(), DType::kFloat32, x.device());
  auto p = Tensor::Empty(x.sizes(), DType::kFloat32, x.device());

  Tensor xx =
      _ATT(x, ln_w, ln_b, sx, k_mix, v_mix, r_mix, kw, kx, vw, vx, rw, rx, ow,
           t_first, k, pp, vw, aa, bb, t_decay, v, r, x_plus_out, t1, t2, p);
  return std::make_tuple(x_plus_out, xx, t1, t2, p);
}

// The v5 version implementation of att
inline std::tuple<Tensor, Tensor, Tensor>
att_one_v5(const Tensor &x, const Tensor &sx, const Tensor &s,
           const Tensor &ln_w, const Tensor &ln_b, const Tensor &lx_w,
           const Tensor &lx_b, const Tensor &k_mix, const Tensor &v_mix,
           const Tensor &r_mix, const Tensor &t_decay, const Tensor &t_first,
           const Tensor &kw, const Tensor &vw, const Tensor &rw,
           const Tensor &ow) {

  auto kx = Tensor::Empty(x.sizes(), x.dtype(), x.device());
  auto vx = Tensor::Empty(x.sizes(), x.dtype(), x.device());
  auto rx = Tensor::Empty(x.sizes(), x.dtype(), x.device());
  auto k = Tensor::Empty({kw.size(0)}, DType::kFloat32, x.device());
  auto v = Tensor::Empty({vw.size(0)}, DType::kFloat32, x.device());
  auto r = Tensor::Empty({rw.size(0)}, DType::kFloat32, x.device());
  auto x_plus_out = Tensor::Empty(x.sizes(), x.dtype(), x.device());

  auto H = t_decay.size(0);
  auto S = x.size(x.shape().size() - 1) / H;

  auto decayed_s = Tensor::Empty(s.sizes(), DType::kFloat32, s.device());
  auto a = Tensor::Empty(s.sizes(), DType::kFloat32, s.device());
  auto out_temp1 = Tensor::Empty({s.size(0), 1, s.size(2)}, DType::kFloat32, s.device());
  auto out_temp2 = Tensor::Empty(s.sizes(), DType::kFloat32, s.device());

  Tensor xx = _ATT_ONE_V5(x, s, ln_w, ln_b, lx_w, lx_b, sx, k_mix, v_mix, r_mix,
                          kw, kx, vw, vx, rw, rx, ow, t_first, k, t_decay, v, r,
                          decayed_s, x_plus_out, a, out_temp1, out_temp2, H, S);
  return std::make_tuple(x_plus_out, xx, decayed_s);
}

KernelRegister att_reg("att", Device::kCUDA, att);
KernelRegister att_one_v5_reg("att_one_v5", Device::kCUDA, att_one_v5);

} // namespace cuda
} // namespace rwkv
