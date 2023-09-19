#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "element_wise.cuh"
#include "layer_norm.cuh"
#include <math.h>
#include <tensor.h>
#include <kernels/registry.h>
#include <kernels/kernels.h>

namespace rwkv {
namespace cuda {

Tensor layer_norm_op(const Tensor &x, const Tensor &weight, const Tensor &bias);
Tensor group_norm_op(const Tensor& x, int num_groups, const Tensor& weight, const Tensor& bias);
void gemm_cublas_tensor(const Tensor &a, const Tensor &b, Tensor &c);

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

struct PowScalar {
  const half *w;
  const float T;
  /* out */ float *ws;

  __device__ void operator()(int i) const {
    ws[i] = __powf(__half2float(w[i]), T);
  }
};

struct PowTensor {
  const half *w;
  const half *ind;
  float *pow_w;

  __device__ void operator()(int i) const {
    pow_w[i] = __powf(__half2float(w[i]), __half2float(ind[i]));
  }
};

Tensor _ATT_SEQ_V5(const Tensor &x, const Tensor &s, const Tensor &ln_w,
                   const Tensor &ln_b,
                   const Tensor& lx_w, const Tensor& lx_b, const Tensor& sx, const Tensor& k_mix,
            const Tensor& v_mix, const Tensor& r_mix, const Tensor& kw,
            Tensor& kx, const Tensor& vw, Tensor& vx, const Tensor& rw,
            Tensor& rx, const Tensor& ow, const Tensor& t_first,
            Tensor& k,
            const Tensor& t_decay, Tensor& v, Tensor& r, Tensor& decayed_s,
            Tensor& x_plus_out, Tensor& s_out, Tensor& out_temp1, Tensor& out_temp2,
            LengthType H, LengthType S, LengthType T) {
  Tensor xx = cuda::layer_norm_op(x, ln_w, ln_b);
  Tensor converted_sx = cat(unsqueeze(sx, -1), xx.slice({Range(0, 1, -1), Range::All}), 0);

  element_wise(Mix{xx.data_ptr<half>(), sx.data_ptr<half>(),
                   k_mix.data_ptr<half>(), v_mix.data_ptr<half>(),
                   r_mix.data_ptr<half>(), kx.data_ptr<half>(),
                   vx.data_ptr<half>(), rx.data_ptr<half>()},
               x.numel());

  Tensor w = rwkv::reshape(t_decay, Shape({-1, 1}));  // [H, 1]
  Tensor u = rwkv::reshape(t_first, Shape({-1, 1}));
  Tensor ws = Tensor::Empty(w.shape(), DType::kFloat32, w.device());
  element_wise(PowScalar{w.data_ptr<half>(), static_cast<float>(T), ws.data_ptr<float>()}, w.numel());
  ws = rwkv::reshape(ws, Shape({H, 1, 1}));
  Tensor ind = Tensor::Arange(static_cast<int>(T - 1), -1, -1, w.dtype(), w.device()).unsqueeze(0).repeat({H, 1}); // [H, T]
  w = w.repeat({1, T});
  Tensor pow_w = Tensor::Empty(w.shape(), DType::kFloat32, w.device());
  element_wise(PowTensor{w.data_ptr<half>(), ind.data_ptr<half>(),
                         pow_w.data_ptr<float>()},
               w.numel());
  w = pow_w;

  Tensor wk = rwkv::reshape(w, Shape({H, 1, T}));
  Tensor wb = wk.transpose(-2, -1).flip(1);
  w = cat(w.slice({Range::All, Range(1, 1, -1)}), u, 1);
  w = pad(w, {0, T}, "constant");
  w = w.repeat({T});
  w = w.slice({Range::All, Range(-1, -1, -T)})
          .reshape(Shape({-1, T, 2 * T - 1}));
  w = w.slice({Range::All, Range::All, Range(T - 1, 1, w.size(2))})
          .reshape(Shape({H, T, T}));

  gemm_cublas_tensor(rx, rw, r);
  r = r.view({T, H, S}).transpose(0, 1);
  gemm_cublas_tensor(kx, kw, k);
  k = k.view({T, H, S}).transpose(0, 1).transpose(-2, -1);
  gemm_cublas_tensor(vx, vw, v);
  v = v.view({T, H, S}).transpose(0, 1);

  gemm_cublas_tensor(r, k, out_temp1);
  out_temp1 = out_temp1 * w;
  gemm_cublas_tensor(out_temp1, v, out_temp2);
  gemm_cublas_tensor(r, s, out_temp1);
  out_temp1 = out_temp1 * wb;
  Tensor out = out_temp2 + out_temp1;
  gemm_cublas_tensor(k * wk, v, out_temp1);
  s_out = ws * s + out_temp1;

  out = out.transpose(0, 1).reshape({T, H * S});
  out = group_norm_op(out, H, lx_w, lx_b);
  out = cast_dtype(out, x.dtype());
  gemm_cublas_tensor(out, ow, x_plus_out);
  x_plus_out = x + x_plus_out;

  // w = t_decay.reshape(-1, 1)
  // u = t_first.reshape(-1, 1)
  // ws = w.pow(T).reshape(H, 1, 1)
  // ind = torch.arange(T-1, -1, -1, device=w.device).unsqueeze(0).repeat(H, 1)
  // w = w.repeat(1, T).pow(ind)
  // wk = w.reshape(H, 1, T)
  // wb = wk.transpose(-2, -1).flip(1)
  // w = torch.cat([w[:, 1:], u], dim=1)
  // w = F.pad(w, (0, T))
  // w = torch.tile(w, [T])
  // w = w[:, :-T].reshape(-1, T, 2 * T - 1)
  // w = w[:, :, T-1:].reshape(H, T, T)

  // r = gemm(rx, rw, output_dtype=torch.float32).view(T, H, S).transpose(0, 1)
  // k = gemm(kx, kw, output_dtype=torch.float32).view(T, H, S).transpose(0, 1).transpose(-2, -1)
  // v = gemm(vx, vw, output_dtype=torch.float32).view(T, H, S).transpose(0, 1)

  // out = ((r @ k) * w) @ v + (r @ s) * wb
  // s = ws * s + (k * wk) @ v
  
  // out = out.transpose(0, 1).contiguous().reshape(T, H*S)
  // out = F.group_norm(out, num_groups=H, weight=lx_w, bias=lx_b)
  // out = out.to(dtype=x.dtype)
  // out = gemm(out, ow)

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
  auto k = Tensor::Empty({kw.size(0)}, DType::kFloat32, x.device());
  auto v = Tensor::Empty({vw.size(0)}, DType::kFloat32, x.device());
  auto r = Tensor::Empty({rw.size(0)}, DType::kFloat32, x.device());
  auto x_plus_out = Tensor::Empty(x.sizes(), x.dtype(), x.device());

  auto H = t_decay.size(0);
  auto S = x.size(x.shape().size() - 1) / H;

  auto decayed_s = Tensor::Empty(s.sizes(), DType::kFloat32, s.device());
  auto s_out = Tensor::Empty(s.sizes(), DType::kFloat32, s.device());
  auto out_temp1 = Tensor::Empty({s.size(0), 1, s.size(2)}, DType::kFloat32, s.device());
  auto out_temp2 = Tensor::Empty(s.sizes(), DType::kFloat32, s.device());

  Tensor xx = _ATT_SEQ_V5(x, s, ln_w, ln_b, lx_w, lx_b, sx, k_mix, v_mix, r_mix,
                          kw, kx, vw, vx, rw, rx, ow, t_first, k, t_decay, v, r,
                          decayed_s, x_plus_out, s_out, out_temp1, out_temp2, H, S, -1);
  return std::make_tuple(x_plus_out, xx, decayed_s);
}

KernelRegister att_seq_v5_reg("att_seq_v5", Device::kCUDA, att_seq_v5);

}
}