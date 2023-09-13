#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "element_wise.cuh"
#include "layer_norm.cuh"
#include <tensor.h>
#include <kernels/registry.h>
#include <kernels/kernels.h>

namespace rwkv {
namespace cuda {

Tensor layer_norm_op(const Tensor &x, const Tensor &weight, const Tensor &bias);

Tensor _ATT_SEQ_V5(const Tensor &x, const Tensor &s, const Tensor &ln_w,
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
  Tensor converted_sx = cat(unsqueeze(sx, -1), xx.slice({Range(0, 1, -1), Range::All}));


  // element_wise(Mix{xx.data_ptr<half>(), sx.data_ptr<half>(),
  //                  k_mix.data_ptr<half>(), v_mix.data_ptr<half>(),
  //                  r_mix.data_ptr<half>(), kx.data_ptr<half>(),
  //                  vx.data_ptr<half>(), rx.data_ptr<half>()},
  //              x.numel());

  // gemm_cublas_tensor(kx, kw, k);
  // gemm_cublas_tensor(vx, vw, v);
  // gemm_cublas_tensor(rx, rw, r);

  // r = r.view({H, 1, S});
  // k = k.view({H, S, 1});
  // v = v.view({H, 1, S});

  // gemm_cublas_tensor(k, v, a);

  // element_wise(OneV5MulAdd{static_cast<int>(s.size(1) * s.size(2)), t_first.data_ptr<float>(), a.data_ptr<float>(), s.data_ptr<float>(), t_decay.data_ptr<float>(), out_temp2.data_ptr<float>(), decayed_s.data_ptr<float>()}, s.numel());

  // gemm_cublas_tensor(r, out_temp2, out_temp1);
  // out_temp1 = out_temp1.flatten().unsqueeze(0);

  // Tensor out_temp3 = cuda::group_norm_op(out_temp1, H, lx_w, lx_b).flatten();
  
  // Tensor out_temp4 = cast_dtype(out_temp3, DType::kFloat16);
  // gemm_cublas_tensor(out_temp4, ow, x_plus_out);
  // element_wise(InplaceAdd{x_plus_out.data_ptr<half>(), x.data_ptr<half>()},
  //              x.numel());

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
  auto a = Tensor::Empty(s.sizes(), DType::kFloat32, s.device());
  auto out_temp1 = Tensor::Empty({s.size(0), 1, s.size(2)}, DType::kFloat32, s.device());
  auto out_temp2 = Tensor::Empty(s.sizes(), DType::kFloat32, s.device());

  Tensor xx = _ATT_SEQ_V5(x, s, ln_w, ln_b, lx_w, lx_b, sx, k_mix, v_mix, r_mix,
                          kw, kx, vw, vx, rw, rx, ow, t_first, k, t_decay, v, r,
                          decayed_s, x_plus_out, a, out_temp1, out_temp2, H, S);
  return std::make_tuple(x_plus_out, xx, decayed_s);
}

}
}