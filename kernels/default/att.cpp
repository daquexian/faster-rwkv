#include <kernels/kernels.h>
#include <kernels/registry.h>
#include <tensor.h>

namespace rwkv {
namespace def {

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
att(const Tensor &x, const Tensor &sx, const Tensor &aa, const Tensor &bb,
    const Tensor &pp, const Tensor &ln_w, const Tensor &ln_b,
    const Tensor &k_mix, const Tensor &v_mix, const Tensor &r_mix,
    const Tensor &t_decay, const Tensor &t_first, const Tensor &kw,
    const Tensor &vw, const Tensor &rw, const Tensor &ow) {
  auto xx = layernorm(x, ln_w, ln_b);
  // auto [kx, vx, rx] = time_mix()
  auto kx = xx * k_mix + sx * (1 - k_mix);
  auto vx = xx * v_mix + sx * (1 - v_mix);
  auto rx = xx * r_mix + sx * (1 - r_mix);

  auto r = sigmoid(matmul(rx, rw));
  auto k = matmul(kx, kw);
  auto v = matmul(vx, vw);

  auto ww = t_first + k;
  auto p = maximum(pp, ww);
  auto e1 = exp(pp - p);
  auto e2 = exp(ww - p);
  auto wkv = ((e1 * aa + e2 * v) / (e1 * bb + e2));
  // if (wkv->dtype() == DType::kFloat16) ...
  ww = t_decay + pp;
  p = maximum(ww, k);
  e1 = exp(ww - p);
  e2 = exp(k - p);

  auto out = matmul(r * wkv, ow);
  return {x + out, xx, e1 * aa + e2 * v, e1 * bb + e2, p};
}

std::tuple<Tensor, Tensor, Tensor>
att_one_v5(const Tensor &x, const Tensor &sx, const Tensor &s,
           const Tensor &ln_w, const Tensor &ln_b, const Tensor &lx_w,
           const Tensor &lx_b, const Tensor &k_mix, const Tensor &v_mix,
           const Tensor &r_mix, const Tensor &t_decay, const Tensor &t_first,
           const Tensor &kw, const Tensor &vw, const Tensor &rw,
           const Tensor &ow) {

  auto xx = layernorm(x, ln_w, ln_b);
  // auto [kx, vx, rx] = time_mix()
  auto kx = xx * k_mix + sx * (1 - k_mix);
  auto vx = xx * v_mix + sx * (1 - v_mix);
  auto rx = xx * r_mix + sx * (1 - r_mix);

  auto H = t_decay.size(0);
  auto S = x.size(x.shape().size() - 1) / H;

  auto r = matmul(rx, rw).view({H, 1, S});
  auto k = matmul(kx, kw).view({H, S, 1});
  auto v = matmul(vx, vw).view({H, 1, S});

  auto a = matmul(k, v);
  auto out = matmul(r, t_first * a + s);
  auto decayed_s = a + t_decay * s;

  out = out.flatten();
  out = groupnorm(out.unsqueeze(0), static_cast<int>(H), lx_w, lx_b).flatten();
  out = matmul(out, ow);

  return {x + out, xx, decayed_s};
}

KernelRegister att_reg_2("att", Device::kONNXMeta, att);
KernelRegister att_one_v5_reg("att_one_v5", Device::kONNXMeta, att_one_v5);

} // namespace def
} // namespace rwkv
