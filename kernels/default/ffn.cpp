#include <kernels/kernels.h>
#include <kernels/registry.h>
#include <tensor.h>

namespace rwkv {
namespace def {

std::tuple<Tensor, Tensor> ffn(const Tensor &x, const Tensor &sx,
                               const Tensor &ln_w, const Tensor &ln_b,
                               const Tensor &k_mix, const Tensor &r_mix,
                               const Tensor &kw, const Tensor &vw,
                               const Tensor &rw) {
  auto xx = layernorm(x, ln_w, ln_b);
  // auto [kx, rx] = channel_mix(xx, sx, k_mix, r_mix);
  auto kx = xx * k_mix + sx * (1 - k_mix);
  auto rx = xx * r_mix + sx * (1 - r_mix);

  auto r = sigmoid(matmul(rx, rw));
  auto vx = relu(matmul(kx, kw));
  vx = vx * vx;
  auto out = r * matmul(vx, vw);
  return {x + out, xx};
}

KernelRegister ffn_reg_2("ffn", Device::kONNXMeta, ffn);

} // namespace def
} // namespace rwkv
