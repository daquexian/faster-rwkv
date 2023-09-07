#pragma once

#include <kernels/allocator.h>
#include <kernels/registry.h>
#include <tensor.h>

namespace rwkv {
//     def att_one(self, x, sx, aa, bb, pp, ln_w, ln_b, k_mix, v_mix, r_mix,
//     t_decay, t_first, kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry,
//     rmx, rrx, rmy, rry, omx, orx, omy, ory):

inline std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
att(const Tensor &x, const Tensor &sx, const Tensor &aa, const Tensor &bb,
    const Tensor &pp, const Tensor &ln_w, const Tensor &ln_b,
    const Tensor &k_mix, const Tensor &v_mix, const Tensor &r_mix,
    const Tensor &t_decay, const Tensor &t_first, const Tensor &kw,
    const Tensor &vw, const Tensor &rw, const Tensor &ow) {
  auto tmp = KernelRegistry::Instance().Get<decltype(att) *>("att", x.device());
  return tmp(x, sx, aa, bb, pp, ln_w, ln_b, k_mix, v_mix, r_mix, t_decay,
             t_first, kw, vw, rw, ow);
}

inline std::tuple<Tensor, Tensor, Tensor>
att_one_v5(const Tensor &x, const Tensor &sx, const Tensor &s,
           const Tensor &ln_w, const Tensor &ln_b, const Tensor &lx_w,
           const Tensor &lx_b, const Tensor &k_mix, const Tensor &v_mix,
           const Tensor &r_mix, const Tensor &t_decay, const Tensor &t_first,
           const Tensor &kw, const Tensor &vw, const Tensor &rw,
           const Tensor &ow) {
  auto tmp = KernelRegistry::Instance().Get<decltype(att_one_v5) *>(
      "att_one_v5", x.device());
  return tmp(x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, t_decay,
             t_first, kw, vw, rw, ow);
}

inline std::tuple<Tensor, Tensor, Tensor>
att_one_v5_1(const Tensor &x, const Tensor &sx, const Tensor &s,
             const Tensor &ln_w, const Tensor &ln_b, const Tensor &lx_w,
             const Tensor &lx_b, const Tensor &k_mix, const Tensor &v_mix,
             const Tensor &r_mix, const Tensor &g_mix, const Tensor &t_decay,
             const Tensor &t_first, const Tensor &kw, const Tensor &vw,
             const Tensor &rw, const Tensor &gw, const Tensor &ow) {
  auto tmp = KernelRegistry::Instance().Get<decltype(att_one_v5_1) *>(
      "att_one_v5_1", x.device());
  return tmp(x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, g_mix,
             t_decay, t_first, kw, vw, rw, gw, ow);
}

//         def cuda_ffn_one_fp16(self, x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw,
//         rw, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry):
inline std::tuple<Tensor, Tensor> ffn(const Tensor &x, const Tensor &sx,
                                      const Tensor &ln_w, const Tensor &ln_b,
                                      const Tensor &k_mix, const Tensor &r_mix,
                                      const Tensor &kw, const Tensor &vw,
                                      const Tensor &rw) {
  auto tmp = KernelRegistry::Instance().Get<decltype(ffn) *>("ffn", x.device());
  return tmp(x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw, rw);
}

inline Tensor cast_dtype(const Tensor &x, DType dtype) {
  return KernelRegistry::Instance().Get<decltype(cast_dtype) *>(
      "cast_dtype", x.device())(x, dtype);
}

inline Tensor &fill_(Tensor &x, float val) {
  return KernelRegistry::Instance().Get<decltype(fill_) *>("fill_",
                                                           x.device())(x, val);
}

inline Tensor &scalar_div_(Tensor &x, float val) {
  return KernelRegistry::Instance().Get<decltype(scalar_div_) *>(
      "scalar_div_", x.device())(x, val);
}

inline Tensor layernorm(const Tensor &x, const Tensor &weight,
                        const Tensor &bias) {
  return KernelRegistry::Instance().Get<decltype(layernorm) *>(
      "layernorm", x.device())(x, weight, bias);
}

inline Tensor groupnorm(const Tensor &x, int num_groups, const Tensor &weight,
                        const Tensor &bias) {
  return KernelRegistry::Instance().Get<decltype(groupnorm) *>(
      "groupnorm", x.device())(x, num_groups, weight, bias);
}

inline Tensor matmul(const Tensor &a, const Tensor &b) {
  return KernelRegistry::Instance().Get<decltype(matmul) *>("matmul",
                                                            a.device())(a, b);
}

// inline Tensor cat(const Tensor &a, const Tensor &b) {
//   return KernelRegistry::Instance().Get<decltype(cat) *>("cat",
//   a.device())(a,
//                                                                             b);
// }

// TODO:
// REGISTER_KERNEL(Tensor, add, const Tensor&, x, const Tensor&, y);

inline Tensor add(const Tensor &x, const Tensor &y) {
  return KernelRegistry::Instance().Get<decltype(add) *>(
      "add", Device::kNCNNMeta)(x, y);
}

inline Tensor sub(float x, const Tensor &y) {
  return KernelRegistry::Instance().Get<Tensor (*)(float, const Tensor &)>(
      "rsub_scalar", Device::kNCNNMeta)(x, y);
}

inline Tensor sub(const Tensor &x, const Tensor &y) {
  return KernelRegistry::Instance()
      .Get<Tensor (*)(const Tensor &, const Tensor &)>("sub", x.device())(x, y);
}

inline Tensor mul(const Tensor &x, const Tensor &y) {
  return KernelRegistry::Instance().Get<decltype(mul) *>("mul", x.device())(x,
                                                                            y);
}

inline Tensor div(const Tensor &x, const Tensor &y) {
  return KernelRegistry::Instance().Get<decltype(div) *>("div", x.device())(x,
                                                                            y);
}

inline Tensor exp(const Tensor &x) {
  return KernelRegistry::Instance().Get<decltype(exp) *>("exp", x.device())(x);
}

inline Tensor relu(const Tensor &x) {
  return KernelRegistry::Instance().Get<decltype(relu) *>("relu",
                                                          x.device())(x);
}

inline Tensor sigmoid(const Tensor &x) {
  return KernelRegistry::Instance().Get<decltype(sigmoid) *>("sigmoid",
                                                             x.device())(x);
}

inline Tensor maximum(const Tensor &x, const Tensor &y) {
  return KernelRegistry::Instance().Get<decltype(maximum) *>("maximum",
                                                             x.device())(x, y);
}

inline Tensor reshape(const Tensor &x, const Shape &shape) {
  return KernelRegistry::Instance().Get<decltype(reshape) *>(
      "reshape", x.device())(x, shape);
}

inline Tensor flatten(const Tensor &x) { return reshape(x, {x.numel()}); }

inline Tensor unsqueeze(const Tensor &x, int dim) {
  auto new_shape = x.shape();
  new_shape.insert(new_shape.begin() + dim, 1);
  return reshape(x, new_shape);
}

inline Tensor mark_as_output(const Tensor &x, const std::string &name) {
  return KernelRegistry::Instance().Get<decltype(mark_as_output) *>(
      "mark_as_output", x.device())(x, name);
}

class Model;

inline void init_model(Model *model, Device device, const std::string &path,
                       const std::string &strategy, const std::any &extra) {
  KernelRegistry::Instance().Get<decltype(init_model) *>("init_model", device)(
      model, device, path, strategy, extra);
}

inline Tensor ModelForward(const Model *model, Device device, int id,
                           std::vector<std::vector<Tensor>> &states) {
  return KernelRegistry::Instance().Get<decltype(ModelForward) *>(
      "model_forward", device)(model, device, id, states);
}

inline Allocator &allocator(Device device) {
  return KernelRegistry::Instance().Get<Allocator &(*)()>("allocator",
                                                          device)();
}

} // namespace rwkv
