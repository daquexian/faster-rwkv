#include <kernels/registry.h>
#include <tensor.h>

namespace rwkv {
namespace onnxmeta {

Tensor layernorm(const Tensor& x, const Tensor& weight, const Tensor& bias) {
  return KernelRegistry::Instance().Get<decltype(layernorm)*>("layernorm", x.device())(x, weight, bias);
}

Tensor matmul(const Tensor& a, const Tensor& b) {
  return KernelRegistry::Instance().Get<decltype(matmul)*>("matmul", a.device())(a, b);
}

Tensor add(const Tensor& x, const Tensor& y) {
  return KernelRegistry::Instance().Get<decltype(add)*>("add", x.device())(x, y);
}

Tensor rsub_scalar(float x, const Tensor& y) {
  return KernelRegistry::Instance().Get<Tensor(*)(float, const Tensor&)>("rsub_scalar", y.device())(x, y);
}

Tensor sub(const Tensor& x, const Tensor& y) {
  return KernelRegistry::Instance().Get<Tensor(*)(const Tensor&, const Tensor&)>("sub", y.device())(x, y);
}

Tensor mul(const Tensor& x, const Tensor& y) {
  return KernelRegistry::Instance().Get<decltype(mul)*>("mul", x.device())(x, y);
}

Tensor div(const Tensor& x, const Tensor& y) {
  return KernelRegistry::Instance().Get<decltype(div)*>("div", x.device())(x, y);
}

Tensor exp(const Tensor& x) {
  return KernelRegistry::Instance().Get<decltype(exp)*>("exp", x.device())(x);
}

Tensor relu(const Tensor& x) {
  return KernelRegistry::Instance().Get<decltype(relu)*>("relu", x.device())(x);
}

Tensor sigmoid(const Tensor& x) {
  return KernelRegistry::Instance().Get<decltype(sigmoid)*>("sigmoid", x.device())(x);
}

Tensor maximum(const Tensor& x, const Tensor& y) {
  return KernelRegistry::Instance().Get<decltype(maximum)*>("maximum", x.device())(x, y);
}

}
}
