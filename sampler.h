#pragma once

#include <tensor.h>

namespace rwkv {
class Sampler {
public:
  int Sample(const Tensor& logits, float temperature, int top_k, float top_p);
};

} // namespace rwkv


