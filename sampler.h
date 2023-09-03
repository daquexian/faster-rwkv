#pragma once

namespace rwkv {
class Sampler {
public:
  int Sample(const float *ptr, int len, float temperature=1.f, int top_k=0, float top_p=0.85f);
};

} // namespace rwkv


