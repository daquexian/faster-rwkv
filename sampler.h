#pragma once

namespace rwkv {
class Sampler {
public:
  int Sample(const float *ptr, int len);
};

} // namespace rwkv


