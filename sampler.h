#pragma once

namespace rwkv {
class Sampler {
public:
  virtual int Sample(const float *ptr, int len) = 0;
};

class GreedySampler : public Sampler {
public:
  int Sample(const float *ptr, int len);
};
} // namespace rwkv


