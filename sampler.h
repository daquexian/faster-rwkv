#pragma once

#include <random>

#include <tensor.h>

namespace rwkv {
class Sampler {
public:
  Sampler();
  int Sample(const Tensor& logits, float temperature, int top_k, float top_p);
  void set_seed(int seed);
private:
  std::minstd_rand0 _generator;
};

} // namespace rwkv


