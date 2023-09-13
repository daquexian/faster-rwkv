#include "sampler.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <check.h>
#include <kernels/kernels.h>
#include <tensor.h>

namespace rwkv {

static const bool kDebug = std::getenv("FR_DEBUG") != nullptr;

int Sampler::Sample(const Tensor &logits, float temperature, int top_k,
                    float top_p) {
  if (kDebug) {
    std::cout << "Sample: temperature=" << temperature << ", top_k=" << top_k
              << ", top_p=" << top_p << std::endl;
  }

  // softmax
  auto probs = softmax(logits, temperature);
  std::vector<std::pair<int, float>> id_and_probs;
  id_and_probs.reserve(probs.numel());
  for (int i = 0; i < probs.numel(); i++) {
    id_and_probs.push_back({i, probs.data_ptr<float>()[i]});
  }

  // sort
  std::sort(id_and_probs.begin(), id_and_probs.end(),
            [&](auto p1, auto p2) { return p1.second > p2.second; });

  int len = id_and_probs.size();

  // top-k
  if (top_k > 0) {
    len = std::min(len, top_k);
  }

  // top-p
  float cumsum = 0;
  for (int i = 0; i < len; i++) {
    cumsum += id_and_probs[i].second;
    if (cumsum > top_p) {
      len = i + 1;
      break;
    }
  }
  if (kDebug) {
    std::cout << "Sample: len=" << len << ", cumsum=" << cumsum << ", probs=[";
    for (int i = 0; i < std::min(len, 10); i++) {
      std::cout << "(" << id_and_probs[i].first << ", "
                << id_and_probs[i].second << "), ";
    }
    if (len > 10) {
      std::cout << "...";
    }
    std::cout << "]" << std::endl;
  }

  static std::default_random_engine generator(time(nullptr));
  std::vector<float> top_probs;
  top_probs.reserve(len);
  for (int i = 0; i < len; i++) {
    top_probs.push_back(id_and_probs[i].second);
  }

  // random choice
  std::discrete_distribution<> distribution(top_probs.begin(), top_probs.end());
  int idx = distribution(generator);
  if (kDebug) {
    std::cout << "Sample: idx=" << idx << ", id=" << id_and_probs[idx].first
              << std::endl;
  }
  return id_and_probs[idx].first;
}
} // namespace rwkv
