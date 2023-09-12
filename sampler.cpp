#include "sampler.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <random>

#include "check.h"

namespace rwkv {

static const bool kDebug = std::getenv("FR_DEBUG") != nullptr;

int Sampler::Sample(const float *ptr, int len, float temperature, int top_k, float top_p) {
  RV_CHECK(len >= 1);
  if (kDebug) {
    std::cout << "Sample: len=" << len << ", temperature=" << temperature
              << ", top_k=" << top_k << ", top_p=" << top_p << std::endl;
  }

  // softmax
  std::vector<std::pair<int, float>> id_and_probs;
  id_and_probs.reserve(len);
  const float max_logit = *std::max_element(ptr, ptr + len);
  float sum = 0;
  for (int i = 0; i < len; i++) {
    id_and_probs.push_back({i, std::exp((ptr[i] - max_logit) / temperature)});
    sum += id_and_probs.back().second;
  }
  for (int i = 0; i < len; i++) {
    id_and_probs[i].second /= sum;
  }

  // sort
  std::sort(id_and_probs.begin(), id_and_probs.end(), [&](auto p1, auto p2) {
    return p1.second > p2.second;
  });

  // top-k
  if (top_k > 0) {
    len = std::min(len, top_k);
  }
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
      std::cout << "(" << id_and_probs[i].first << ", " << id_and_probs[i].second << "), ";
    }
    if (len > 10) {
      std::cout << "...";
    }
    std::cout << "]" << std::endl;
  }
  static std::default_random_engine generator(time(nullptr));
  std::vector<float> probs;
  probs.reserve(len);
  for (int i = 0; i < len; i++) {
    probs.push_back(id_and_probs[i].second);
  }
  std::discrete_distribution<int> distribution(probs.begin(), probs.end());
  int idx = distribution(generator);
  if (kDebug) {
    std::cout << "Sample: idx=" << idx << ", id=" << id_and_probs[idx].first << std::endl;
  }
  return id_and_probs[idx].first;
}
} // namespace rwkv
