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

// hand-made distribution to get the same result on different platforms
// https://stackoverflow.com/questions/48730363/if-we-seed-c11-mt19937-as-the-same-on-different-machines-will-we-get-the-same
int distribution(std::vector<float> probs, std::minstd_rand0 &generator) {
  float sum = std::accumulate(probs.begin(), probs.end(), 0.f);
  float random_value = 1. * (generator() - generator.min()) /
                       (generator.max() - generator.min()) * sum;
  float cumsum = 0;
  for (int i = 0; i < probs.size(); i++) {
    cumsum += probs[i];
    if (cumsum >= random_value) {
      return i;
    }
  }
  RV_UNIMPLEMENTED();
}

Sampler::Sampler() {
  _generator.seed(std::random_device()());
}

int Sampler::Sample(const Tensor &logits, float temperature, int top_k,
                    float top_p) {
  if (kDebug) {
    std::cout << "Sample: temperature=" << temperature << ", top_k=" << top_k
              << ", top_p=" << top_p << std::endl;
  }

  temperature = std::clamp(temperature, 0.1f, 5.f);

  // softmax (temperature is applied in another place)
  auto probs = softmax(logits, 1.f);
  std::vector<std::pair<int, float>> id_and_probs;
  id_and_probs.reserve(probs.numel());
  for (int i = 0; i < probs.numel(); i++) {
    id_and_probs.push_back({i, probs.data_ptr<float>()[i]});
  }

  // sort
  std::sort(id_and_probs.begin(), id_and_probs.end(),
            [&](auto p1, auto p2) { return p1.second > p2.second; });

  int len = id_and_probs.size();

  // top-p
  float cumsum = 0;
  for (int i = 0; i < len; i++) {
    cumsum += id_and_probs[i].second;
    if (cumsum >= top_p) {
      len = i + 1;
      break;
    }
  }

  // top-k
  if (top_k > 0) {
    len = std::min(len, top_k);
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

  std::vector<float> top_probs;
  top_probs.reserve(len);
  for (int i = 0; i < len; i++) {
    top_probs.push_back(std::pow(id_and_probs[i].second, 1.f / temperature));
  }

  // random choice
  int idx = distribution(top_probs, _generator);
  if (kDebug) {
    std::cout << "Sample: idx=" << idx << ", id=" << id_and_probs[idx].first
              << std::endl;
  }
  return id_and_probs[idx].first;
}

void Sampler::set_seed(int seed) { _generator.seed(seed); }

} // namespace rwkv
