#include "kernels/kernels.h"
#include <sampler.h>

#include <gtest/gtest.h>

using namespace rwkv;

std::vector<float> distribution(const std::vector<float> &logits,
                                float temperature, int top_k, float top_p) {
  Sampler sampler;

  int length = logits.size();
  auto logits_t = Tensor::Empty({length}, DType::kFloat32, Device::kCPU);
  for (int i = 0; i < length; i++) {
    logits_t.data_ptr<float>()[i] = logits[i];
  }

  std::vector<float> n;
  n.resize(length);

  static const int N = 100000;
  for (int i = 0; i < N; i++) {
    n[sampler.Sample(logits_t, temperature, top_k, top_p)]++;
  }
  for (int i = 0; i < length; i++) {
    n[i] /= N;
  }
  return n;
}

TEST(Sampler, greedy) {
  {
    std::vector<float> logits = {0, 1, 2};
    auto n = distribution(logits, /*temperature=*/1, /*top_k=*/0, /*top_p=*/0);
    EXPECT_EQ(n.size(), logits.size());
    EXPECT_EQ(n[0], 0);
    EXPECT_EQ(n[1], 0);
    EXPECT_EQ(n[2], 1);
  }
  {
    std::vector<float> logits = {0, 0.5, 0.499};
    auto n =
        distribution(logits, /*temperature=*/1, /*top_k=*/1, /*top_p=*/0.999);
    EXPECT_EQ(n[0], 0);
    EXPECT_EQ(n[1], 1);
    EXPECT_EQ(n[2], 0);
  }
  {
    std::vector<float> logits = {0, 0.5, 0.499};
    auto n = distribution(logits, /*temperature=*/9999, /*top_k=*/1,
                          /*top_p=*/0.999);
    EXPECT_EQ(n[0], 0);
    EXPECT_EQ(n[1], 1);
    EXPECT_EQ(n[2], 0);
  }
  {
    std::vector<float> logits = {0, 1, 2};
    auto n =
        distribution(logits, /*temperature=*/1, /*top_k=*/999, /*top_p=*/0);
    EXPECT_EQ(n[0], 0);
    EXPECT_EQ(n[1], 0);
    EXPECT_EQ(n[2], 1);
  }
}

TEST(Sampler, non_greedy) {
  {
    std::vector<float> logits = {0, 0, 0};
    auto n =
        distribution(logits, /*temperature=*/9999, /*top_k=*/0, /*top_p=*/1);
    EXPECT_NEAR(n[0], 0.333, 0.01);
    EXPECT_NEAR(n[1], 0.333, 0.01);
    EXPECT_NEAR(n[2], 0.333, 0.01);
  }

  std::vector<float> logits = {0.4, 0.5, 0.1};
  Tensor probs_t = softmax(Tensor::FromPtr(logits.data(), {static_cast<long>(logits.size())}, DType::kFloat32, Device::kCPU), 1);
  float* probs = probs_t.data_ptr<float>();
  {
    auto n = distribution(logits, /*temperature=*/1, /*top_k=*/0, /*top_p=*/1);
    EXPECT_NEAR(n[0], probs[0], 0.01);
    EXPECT_NEAR(n[1], probs[1], 0.01);
    EXPECT_NEAR(n[2], probs[2], 0.01);
  }
  {
    auto n =
        distribution(logits, /*temperature=*/1, /*top_k=*/999, /*top_p=*/ probs[1]);
    EXPECT_EQ(n[0], 0);
    EXPECT_EQ(n[1], 1);
    EXPECT_EQ(n[2], 0);
  }
  {
    auto n =
        distribution(logits, /*temperature=*/1, /*top_k=*/999, /*top_p=*/ probs[1] + 0.001);
    EXPECT_NEAR(n[0], probs[0] / (probs[0] + probs[1]), 0.01);
    EXPECT_NEAR(n[1], probs[1] / (probs[0] + probs[1]), 0.01);
    EXPECT_EQ(n[2], 0);
  }
  {
    auto n =
        distribution(logits, /*temperature=*/1, /*top_k=*/1, /*top_p=*/ 1);
    EXPECT_EQ(n[0], 0);
    EXPECT_EQ(n[1], 1);
    EXPECT_EQ(n[2], 0);
  }
  {
    auto n =
        distribution(logits, /*temperature=*/1, /*top_k=*/2, /*top_p=*/ 1);
    EXPECT_NEAR(n[0], probs[0] / (probs[0] + probs[1]), 0.01);
    EXPECT_NEAR(n[1], probs[1] / (probs[0] + probs[1]), 0.01);
    EXPECT_EQ(n[2], 0);
  }
}

TEST(Sampler, compare_with_python) {
  {
    std::vector<float> logits = {3, -5, 0, 4, -1.};
    auto n =
        distribution(logits, /*temperature=*/2, /*top_k=*/0, /*top_p=*/0.8);
    EXPECT_NEAR(n[0], 0.378, 0.01);
    EXPECT_EQ(n[1], 0);
    EXPECT_EQ(n[2], 0);
    EXPECT_NEAR(n[3], 0.622, 0.01);
    EXPECT_EQ(n[4], 0);
  }
  {
    std::vector<float> logits = {3, -5, 0, 4, -1.};
    auto n =
        distribution(logits, /*temperature=*/5, /*top_k=*/0, /*top_p=*/1.0);
    EXPECT_NEAR(n[0], 0.293, 0.02);
    EXPECT_NEAR(n[1], 0.059, 0.02);
    EXPECT_NEAR(n[2], 0.160, 0.02);
    EXPECT_NEAR(n[3], 0.357, 0.02);
    EXPECT_NEAR(n[4], 0.139, 0.02);
  }
}
