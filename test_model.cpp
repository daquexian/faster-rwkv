#include "model.h"

#include <gtest/gtest.h>

TEST(Model, load) {
  // rwkv::Model model("/home/dev/files/repos/rwkv.neo/models/rwkv-4-1.5b.fr", "cuda fp16");
  rwkv::Model model("../rwkv-4-0.1b-fp16.fr", "cuda fp16");
  auto output = rwkv::Copy(model.Run(0), rwkv::Device::kCPU);
  auto output_ptr = output.data_ptr<float>();
  EXPECT_FLOAT_EQ(output_ptr[0], -0.19592285);
  EXPECT_FLOAT_EQ(output_ptr[9], -10.234375);
}

TEST(Model, ncnn) {
  // rwkv::Model model("/home/dev/files/repos/rwkv.neo/models/rwkv-4-1.5b.fr", "cuda fp16");
  rwkv::Model model("../rwkv", "ncnn fp16");
  auto output = rwkv::Copy(model.Run(0), rwkv::Device::kCPU);
  auto output_ptr = output.data_ptr<float>();
  EXPECT_FLOAT_EQ(output_ptr[0], -0.19592285);
  EXPECT_FLOAT_EQ(output_ptr[9], -10.234375);
}
