#include <kernels/ncnn-meta/kernels.h>
#include <model.h>

#include <gtest/gtest.h>

#ifdef FR_ENABLE_CUDA
// TODO: generate models
TEST(Model, cuda_fp16) {
  const std::string model_dir(std::getenv("FR_MODEL_DIR"));
  rwkv::Model model(model_dir + "/RWKV-4-World-0.1B-v1-20230520-ctx4096-fp16.fr", "cuda fp16");
  auto output = rwkv::Copy(model.Run(0), rwkv::Device::kCPU);
  auto output_ptr = output.data_ptr<float>();
  EXPECT_FLOAT_EQ(output_ptr[0], -0.19592285);
  EXPECT_FLOAT_EQ(output_ptr[9], -10.234375);
  output = rwkv::Copy(model.Run(0), rwkv::Device::kCPU);
  output_ptr = output.data_ptr<float>();
  EXPECT_FLOAT_EQ(output_ptr[0], -1.5488281);
  EXPECT_FLOAT_EQ(output_ptr[9], -9.640625);
}
#endif

#ifdef FR_ENABLE_NCNN
TEST(Model, ncnn_fp16) {
  const std::string model_dir(std::getenv("FR_MODEL_DIR"));
  rwkv::ncnnmeta::ExportModel(model_dir + "/RWKV-4-World-0.1B-v1-20230520-ctx4096-fp32.fr",
                              "/tmp/rwkv-4-0.1b-ncnn");
  rwkv::Model model("/tmp/rwkv-4-0.1b-ncnn", "ncnn fp16");
  auto output = rwkv::Copy(model.Run(0), rwkv::Device::kCPU);
  auto output_ptr = output.data_ptr<float>();
  EXPECT_FLOAT_EQ(output_ptr[0], -0.056640625);
  EXPECT_FLOAT_EQ(output_ptr[9], -9.875);
  output = rwkv::Copy(model.Run(0), rwkv::Device::kCPU);
  output_ptr = output.data_ptr<float>();
  EXPECT_FLOAT_EQ(output_ptr[0], -1.28125);
  EXPECT_FLOAT_EQ(output_ptr[9], -9.25);
}

TEST(Model, ncnn_fp16_v5) {
  const std::string model_dir(std::getenv("FR_MODEL_DIR"));
  rwkv::ncnnmeta::ExportModel(model_dir + "/RWKV-5-World-0.1B-v1-20230803-ctx4096-fp32.fr",
                              "/tmp/rwkv-5-0.1b-ncnn");
  rwkv::Model model("/tmp/rwkv-5-0.1b-ncnn", "ncnn fp16");
  auto output = rwkv::Copy(model.Run(0), rwkv::Device::kCPU);
  auto output_ptr = output.data_ptr<float>();
  EXPECT_FLOAT_EQ(output_ptr[0], -7.0625);
  EXPECT_FLOAT_EQ(output_ptr[9], -15.8125);
  output = rwkv::Copy(model.Run(0), rwkv::Device::kCPU);
  output_ptr = output.data_ptr<float>();
  EXPECT_FLOAT_EQ(output_ptr[0], -7.4375);
  EXPECT_FLOAT_EQ(output_ptr[9], -14.9375);
}
#endif
