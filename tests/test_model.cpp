#include <kernels/export-ncnn/kernels.h>
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
  rwkv::ncnnmeta::ExportModel(model_dir + "/RWKV-4-World-0.1B-v1-20230520-ctx4096-fp32.fr", rwkv::DType::kFloat16,
                              "/tmp/rwkv-4-0.1b-ncnn");
  rwkv::Model model("/tmp/rwkv-4-0.1b-ncnn", "ncnn auto");
  auto output = rwkv::Copy(model.Run(0), rwkv::Device::kCPU);
  auto output_ptr = output.data_ptr<float>();
  // NOTE: different machines may have different results
  EXPECT_LT(output_ptr[0], -0.048);
  EXPECT_GT(output_ptr[0], -0.059);
  EXPECT_LT(output_ptr[9], -9.7);
  EXPECT_GT(output_ptr[9], -10.0);
  output = rwkv::Copy(model.Run(0), rwkv::Device::kCPU);
  output_ptr = output.data_ptr<float>();
  EXPECT_LT(output_ptr[0], -1.28);
  EXPECT_GT(output_ptr[0], -1.31);
  EXPECT_LT(output_ptr[9], -9.1);
  EXPECT_GT(output_ptr[9], -9.4);
}

TEST(Model, ncnn_fp16_v5) {
  const std::string model_dir(std::getenv("FR_MODEL_DIR"));
  rwkv::ncnnmeta::ExportModel(model_dir + "/RWKV-5-World-0.1B-v1-20230803-ctx4096-fp32.fr", rwkv::DType::kFloat16,
                              "/tmp/rwkv-5-0.1b-ncnn");
  rwkv::Model model("/tmp/rwkv-5-0.1b-ncnn", "ncnn auto");
  auto output = rwkv::Copy(model.Run(0), rwkv::Device::kCPU);
  auto output_ptr = output.data_ptr<float>();
  EXPECT_LT(output_ptr[0], -7.0);
  EXPECT_GT(output_ptr[0], -7.1);
  EXPECT_LT(output_ptr[9], -15.75);
  EXPECT_GT(output_ptr[9], -15.9);
  output = rwkv::Copy(model.Run(0), rwkv::Device::kCPU);
  output_ptr = output.data_ptr<float>();
  EXPECT_LT(output_ptr[0], -7.3);
  EXPECT_GT(output_ptr[0], -7.5);
  EXPECT_LT(output_ptr[9], -14.8);
  EXPECT_GT(output_ptr[9], -15.0);
}

TEST(Model, ncnn_int8) {
  const std::string model_dir(std::getenv("FR_MODEL_DIR"));
  rwkv::ncnnmeta::ExportModel(model_dir + "/RWKV-4-World-0.1B-v1-20230520-ctx4096-fp32.fr", rwkv::DType::kInt8,
                              "/tmp/rwkv-4-0.1b-ncnn-int8");
  rwkv::Model model("/tmp/rwkv-4-0.1b-ncnn-int8", "ncnn auto");
  auto output = rwkv::Copy(model.Run(0), rwkv::Device::kCPU);
  auto output_ptr = output.data_ptr<float>();
  // NOTE: different machines may have different results
  EXPECT_LT(output_ptr[0], -0.20);
  EXPECT_GT(output_ptr[0], -0.30);
  EXPECT_LT(output_ptr[9], -10.0);
  EXPECT_GT(output_ptr[9], -10.5);
  output = rwkv::Copy(model.Run(0), rwkv::Device::kCPU);
  output_ptr = output.data_ptr<float>();
  EXPECT_LT(output_ptr[0], -1.34);
  EXPECT_GT(output_ptr[0], -1.39);
  EXPECT_LT(output_ptr[9], -9.1);
  EXPECT_GT(output_ptr[9], -9.4);
}

// TODO: add int4 reference implementation
// TEST(Model, ncnn_int4) {
//   const std::string model_dir(std::getenv("FR_MODEL_DIR"));
//   rwkv::ncnnmeta::ExportModel(model_dir + "/RWKV-4-World-0.1B-v1-20230520-ctx4096-fp32.fr", rwkv::DType::kInt4,
//                               "/tmp/rwkv-4-0.1b-ncnn-int4");
//   rwkv::Model model("/tmp/rwkv-4-0.1b-ncnn-int4", "ncnn auto");
//   auto output = rwkv::Copy(model.Run(0), rwkv::Device::kCPU);
//   auto output_ptr = output.data_ptr<float>();
//   // NOTE: different machines may have different results
//   EXPECT_LT(output_ptr[0], -0.20);
//   EXPECT_GT(output_ptr[0], -0.30);
//   EXPECT_LT(output_ptr[9], -10.0);
//   EXPECT_GT(output_ptr[9], -10.5);
//   output = rwkv::Copy(model.Run(0), rwkv::Device::kCPU);
//   output_ptr = output.data_ptr<float>();
//   EXPECT_LT(output_ptr[0], -1.34);
//   EXPECT_GT(output_ptr[0], -1.39);
//   EXPECT_LT(output_ptr[9], -9.1);
//   EXPECT_GT(output_ptr[9], -9.4);
// }

#endif
