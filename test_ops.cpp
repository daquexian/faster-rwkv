#include <kernels/kernels.h>
#include <tensor.h>

#include <gtest/gtest.h>

TEST(RWKV, cpu_scalar_div_fp32) {
  auto x = rwkv::Tensor::Empty({256}, rwkv::DType::kFloat32, rwkv::Device::kCUDA);
  rwkv::fill_(x, 1.0f);
  x = rwkv::Copy(x, rwkv::Device::kCPU);
  rwkv::scalar_div_(x, 2.0f);
  auto x_ptr = x.data_ptr<float>();
  EXPECT_EQ(x_ptr[0], 0.5f);
}

TEST(RWKV, cpu_scalar_div_fp16) {
  auto x = rwkv::Tensor::Empty({256}, rwkv::DType::kFloat16, rwkv::Device::kCUDA);
  rwkv::fill_(x, 1.0f);
  x = rwkv::Copy(x, rwkv::Device::kCPU);
  rwkv::scalar_div_(x, 2.0f);
  x = rwkv::cast_dtype(x, rwkv::DType::kFloat32);
  auto x_ptr = x.data_ptr<float>();
  EXPECT_EQ(x_ptr[0], 0.5f);
}
