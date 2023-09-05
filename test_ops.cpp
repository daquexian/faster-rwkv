#include <kernels/kernels.h>
#include <tensor.h>

#include <gtest/gtest.h>

// TODO: add more op tests

#ifdef FR_ENABLE_CUDA
TEST(RWKV, cuda_scalar_div_fp16) {
  auto x = rwkv::Tensor::Empty({256}, rwkv::DType::kFloat16, rwkv::Device::kCUDA);
  rwkv::fill_(x, 1.0f);
  rwkv::scalar_div_(x, 2.0f);
  x = rwkv::cast_dtype(rwkv::Copy(x, rwkv::Device::kCPU), rwkv::DType::kFloat32);
  auto x_ptr = x.data_ptr<float>();
  EXPECT_EQ(x_ptr[0], 0.5f);
}
#endif
