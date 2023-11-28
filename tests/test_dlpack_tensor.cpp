#include "dlpack_tensor.h"
#include <gtest/gtest.h>

::DLDevice ConvertToDLDevice(const rwkv::Device &device) {
  ::DLDevice dl_device;
  switch (device){
      case rwkv::Device::kCPU:
          dl_device.device_type = ::DLDeviceType::kDLCPU;
          dl_device.device_id = 0;
          break;
      case rwkv::Device::kCUDA:
          dl_device.device_type = ::DLDeviceType::kDLCUDA;
          dl_device.device_id = 1;
          break;
      default:
          RV_UNIMPLEMENTED() << "Unsupport device type: " << static_cast<int>(device);
}
  return dl_device;
}

void TestToDLPack(rwkv::Device device, rwkv::DType dtype) {
  auto x = rwkv::Tensor::Empty({256}, dtype, device);
  auto* x_data = x.data_ptr();
  ::DLManagedTensor* dlpack_tensor = toDLPack(x);

  // check data pointer
  RV_CHECK(x_data == dlpack_tensor->dl_tensor.data);
  // check device type
  RV_CHECK(dlpack_tensor->dl_tensor.device.device_type == ConvertToDLDevice(device).device_type);
  // check ndim
  RV_CHECK(dlpack_tensor->dl_tensor.ndim == x.ndim());
  // check size of bits
  RV_CHECK(dlpack_tensor->dl_tensor.dtype.bits == rwkv::elem_size(dtype) * 8);
  // check shape
  RV_CHECK(dlpack_tensor->dl_tensor.shape == x.shape().data());
}

void TestFromDLPack(rwkv::Device device, rwkv::DType dtype) {
  auto tensor1 = rwkv::Tensor::Empty({256}, dtype, device);
  rwkv::Tensor tensor2 = rwkv::fromDLPack(rwkv::toDLPack(tensor1));
  // check whether tensor is equal
  RV_CHECK(tensor1.data_ptr() == tensor2.data_ptr());
  RV_CHECK(tensor1.device() == tensor2.device());
  RV_CHECK(tensor1.dtype() == tensor2.dtype());
  RV_CHECK(tensor1.elem_size() == tensor2.elem_size());
  RV_CHECK(tensor1.shape() == tensor2.shape());
}

TEST(Dlpack, to_dlpack_cpu) {
  TestToDLPack(rwkv::Device::kCPU, rwkv::DType::kFloat32);
  TestToDLPack(rwkv::Device::kCPU, rwkv::DType::kFloat16);
  TestToDLPack(rwkv::Device::kCPU, rwkv::DType::kInt8);
  TestToDLPack(rwkv::Device::kCPU, rwkv::DType::kInt32);
}

TEST(Dlpack, from_dlpack_cpu) {
  TestFromDLPack(rwkv::Device::kCPU, rwkv::DType::kFloat32);
  TestFromDLPack(rwkv::Device::kCPU, rwkv::DType::kFloat16);
  TestFromDLPack(rwkv::Device::kCPU, rwkv::DType::kInt8);
  TestFromDLPack(rwkv::Device::kCPU, rwkv::DType::kInt32);
}

#ifdef FR_ENABLE_CUDA
TEST(Dlpack, to_dlpack_gpu) {
  TestToDLPack(rwkv::Device::kCUDA, rwkv::DType::kFloat32);
  TestToDLPack(rwkv::Device::kCUDA, rwkv::DType::kFloat16);
  TestToDLPack(rwkv::Device::kCUDA, rwkv::DType::kInt8);
  TestToDLPack(rwkv::Device::kCUDA, rwkv::DType::kInt32);
}

TEST(Dlpack, from_dlpack_gpu) {
  TestFromDLPack(rwkv::Device::kCUDA, rwkv::DType::kFloat32);
  TestFromDLPack(rwkv::Device::kCUDA, rwkv::DType::kFloat16);
  TestFromDLPack(rwkv::Device::kCUDA, rwkv::DType::kInt8);
  TestFromDLPack(rwkv::Device::kCUDA, rwkv::DType::kInt32);
}
#endif


