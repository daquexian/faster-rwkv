#include "tensor.h"
#include "check.h"
#include <iostream>
#include <kernels/kernels.h>
#include <stdexcept>

#include <kernels/ncnn-meta/kernels.h>
#include <kernels/onnx-meta/kernels.h>

namespace rwkv {

void print_tensor(const Tensor &t, const std::string &name) {
  std::cout << "Tensor " << name << std::endl;
  auto t_cpu = Copy(t, Device::kCPU);
  if (t.dtype() == DType::kFloat32) {
    const float *ptr = t_cpu.data_ptr<float>();
    for (int i = 0; i < std::min<int>(20, t.numel()); i++) {
      std::cout << ptr[i] << ", ";
    }
    std::cout << std::endl;
  } else if (t.dtype() == DType::kFloat16) {
    const float16 *ptr = t_cpu.data_ptr<float16>();
    for (int i = 0; i < std::min<int>(20, t.numel()); i++) {
      std::cout << ptr[i] << ", ";
    }
    std::cout << std::endl;
  }
}

Tensor Copy(const Tensor &x, Device device, bool always_copy) {
  if (x.device() == device && !always_copy) {
    return x;
  }
  Tensor y = Tensor::Empty(x.sizes(), x.dtype(), device);
  // TODO: registry
#ifdef FR_ENABLE_CUDA
  if (device == Device::kCPU && x.device() == Device::kCUDA) {
    cudaMemcpy(y.data_ptr(), x.data_ptr(), x.numel() * x.elem_size(),
               cudaMemcpyDeviceToHost);
    return y;
  }
  if (device == Device::kCUDA && x.device() == Device::kCPU) {
    cudaMemcpy(y.data_ptr(), x.data_ptr(), x.numel() * x.elem_size(),
               cudaMemcpyHostToDevice);
    return y;
  }
#endif

#ifdef FR_ENABLE_ONNX
  if (device == Device::kONNXMeta && x.device() == Device::kCPU) {
    y = onnxmeta::possible_initializer(x);
    return y;
  }
#endif

  if (device == Device::kNCNNMeta && x.device() == Device::kCPU) {
    y = ncnnmeta::MemoryData(x);
    return y;
  }
  if (device == Device::kCPU && x.device() == Device::kCPU) {
    memcpy(y.data_ptr(), x.data_ptr(), x.numel() * x.elem_size());
    return y;
  }

  RV_UNIMPLEMENTED();
}

namespace {
int unique_id() {
  static int _unique_id = 0;
  return _unique_id++;
}
} // namespace

Tensor Tensor::Empty(const Shape &shape, DType dtype, Device device) {
  auto storage = std::make_shared<TensorStorage>(
      num_elements(shape) * ::rwkv::elem_size(dtype), device);
  Tensor tensor;
  tensor._storage = storage;
  tensor._shape = shape;
  tensor._dtype = dtype;
  tensor.name = "tensor_" + std::to_string(unique_id());
  return tensor;
}

Tensor Tensor::FromPtr(void *dptr, const Shape &shape, DType dtype,
                       Device device) {
  auto storage = std::make_shared<TensorStorage>(dptr, device);
  Tensor tensor;
  tensor._storage = storage;
  tensor._shape = shape;
  tensor._dtype = dtype;
  tensor.name = "tensor_" + std::to_string(unique_id());
  return tensor;
}

Tensor Tensor::FromOther(const Tensor &other, const Shape &shape) {
  auto storage = other._storage;
  Tensor tensor;
  tensor._storage = storage;
  tensor._shape = shape;
  tensor._dtype = other._dtype;
  tensor.name = "tensor_" + std::to_string(unique_id());
  return tensor;
}

Tensor Tensor::view(const Shape &shape) { return rwkv::reshape(*this, shape); }

Tensor Tensor::flatten() { return rwkv::flatten(*this); }

Tensor Tensor::unsqueeze(int dim) { return rwkv::unsqueeze(*this, dim); }

Tensor operator+(const Tensor &lhs, const Tensor &rhs) { return add(lhs, rhs); }

Tensor operator-(const Tensor &lhs, const Tensor &rhs) { return sub(lhs, rhs); }

Tensor operator-(float lhs, const Tensor &rhs) { return sub(lhs, rhs); }

Tensor operator*(const Tensor &lhs, const Tensor &rhs) { return mul(lhs, rhs); }

Tensor operator/(const Tensor &lhs, const Tensor &rhs) { return div(lhs, rhs); }

TensorStorage::TensorStorage(size_t nbytes, Device device) {
  _data = allocator(device).Allocate(nbytes);
  _device = device;
  _is_view = false;
}

TensorStorage::TensorStorage(void *external_ptr, Device device) {
  _data = external_ptr;
  _device = device;
  _is_view = true;
}

TensorStorage::~TensorStorage() {
  if (!_is_view) {
    allocator(_device).Deallocate(_data);
  }
}

} // namespace rwkv
