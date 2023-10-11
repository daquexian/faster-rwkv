#include "tensor.h"
#include "check.h"
#include <initializer_list>
#include <iostream>

#include <check.h>
#include <kernels/export-ncnn/kernels.h>
#include <kernels/kernels.h>
#include <stdexcept>
#ifdef FR_ENABLE_ONNX
#include <kernels/export-onnx/kernels.h>
#endif

namespace rwkv {
Range Range::All = Range(0, 0, 0);

std::optional<Device>& default_dispatch_device() {
  static std::optional<Device> _default_dispatch_device = std::nullopt;
  return _default_dispatch_device;
}

// operator<< for Shape
std::ostream &operator<<(std::ostream &os, const Shape &shape) {
  os << "(";
  for (int i = 0; i < shape.size(); i++) {
    os << shape[i];
    if (i != shape.size() - 1) {
      os << ", ";
    }
  }
  os << ")";
  return os;
}

std::ostream &operator<<(std::ostream &os, DType shape) {
  os << dtype_to_string(shape);
  return os;
}

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

  RV_UNIMPLEMENTED() << "Copy from device " << static_cast<int>(x.device())
                     << " to device " << static_cast<int>(device)
                     << " is not supported yet.";
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

Tensor Tensor::FromMsgPack(const msgpack::object &obj) {
  auto from_mp_dtype = [](const std::string &mp_dtype) -> DType {
    if (mp_dtype == "torch.int8") {
      return DType::kInt8;
    } else if (mp_dtype == "torch.float16") {
      return DType::kFloat16;
    } else if (mp_dtype == "torch.float32") {
      return DType::kFloat32;
    } else {
      RV_UNIMPLEMENTED();
    }
  };

  auto mp_tensor_map =
      obj.as<std::unordered_map<std::string, msgpack::object>>();
  // NOTE: `mp_tensor_data` will be destroyed after this function returns
  auto mp_tensor_data = mp_tensor_map["data"].as<std::vector<char>>();
  auto mp_tensor_shape = mp_tensor_map["shape"].as<std::vector<int64_t>>();
  auto mp_tensor_dtype = mp_tensor_map["dtype"].as<std::string>();
  auto fr_cpu_tensor =
      Tensor::FromPtr(mp_tensor_data.data(), Shape(mp_tensor_shape),
                      from_mp_dtype(mp_tensor_dtype), Device::kCPU);
  auto ret = Copy(fr_cpu_tensor, Device::kCPU, true);
  return ret;
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

Tensor Tensor::view(const Shape &shape) const {
  return rwkv::reshape(*this, shape);
}

Tensor Tensor::flatten() const { return rwkv::flatten(*this); }

Tensor Tensor::cat(const Tensor &other, int dim) const {
  return rwkv::cat(*this, other, dim);
}

Tensor Tensor::unsqueeze(int dim) const { return rwkv::unsqueeze(*this, dim); }

Tensor Tensor::squeeze(int dim) const { return rwkv::squeeze(*this, dim); }

Tensor Tensor::slice(const std::vector<Range> &ranges) const {
  return rwkv::slice(*this, ranges);
}

Tensor Tensor::slice(const std::initializer_list<Range> &ranges) const {
  return rwkv::slice(*this, std::vector<Range>(ranges));
}

Tensor Tensor::repeat(const std::initializer_list<LengthType> &repeats) const {
  return ::rwkv::repeat(*this, repeats);
}

Tensor Tensor::repeat(LengthType repeats) const {
  return this->repeat({repeats});
}

Tensor Tensor::pad(const std::initializer_list<LengthType> &paddings,
                   const std::string &mode) const {
  return ::rwkv::pad(*this, paddings, mode);
}

Tensor Tensor::reshape(const Shape &shape) const {
  return ::rwkv::reshape(*this, shape);
}

Tensor Tensor::transpose(int dim_a, int dim_b) const {
  return ::rwkv::transpose(*this, dim_a, dim_b);
}

Tensor Tensor::flip(const std::initializer_list<LengthType> &dims) const {
  return ::rwkv::flip(*this, dims);
}

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

void print_n(const rwkv::Tensor &x, const std::string &name, int skip,
             int cnt) {
  auto x_cpu = rwkv::Copy(x, rwkv::Device::kCPU);
  auto max_elems = x.numel();
  if (cnt > max_elems) {
    cnt = max_elems;
    skip = 0;
  }
  std::cout << ">>>>>>>> " << name << ": ";
  for (int i = 0; i < cnt; i++) {
    if (x.dtype() == rwkv::DType::kFloat32) {
      std::cout << std::fixed << std::setprecision(6)
                << x_cpu.data_ptr<float>()[skip + i] << ", ";
    } else if (x.dtype() == rwkv::DType::kFloat16) {
      std::cout << std::fixed << std::setprecision(6)
                << static_cast<float>(x_cpu.data_ptr<float16>()[skip + i])
                << ", ";
    }
  }
  std::cout << std::endl;
}

void print_shape(const rwkv::Tensor &x, const std::string &name) {
  std::cout << ">>>>>>>> " << name << ": " << x.sizes().size() << " dims  "
            << x.size(0) << ", " << x.size(1) << ", " << x.size(2) << std::endl;
}

} // namespace rwkv
