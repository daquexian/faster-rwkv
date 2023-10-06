#pragma once

#include <cassert>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>
#ifdef FR_ENABLE_CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

#ifdef FR_ENABLE_NCNN
#include <mat.h>
#endif
#include <msgpack.hpp>

#include <check.h>
#include <half.hpp>

#include "kernels/allocator.h"

#include <iostream>

namespace rwkv {

class Tensor;
void print_n(const rwkv::Tensor &x, const std::string &name, int skip,
             int cnt = 20);

void print_shape(const rwkv::Tensor &x, const std::string &name);

enum class DType {
  kUndefined,
  kInt8,
  kFloat16,
  kFloat32,
  kInt64,
};
using float16 = half_float::half;
enum class Device {
  kCPU,
  kCUDA,
  kNCNNMeta,
  kONNXMeta,
  kNCNN,
  kONNX,
};
template <typename T> inline const DType dtype_v = DType::kFloat32;
template <> inline const DType dtype_v<float16> = DType::kFloat16;
#ifdef FR_ENABLE_CUDA
template <> inline const DType dtype_v<half> = DType::kFloat16;
#endif
template <> inline const DType dtype_v<int8_t> = DType::kInt8;

using LengthType = int64_t;
using Shape = std::vector<LengthType>;

// operator<< for Shape
std::ostream &operator<<(std::ostream &os, const Shape &shape);

inline std::string dtype_to_string(DType dtype) {
  if (dtype == DType::kFloat32) {
    return "fp32";
  } else if (dtype == DType::kFloat16) {
    return "fp16";
  } else if (dtype == DType::kInt8) {
    return "int8";
  } else {
    RV_UNIMPLEMENTED();
  }
}

inline LengthType num_elements(const std::vector<LengthType> &shape) {
  LengthType ret = 1;
  for (auto &x : shape) {
    ret *= x;
  }
  return ret;
}

inline int32_t elem_size(DType dtype) {
  switch (dtype) {
  case DType::kInt8:
    return 1;
  case DType::kFloat16:
    return 2;
  case DType::kFloat32:
    return 4;
  case DType::kInt64:
    return 8;
  default:
    RV_UNIMPLEMENTED();
  }
}

struct Range {
  int start;
  int interval;
  int end;

  Range(int start, int interval, int end)
      : start(start), interval(interval), end(end) {}

  static Range All;
};

class TensorStorage {
public:
  TensorStorage(size_t nbytes, Device device);
  TensorStorage(void *external_ptr, Device device);
  ~TensorStorage();
  void *data_ptr() const { return _data; }
  Device device() const { return _device; }
  FR_DISALLOW_COPY_AND_MOVE(TensorStorage);

private:
  void *_data;
  size_t _nbytes;
  bool _is_view = false;
  Device _device;
};

// prefer to pass Tensor by reference, but even if we pass by value, it's
// data is shared.
class Tensor {
public:
  template <typename T = void> const T *data_ptr() const {
    RV_CHECK(std::is_same_v<T, void> || dtype_v<T> == _dtype);
    return static_cast<T *>(_storage->data_ptr());
  }
  template <typename T = void> T *data_ptr() {
    RV_CHECK(std::is_same_v<T, void> || dtype_v<T> == _dtype);
    return static_cast<T *>(_storage->data_ptr());
  }
  DType dtype() const { return _dtype; }
  Device device() const { return _storage->device(); }
  const Shape &sizes() const { return _shape; }
  const Shape &shape() const { return _shape; }
  LengthType size(int64_t dim) const { return _shape[dim]; }
  LengthType numel() const { return num_elements(_shape); }
  int32_t elem_size() const { return ::rwkv::elem_size(_dtype); }

  Tensor view(const Shape &shape) const;

  Tensor flatten() const;

  Tensor cat(const Tensor &other, int dim) const;

  Tensor unsqueeze(int dim) const;

  Tensor squeeze(int dim) const;

  Tensor slice(const std::vector<Range> &ranges) const;

  Tensor slice(const std::initializer_list<Range> &ranges) const;

  Tensor repeat(const std::initializer_list<LengthType> &repeats) const;

  Tensor repeat(LengthType repeats) const;

  Tensor transpose(int dim_a, int dim_b) const;

  Tensor reshape(const Shape &shape) const;

  Tensor pad(const std::initializer_list<LengthType> &paddings,
             const std::string &mode) const;

  Tensor flip(const std::initializer_list<LengthType> &dims) const;

  Tensor flip(LengthType dim) const { return flip({dim}); }

  template <typename T>
  static Tensor Arange(T start, T interval, T end, DType dtype, Device device) {
    RV_CHECK(start < end && interval > 0 || start > end && interval < 0);
    RV_CHECK(device == Device::kCPU || device == Device::kCUDA);
    // T *host_data = (T *)malloc(numel * sizeof(T));
    std::vector<T> host_data;
    if (start < end) {
      for (T i = start; i < end; i += interval) {
        host_data.push_back(i);
      }
    } else {
      for (T i = start; i > end; i += interval) {
        host_data.push_back(i);
      }
    }
    LengthType numel = host_data.size();

    auto ret = Tensor::Empty({numel}, dtype, device);
    T *data = ret.template data_ptr<T>();
    if (device == Device::kCPU) {
      for (LengthType i = 0; i < numel; i++) {
        data[i] = start + i * interval;
      }
    } else {
#ifdef FR_ENABLE_CUDA
      std::vector<T> host_data;
      for (LengthType i = 0; i < numel; i++) {
        host_data.push_back(start + i * interval);
      }
      auto err = cudaMemcpy(data, host_data.data(), numel * sizeof(T),
                            cudaMemcpyHostToDevice);
#else
      RV_UNIMPLEMENTED();
#endif
    }
    return ret;
  }

  static Tensor Empty(const Shape &shape, DType dtype, Device device);
  static Tensor FromPtr(void *ptr, const Shape &shape, DType dtype,
                        Device device);
  static Tensor FromMsgPack(const msgpack::object &obj);
  static Tensor FromOther(const Tensor &other, const Shape &shape);

  template <typename T> T FromTensor() const;

  template <typename T> static Tensor ToTensor(const T &x);

  std::string name;
  bool is_constant = false;

private:
  Tensor() = default;
  std::shared_ptr<TensorStorage> _storage;
  Shape _shape;
  DType _dtype;
};

Tensor operator+(const Tensor &lhs, const Tensor &rhs);
Tensor operator-(const Tensor &lhs, const Tensor &rhs);
Tensor operator-(float lhs, const Tensor &rhs);
Tensor operator*(const Tensor &lhs, const Tensor &rhs);
Tensor operator/(const Tensor &lhs, const Tensor &rhs);

Tensor Copy(const Tensor &x, Device device, bool always_copy = false);

void print_tensor(const Tensor &t, const std::string &name);

} // namespace rwkv