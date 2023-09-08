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

#include <check.h>
#include <half.hpp>

namespace rwkv {
enum class DType {
  kInt8,
  kFloat16,
  kFloat32,
};
using float16 = half_float::half;
enum class Device {
  kCPU,
  kCUDA,
  kNCNNMeta,
  kONNXMeta,
  kNCNN,
};
template <typename T> inline const DType dtype_v = DType::kFloat32;
template <> inline const DType dtype_v<float16> = DType::kFloat16;
#ifdef FR_ENABLE_CUDA
template <> inline const DType dtype_v<half> = DType::kFloat16;
#endif
template <> inline const DType dtype_v<int8_t> = DType::kInt8;

using LengthType = int64_t;
using Shape = std::vector<LengthType>;

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
  default:
    RV_UNIMPLEMENTED();
  }
}

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

  Tensor view(const Shape &shape);

  Tensor flatten();

  Tensor unsqueeze(int dim);

  static Tensor Empty(const Shape &shape, DType dtype, Device device);
  static Tensor FromPtr(void *ptr, const Shape &shape, DType dtype,
                        Device device);
  static Tensor FromOther(const Tensor& other, const Shape &shape);

  template<typename T>
  T FromTensor() const;

  template<typename T>
  static Tensor ToTensor(const T& x);

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
