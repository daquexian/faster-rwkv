#include "check.h"
#include "utils.h"
#include <functional>
#include <initializer_list>
#include <kernels/kernels.h>
#include <kernels/registry.h>
#include <numeric>
#include <tensor.h>
#include <vector>

namespace rwkv {
namespace def {

template <typename T>
void vgather_internal_cpu(const std::vector<Tensor> &x,
                          const std::vector<int> &idx, Tensor &res, int count,
                          int single_elems, int elem_size) {
  auto *res_ptr = res.data_ptr<T>();
  for (int i = 0; i < count; i++) {
    RV_CHECK(idx[i] < x.size());
    auto *x_ptr = x[idx[i]].data_ptr<T>();
    memcpy(static_cast<void *>(res_ptr), static_cast<const void *>(x_ptr),
           single_elems * elem_size);
    res_ptr += single_elems;
  }
}

#if FR_ENABLE_CUDA

template <typename T>
void vgather_internal_cuda(const std::vector<Tensor> &x,
                           const std::vector<int> &idx, Tensor &res, int count,
                           int single_elems, int elem_size) {
  auto *res_ptr = res.data_ptr<T>();
  for (int i = 0; i < count; i++) {
    RV_CHECK(idx[i] < x.size());
    auto *x_ptr = x[idx[i]].data_ptr<T>();
    cudaMemcpy(static_cast<void *>(res_ptr), static_cast<const void *>(x_ptr),
               single_elems * elem_size, cudaMemcpyDeviceToDevice);
    res_ptr += single_elems;
  }
}

#endif

Tensor vgather(const std::vector<Tensor> &x, const std::vector<int> &idx) {
  auto single_shape = x[0].shape();
  auto count = idx.size();
  Shape output_shape;
  output_shape.push_back(count);
  output_shape.insert(output_shape.end(), single_shape.begin(),
                      single_shape.end());
  Tensor res = Tensor::Empty(output_shape, x[0].dtype(), x[0].device());

#define LAUNCH_VGATHER_KERNEL(func, type)                                      \
  func<type>(x, idx, res, count, x[0].numel(), x[0].elem_size());

  if (x[0].device() == Device::kCPU) {
    if (x[0].dtype() == DType::kFloat32) {
      LAUNCH_VGATHER_KERNEL(vgather_internal_cpu, float)
    } else if (x[0].dtype() == DType::kFloat16) {
      LAUNCH_VGATHER_KERNEL(vgather_internal_cpu, float16)
    } else if (x[0].dtype() == DType::kInt8) {
      LAUNCH_VGATHER_KERNEL(vgather_internal_cpu, int8_t)
    } else {
      RV_CHECK(false);
    }
  }
#if FR_ENABLE_CUDA
  else if (x[0].device() == Device::kCUDA) {
    if (x[0].dtype() == DType::kFloat32) {
      LAUNCH_VGATHER_KERNEL(vgather_internal_cuda, float)
    } else if (x[0].dtype() == DType::kFloat16) {
      LAUNCH_VGATHER_KERNEL(vgather_internal_cuda, float16)
    } else if (x[0].dtype() == DType::kInt8) {
      LAUNCH_VGATHER_KERNEL(vgather_internal_cuda, int8_t)
    } else {
      RV_UNIMPLEMENTED();
    }
  }
#endif
  else {
    RV_UNIMPLEMENTED();
  }

#undef LAUNCH_VGATHER_KERNEL

  return res;
}

KernelRegister vgather_reg_cpu("vgather", Device::kCPU, vgather);
KernelRegister vgather_reg_cuda("vgather", Device::kCUDA, vgather);

} // namespace def
} // namespace rwkv