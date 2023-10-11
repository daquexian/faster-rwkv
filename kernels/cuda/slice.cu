#include "check.h"
#include "element_wise.cuh"
#include "util.cuh"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <kernels/macro.h>
#include <kernels/registry.h>
#include <numeric>
#include <stdio.h>
#include <tensor.h>
#include <vector>

namespace rwkv {
namespace cuda {

template <typename T, int ndim>
__global__ void _slice(int output_total, int offset, const T *input, T *output,
                       const LengthType *input_shape,
                       const LengthType *output_shape, const int *starts,
                       const int *intervals) {
  LengthType indices[ndim];
  for (int i = blockIdx.x * blockDim.x + threadIdx.x + offset; i < output_total;
       i += blockDim.x * gridDim.x) {
    ::cuda::offset_to_indices(i, output_shape, indices, output_total, ndim);
    for (int j = 0; j < ndim; j++) {
      int start = starts[j];
      int interval = intervals[j];
      indices[j] = start + interval * indices[j];
    }

    LengthType input_offset =
        ::cuda::indices_to_offset(input_shape, indices, ndim);
    output[i] = input[input_offset];
  }
}

// NOTE: packed data type (e.g. float4) is a overkill for current sizes
// (4096 in 7B model and 768 in 0.1B model),
// and is not faster than the plain float version.
template <int ndim>
Tensor slice_internal(const Tensor &x, const std::vector<Range> &ranges) {
  RV_CHECK(x.sizes().size() == ranges.size());
  auto input_shape = x.shape();
  std::vector<int> no_negative_starts;
  std::vector<int> no_negative_intervals;

  auto slice_deduce_shape = [&input_shape, &ranges, &no_negative_starts,
                             &no_negative_intervals]() {
    Shape output_shape;
    for (int i = 0; i < input_shape.size(); i++) {
      LengthType input_dim = input_shape[i];
      auto [start, interval, end] = ranges[i];
      if (start == 0 && interval == 0 && end == 0) {
        end = input_dim;
        interval = 1;
      }
      if (start < 0)
        start += input_dim;
      if (end < 0)
        end += input_dim;
      RV_CHECK(interval > 0);
      RV_CHECK(start <= end && end <= input_dim);
      output_shape.push_back((end - start) / interval);
      no_negative_starts.push_back(start);
      no_negative_intervals.push_back(interval);
    }
    return output_shape;
  };

  auto output_shape = slice_deduce_shape();
  Tensor output = Tensor::Empty(output_shape, x.dtype(), x.device());
  int total_elems = output.numel();

  int *start_ptr;
  int *interval_ptr;
  LengthType *input_shape_gpu;
  LengthType *output_shape_gpu;
  cudaMalloc(&start_ptr, ndim * sizeof(int));
  cudaMalloc(&interval_ptr, ndim * sizeof(int));
  cudaMalloc(&input_shape_gpu, ndim * sizeof(LengthType));
  cudaMalloc(&output_shape_gpu, ndim * sizeof(LengthType));
  cudaMemcpy(start_ptr, no_negative_starts.data(), ndim * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(interval_ptr, no_negative_intervals.data(), ndim * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(input_shape_gpu, input_shape.data(), ndim * sizeof(LengthType),
             cudaMemcpyHostToDevice);
  cudaMemcpy(output_shape_gpu, output_shape.data(), ndim * sizeof(LengthType),
             cudaMemcpyHostToDevice);

#define LAUNCH_KERNEL(type)                                                    \
  FR_LAUNCH_CUDA_KERNEL_BASE_256(_slice, type, ndim, total_elems,              \
                                 x.data_ptr<type>(), output.data_ptr<type>(),  \
                                 input_shape_gpu, output_shape_gpu, start_ptr, \
                                 interval_ptr);

  if (output.dtype() == DType::kFloat32) {
    LAUNCH_KERNEL(float)
  } else if (output.dtype() == DType::kFloat16) {
    LAUNCH_KERNEL(half)
  } else if (output.dtype() == DType::kInt8) {
    LAUNCH_KERNEL(int8_t);
  } else {
    RV_UNIMPLEMENTED();
  }

  cudaFree(start_ptr);
  cudaFree(interval_ptr);
  cudaFree(input_shape_gpu);
  cudaFree(output_shape_gpu);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("slice CUDA error: %s\n", cudaGetErrorString(error));
  }

  return output;

#undef LAUNCH_KERNEL
}

Tensor slice(const Tensor &x, const std::vector<Range> &ranges) {
  int ndim = x.sizes().size();
  if (ndim == 1) {
    return slice_internal<1>(x, ranges);
  } else if (ndim == 2) {
    return slice_internal<2>(x, ranges);
  } else if (ndim == 3) {
    return slice_internal<3>(x, ranges);
  } else if (ndim == 4) {
    return slice_internal<4>(x, ranges);
  } else {
    RV_UNIMPLEMENTED();
  }
}

KernelRegister slice_reg_cuda("slice", Device::kCUDA, slice);

} // namespace cuda
} // namespace rwkv
