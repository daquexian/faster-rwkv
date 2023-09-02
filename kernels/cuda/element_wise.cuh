#pragma once

#include "check.h"
#include <cassert>
#include <cstddef>
#include <cstdint>

template <typename Func> __global__ void _element_wise(Func func, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    func(i);
  }
}

// NOTE: packed data type (e.g. float4) is a overkill for current sizes
// (4096 in 7B model and 768 in 0.1B model),
// and is not faster than the plain float version.
template <typename Func>
void element_wise(Func func, int n) {
  // 256 is good enough on most GPUs
  RV_CHECK(n % 128 == 0);
  if (n % 256 == 0) {
    _element_wise<<<n / 256, 256>>>(func, n);
  } else {
    _element_wise<<<n / 128, 128>>>(func, n);
  }
}
