#pragma once

#include "check.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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
  // RV_CHECK(n % 128 == 0);
  if (n % 256 == 0) {
    _element_wise<<<n / 256, 256>>>(func, n);
  } else if (n % 128 == 0) {
    _element_wise<<<n / 128, 128>>>(func, n);
  } else {
    _element_wise<<<1, 256>>>(func, n);
  }
}

template<typename T>
__global__ void _element_wise2(int n, const half *k_mix,
  const half *r_mix,
  const half *xx,
  const half *sx,
  half *kx,
  half *rx,
  int mix_numel) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += blockDim.x * gridDim.x) {
    half k_mix_ = k_mix[idx % mix_numel];
    half r_mix_ = r_mix[idx % mix_numel];
    half xx_ = xx[idx];
    half sx_ = sx[idx];
    kx[idx] = __hadd(__hmul(xx_, k_mix_),
                     __hmul(sx_, __hsub(__float2half(1), k_mix_)));
    rx[idx] = __hadd(__hmul(xx_, r_mix_),
                     __hmul(sx_, __hsub(__float2half(1), r_mix_)));
    // printf("idx: %d, k_mix: %f, r_mix: %f, xx: %f, sx: %f, kx: %f, rx: %f\n",
    //        idx, __half2float(k_mix_), __half2float(r_mix_), __half2float(xx_),
    //        __half2float(sx_), __half2float(kx[idx]), __half2float(rx[idx]));
  }
}

template<typename T>
void element_wise2(int n, const half *k_mix,
  const half *r_mix,
  const half *xx,
  const half *sx,
  half *kx,
  half *rx,
  int mix_numel) {
  // 256 is good enough on most GPUs
  // RV_CHECK(n % 128 == 0);
  if (n % 256 == 0) {
    _element_wise2<T><<<n / 256, 256>>>(n, k_mix,
  r_mix,
  xx,
  sx,
  kx,
  rx,
  mix_numel);
  } else if (n % 128 == 0) {
    _element_wise2<T><<<n / 128, 128>>>(n, k_mix,
  r_mix,
  xx,
  sx,
  kx,
  rx,
  mix_numel);
  } else {
    _element_wise2<T><<<1, n>>>(n, k_mix,
  r_mix,
  xx,
  sx,
  kx,
  rx,
  mix_numel);
  }
}
