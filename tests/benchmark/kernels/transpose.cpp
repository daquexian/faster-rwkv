#include "tensor.h"
#include <benchmark/benchmark.h>
#include <iostream>
#include <tests/benchmark/common.h>
#include <tests/benchmark/tensor_cache.h>

namespace rwkv {
namespace test {
#if FR_ENABLE_CUDA

static void bench_transpose(benchmark::State &state) {
  for (auto _ : state) {
    auto x = TensorCache::get_instance().get_tensors(state.name())[0];
    auto y = x.transpose(state.range(0), state.range(1));
  }
}

FR_BENCHMARK(bench_transpose, DType::kFloat32, Device::kCUDA, 2, {2})
    ->Args({0, 1, 20, 30})
    ->Args({0, 1, 400, 500});

FR_BENCHMARK(bench_transpose, DType::kFloat32, Device::kCUDA, 2, {3})
    ->Args({0, 2, 20, 30, 100})
    ->Args({-1, -2, 20, 30, 100})
    ->Args({0, 1, 400, 500, 600})
    ->Args({-1, -2, 400, 500, 600});

#endif

} // namespace test
} // namespace rwkv

BENCHMARK_MAIN();