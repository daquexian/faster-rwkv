#include <benchmark/benchmark.h>
#include <tensor.h>
#include <tests/benchmark/random.h>
#include <tests/benchmark/tensor_cache.h>

namespace rwkv {
namespace test {
#define FR_BENCHMARK(bench_name, dtype, device, param_cnt, ndims)              \
  BENCHMARK(bench_name)                                                        \
      ->Setup([](const benchmark::State &state) {                              \
        int offset = param_cnt;                                                \
        std::vector<int> ndims_vec(ndims);                                     \
        for (auto ndim : ndims_vec) {                                          \
          Shape shape;                                                         \
          for (auto i = 0; i < ndim; i++) {                                    \
            shape.push_back(state.range(offset + i));                          \
          }                                                                    \
          TensorCache::get_instance().register_tensors(                        \
              state.name(), {uniform(shape, -1.0, 1.0, dtype, device)});       \
          offset += ndim;                                                      \
        }                                                                      \
      })                                                                       \
      ->Teardown([](const benchmark::State &state) {                           \
        TensorCache::get_instance().unregister_tensors(state.name());          \
      })
} // namespace test
} // namespace rwkv
