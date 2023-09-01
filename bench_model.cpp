#include "model.h"

#include <benchmark/benchmark.h>

static void BM_0p1BV4Model(benchmark::State &state) {
  rwkv::Model model("../rwkv-4-0.1b-fp16.fr", "cuda fp16");
  for (auto _ : state)
    auto output = rwkv::Copy(model.Run(0), rwkv::Device::kCPU);
}
BENCHMARK(BM_0p1BV4Model);

// static void BM_1p5BV4Model(benchmark::State &state) {
//   rwkv::Model model("../rwkv-4-1.5b.fr", "cuda fp16");
//   for (auto _ : state)
//     auto output = rwkv::Copy(model.Run(0), rwkv::Device::kCPU);
// }
// BENCHMARK(BM_1p5BV4Model);

BENCHMARK_MAIN();
