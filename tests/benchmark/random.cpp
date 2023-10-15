#include <tests/benchmark/random.h>

namespace rwkv {
namespace test {
Tensor uniform(const Shape &shape, float low, float high, DType dtype,
               Device device) {
  RV_CHECK(dtype == DType::kFloat32 || dtype == DType::kFloat16);
  auto ret = Tensor::Empty(shape, DType::kFloat32, device);
  auto *data = ret.data_ptr<float>();
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(low, high);
  if (device == Device::kCPU) {
    for (LengthType i = 0; i < ret.numel(); i++) {
      data[i] = dis(gen);
    }
  } else if (device == Device::kCUDA) {
    std::vector<float> vec(ret.numel());
    for (LengthType i = 0; i < ret.numel(); i++) {
      vec[i] = dis(gen);
    }
    cudaMemcpy(data, vec.data(), ret.numel() * sizeof(float),
               cudaMemcpyHostToDevice);
  } else {
    RV_UNIMPLEMENTED();
  }
  return cast_dtype(ret, dtype);
}
} // namespace test
} // namespace rwkv