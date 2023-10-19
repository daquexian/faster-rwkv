#include "check.h"
#include "kernels/kernels.h"
#include <random>
#include <tensor.h>

namespace rwkv {
namespace test {
Tensor uniform(const Shape &shape, float low, float high, DType dtype,
               Device device);
} // namespace test
} // namespace rwkv
