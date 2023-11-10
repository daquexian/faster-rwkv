#include "tensor.h"
#include <numeric>
#include <vector>

namespace rwkv {
namespace utils {
LengthType indices_to_offset(const Shape &shape,
                             const std::vector<LengthType> &indices) {
  LengthType offset = 0;
  LengthType base = 1;
  for (int i = shape.size() - 1; i >= 0; i--) {
    offset += indices[i] * base;
    base *= shape[i];
  }
  return offset;
}

void offset_to_indices(LengthType offset, const Shape &shape,
                       std::vector<LengthType> &indices) {
  LengthType total = std::accumulate(shape.begin(), shape.end(), 1,
                                     std::multiplies<LengthType>());
  for (LengthType i = 0; i < shape.size(); i++) {
    total /= shape[i];
    indices[i] = offset / total;
    offset %= total;
  }
}
} // namespace utils
} // namespace rwkv
