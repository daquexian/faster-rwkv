#include "tensor.h"
#include <vector>

namespace rwkv {
namespace utils {
LengthType indices_to_offset(const Shape &shape,
                             const std::vector<LengthType> &indices) {
  LengthType offset = 0;
  LengthType base = 1;
  for (LengthType i = 0; i < shape.size(); i++) {
    auto reversed_index = shape.size() - i - 1;
    offset += indices[reversed_index] * base;
    base *= shape[reversed_index];
  }
  return offset;
}

void offset_to_indices(LengthType offset, const Shape &shape,
                       std::vector<LengthType> &indices) {
  for (LengthType i = 0; i < shape.size(); i++) {
    auto reversed_index = shape.size() - i - 1;
    indices[i] = offset % shape[reversed_index];
    offset /= shape[reversed_index];
  }
}
} // namespace utils
} // namespace rwkv