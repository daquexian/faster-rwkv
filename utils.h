#include "tensor.h"
#include <iostream>
#include <vector>

#define LOG_INFO(msg) std::cout << "[INFO] " << msg << std::endl;

namespace rwkv {
namespace utils {
LengthType indices_to_offset(const Shape &shape,
                             const std::vector<LengthType> &indices);

void offset_to_indices(LengthType offset, const Shape &shape,
                       std::vector<LengthType> &indices);
} // namespace utils
} // namespace rwkv