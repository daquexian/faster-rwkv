#include <tensor.h>

namespace rwkv {
namespace ncnnmeta {
Tensor add_input(const Shape &shape, const std::string &name);
Tensor MemoryData(const Tensor &x);
}
}
