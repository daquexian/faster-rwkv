#include <tensor.h>

namespace rwkv {
namespace shape {
Shape matmul(const Shape &x, const Shape &y);

Shape broadcast_binary(const Shape &x, const Shape &y);

} // namespace shape
} // namespace rwkv
