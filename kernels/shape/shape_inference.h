#include <tensor.h>

namespace rwkv {
namespace shape {
Shape matmul(const Shape &x, const Shape &y);

Shape broadcast_binary(const Shape &x, const Shape &y);

Shape concat(const std::vector<Shape> &xs, int axis);

Shape slice(const Shape &x, const std::vector<int> &starts,
    const std::vector<int> &ends, const std::vector<int> &axes);

} // namespace shape
} // namespace rwkv
