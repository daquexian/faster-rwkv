#include <tensor.h>

namespace rwkv {
namespace ncnnmeta {
Tensor add_input(const Shape &shape, const std::string &name);
Tensor MemoryData(const Tensor &x);

void ExportModel(const std::string &input_path, DType weight_dtype,
                 const std::string &output_prefix);
} // namespace ncnnmeta
} // namespace rwkv
