#include <tensor.h>

namespace rwkv {
namespace ncnnmeta {
Tensor add_input(const Shape &shape, const std::string &name);
Tensor Embedding(const Tensor &weight, const Tensor &id);
Tensor MemoryData(const Tensor &x);

void ExportModel(const std::string &input_path, DType weight_dtype,
                 const std::string &output_prefix);
void disable_int4(bool use_fp16);
} // namespace ncnnmeta
} // namespace rwkv
