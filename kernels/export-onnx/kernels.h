#include <tensor.h>

namespace rwkv {
namespace onnxmeta {
Tensor add_input(const Shape &shape, DType dtype, const std::string &name);
Tensor possible_initializer(const Tensor &x);
Tensor gather(const Tensor& x, const Tensor& index);

void ExportModel(const std::string &input_path, const std::string &output_path);
} // namespace onnxmeta
} // namespace rwkv
