#include "model.h"

namespace rwkv {
namespace ncnnmeta {
void init(const std::string &bp_path, const std::string &pp_path);
void destroy();
} // namespace ncnnmeta
} // namespace rwkv

int main(int argc, char **argv) {
  std::string output_prefix(argv[1]);
  // TODO: refactor
  rwkv::ncnnmeta::init(output_prefix + ".bin", output_prefix + ".param");

  // NOTE: fp32 here is just a placeholder. The dtype used by ncnn is determined
  // when the model is loaded.
  rwkv::Model model(argv[2], "ncnn-meta fp32");
  model.Run(0);
  rwkv::ncnnmeta::destroy();
}
