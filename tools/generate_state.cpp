#include <sstream>
#include <fstream>
#include <iostream>

#include <msgpack.hpp>

#include <model.h>
#include <tokenizer.h>

using namespace rwkv;

std::string dtype_to_string_in_msgpack(DType dtype) {
  if (dtype == DType::kFloat32) {
    return "torch.float32";
  } else if (dtype == DType::kFloat16) {
    return "torch.float16";
  } else if (dtype == DType::kInt8) {
    return "torch.int8";
  } else {
    RV_UNIMPLEMENTED();
  }
}

int main(int argc, char** argv) {
  Tokenizer tokenizer(argv[1]);
  Model model(argv[2], argv[3]);
  // It is recommended to preprocess the init prompt by tools/preprocess_init_prompt.py
  std::ifstream ifs(argv[4]);
  std::stringstream buffer;
  buffer << ifs.rdbuf();
  std::string input = buffer.str();

  std::vector<int> input_ids = tokenizer.encode(input);
  model.Run(input_ids);

  model.SaveStateFile(argv[5]);

  return 0;
}
