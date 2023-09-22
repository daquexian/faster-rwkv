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

  States states = model.states();
  std::vector<std::vector<std::unordered_map<std::string, msgpack::object>>> mp_states;

  msgpack::zone z;
  for (const auto& state : states) {
    std::vector<std::unordered_map<std::string, msgpack::object>> mp_state;
    for (const auto& s : state) {
      std::unordered_map<std::string, msgpack::object> mp_s;
      std::vector<char> data_vec;
      data_vec.resize(s.numel() * s.elem_size());
      memcpy(data_vec.data(), s.data_ptr(), s.numel() * s.elem_size());
      mp_s["dtype"] = msgpack::object(dtype_to_string_in_msgpack(s.dtype()), z);
      mp_s["data"] = msgpack::object(data_vec, z);
      mp_s["shape"] = msgpack::object(s.shape(), z);
      mp_state.push_back(mp_s);
    }
    mp_states.push_back(mp_state);
  }

  std::ofstream ofs(argv[5]);
  msgpack::pack(ofs, mp_states);

  return 0;
}
