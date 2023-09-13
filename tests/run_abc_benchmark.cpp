#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

#include <msgpack.hpp>

#include <model.h>
#include <sampler.h>
#include <tokenizer.h>
#include <kernels/kernels.h>

static const bool kShowSpeed = std::getenv("FR_SHOW_SPEED") != nullptr;

int main(int argc, char **argv) {
  rwkv::ABCTokenizer tokenizer;
  rwkv::Sampler sampler;
  rwkv::Model model(argv[1], argv[2]);

  const std::vector<std::string> eval_set =
      [&]() {
        std::ifstream infile;
        infile.open(argv[3], std::ios::binary | std::ios::in);
        infile.seekg(0, std::ios::end);
        int64_t length = infile.tellg();
        infile.seekg(0, std::ios::beg);
        char *data = new char[length];
        infile.read(data, length);
        infile.close();
        auto unpacker = msgpack::unpack(data, length);
        auto obj = unpacker.get();
        auto ret = obj.as<std::vector<std::string>>();
        delete[] data;
        return ret;
      }();

  std::vector<std::vector<float>> probs_vec;
  const int limit = std::getenv("LIMIT") ? std::stoi(std::getenv("LIMIT")) : eval_set.size();
  for (int j = 0; j < limit; j++) {
    const auto& input = eval_set[j];
    std::cout << j << "/" << limit << std::endl;
    std::vector<float> probs;
    std::vector<int> input_ids = tokenizer.encode(input);
    input_ids.insert(input_ids.begin(), tokenizer.bos_token_id);
    input_ids.push_back(tokenizer.eos_token_id);
    model.ResetStates();
    for (int i = 0; i < input_ids.size() - 1; i++) {
      auto output_tensor = Copy(model.Run(input_ids[i]), rwkv::Device::kCPU);

      output_tensor = rwkv::softmax(output_tensor, 1);

      probs.push_back(output_tensor.data_ptr<float>()[input_ids[i + 1]]);
    }
    probs_vec.push_back(probs);
  }

  std::ofstream ofs("abc_probs");
  msgpack::pack(ofs, probs_vec);

  return 0;
}
