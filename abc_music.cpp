#include <fstream>
#include <iostream>
#include <sstream>

#include <model.h>
#include <sampler.h>
#include <tokenizer.h>

int main(int argc, char **argv) {
  std::cout.setf(std::ios::unitbuf);
  rwkv::ABCTokenizer tokenizer;
  rwkv::Sampler sampler;
  rwkv::Model model(argv[1], argv[2]);
  std::ifstream ifs(argv[3]);
  std::stringstream buffer;

  buffer << ifs.rdbuf();
  std::string input = buffer.str();
  input.erase(input.find_last_not_of(" \t\n\r\f\v") + 1);
  std::cout << input;
  std::vector<int> input_ids = tokenizer.encode(input);
  input_ids.insert(input_ids.begin(), tokenizer.bos_token_id);
  static const int N_TRIAL = 1;
  for (int t = 0; t < N_TRIAL; t++) {
    std::ofstream ofs("output_" + std::to_string(t) + ".txt");
    ofs << input;
    auto output_tensor = Copy(model.Run(input_ids), rwkv::Device::kCPU);
    for (int i = 0; i < 1024; i++) {
      auto output_id = sampler.Sample(output_tensor.data_ptr<float>(),
                                      output_tensor.numel(), 1.f, 1, 0.f);
      if (output_id == tokenizer.eos_token_id) {
        break;
      }
      std::string output = tokenizer.decode(output_id);
      std::cout << output;
      ofs << output;
      output_tensor = model.Run(output_id);
    }
  }

  return 0;
}
