#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>

#include <model.h>
#include <sampler.h>
#include <tokenizer.h>

static const bool kShowSpeed = std::getenv("FR_SHOW_SPEED") != nullptr;

int main(int argc, char **argv) {
  std::cout.setf(std::ios::unitbuf);
  rwkv::Tokenizer tokenizer(argv[1]);
  rwkv::Sampler sampler;
  rwkv::Model model(argv[1], argv[2]);
  std::ifstream ifs(argv[3]);
  std::stringstream buffer;

  buffer << ifs.rdbuf();
  std::string input = buffer.str();
  input.erase(input.find_last_not_of(" \t\n\r\f\v") + 1);
  std::cout << input;
  std::vector<int> input_ids = tokenizer.encode(input);
  input_ids.insert(input_ids.begin(), tokenizer.bos_token_id());
  static const int N_TRIAL = 1;
  for (int t = 0; t < N_TRIAL; t++) {
    std::string result = input;
    auto start = std::chrono::system_clock::now();
    auto output_tensor = Copy(model.Run(input_ids), rwkv::Device::kCPU);
    for (int i = 0; i < 1024; i++) {
      auto output_id = sampler.Sample(output_tensor, 1.f, 1, 0.f);
      if (output_id == tokenizer.eos_token_id()) {
        break;
      }
      std::string output = tokenizer.decode(output_id);
      std::cout << output;
      result += output;
      output_tensor = model.Run(output_id);
    }
    std::cout << std::endl;
    auto end = std::chrono::system_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start);
    if (kShowSpeed) {
      std::cout << "time: " << total_time.count() << "ms" << std::endl;
      std::cout << "num tokens: " << result.size() << std::endl;
      std::cout << "ms per token: " << 1. * total_time.count() / result.size() << std::endl;
      std::cout << "tokens per second: " << 1. * result.size() / total_time.count() * 1000 << std::endl;
    }
    std::ofstream ofs("output_" + std::to_string(t) + ".txt");
    ofs << result;
  }

  return 0;
}
