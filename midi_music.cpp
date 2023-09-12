#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

#include <model.h>
#include <sampler.h>
#include <tokenizer.h>

static const bool kShowSpeed = std::getenv("FR_SHOW_SPEED") != nullptr;

int main(int argc, char **argv) {
  std::cout.setf(std::ios::unitbuf);
  rwkv::MIDITokenizer tokenizer(argv[1]);
  rwkv::Sampler sampler;
  rwkv::Model model(argv[2], argv[3]);

  // empty string
  std::string input;
  std::vector<int> input_ids;
  input_ids.insert(input_ids.begin(), tokenizer.bos_token_id);
  static const int N_TRIAL = 3;
  for (int t = 0; t < N_TRIAL; t++) {
    std::string result = input;
    auto start = std::chrono::system_clock::now();
    auto output_tensor = Copy(model.Run(input_ids), rwkv::Device::kCPU);
    std::map<int, float> occurences;
    for (int i = 0; i < 1024; i++) {
      // translated from ChatRWKV
      for (const auto &[id, occurence] : occurences) {
        output_tensor.data_ptr<float>()[id] -= 0.5 * occurence;
      }
      output_tensor.data_ptr<float>()[0] +=
          (i - 2000) / 500.;                      // not too short, not too long
      output_tensor.data_ptr<float>()[127] -= 1.; // avoid "t125"

      auto output_id = sampler.Sample(output_tensor.data_ptr<float>(),
                                      output_tensor.numel(), 1.f, 1, 1.0f);

      // translated from ChatRWKV
      for (const auto &[id, occurence] : occurences) {
        occurences[id] *= 0.997;
      }
      if (output_id >= 128 || output_id == 127) {
        occurences[output_id] =
            1 + (occurences.find(output_id) != occurences.end()
                     ? occurences[output_id]
                     : 0);
      } else {
        occurences[output_id] =
            0.3 + (occurences.find(output_id) != occurences.end()
                       ? occurences[output_id]
                       : 0);
      }

      if (output_id == tokenizer.eos_token_id) {
        break;
      }
      std::string output = " " + tokenizer.decode(output_id);
      std::cout << output;
      result += output;
      output_tensor = model.Run(output_id);
    }
    auto end = std::chrono::system_clock::now();
    auto total_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if (kShowSpeed) {
      std::cout << "time: " << total_time.count() << "ms" << std::endl;
      std::cout << "num tokens: " << result.size() << std::endl;
      std::cout << "ms per token: " << 1. * total_time.count() / result.size()
                << std::endl;
      std::cout << "tokens per second: "
                << 1. * result.size() / total_time.count() * 1000 << std::endl;
    }
    std::ofstream ofs("midi_" + std::to_string(t) + ".txt");
    // the str_to_midi.py requires <start> & <end> as separator
    result = "<start>" + result + " <end>";
    ofs << result;
    std::cout << std::endl;
    std::cout << std::endl;
  }

  return 0;
}
