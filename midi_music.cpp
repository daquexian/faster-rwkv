#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

#include <model.h>
#include <sampler.h>
#include <tokenizer.h>

static const bool kShowSpeed = std::getenv("FR_SHOW_SPEED") != nullptr;

// ./midi_music <tokenizer> <model> <strategy> <sample>
// Example: ./midi_music midi_tokenizer midi_model "ncnn fp16"
int main(int argc, char **argv) {
  std::cout.setf(std::ios::unitbuf);
  rwkv::MIDITokenizer tokenizer(argv[1]);
  rwkv::Sampler sampler;
  rwkv::Model model(argv[2], argv[3]);

  std::string input =
      "v:5b:3 v:5b:2 t125 t125 t125 t106 pi:43:5 t24 pi:4a:7 t15 pi:4f:7 t17 "
      "pi:56:7 t18 pi:54:7 t125 t49 pi:51:7 t117 pi:4d:7 t125 t125 t111 "
      "pi:37:7 t14 pi:3e:6 t15 pi:43:6 t12 pi:4a:7 t17 pi:48:7 t125 t60 "
      "pi:45:7 t121 pi:41:7 t125 t117 s:46:5 s:52:5 f:46:5 f:52:5 t121 s:45:5 "
      "s:46:0 s:51:5 s:52:0 f:45:5 f:46:0 f:51:5 f:52:0 t121 s:41:5 s:45:0 "
      "s:4d:5 s:51:0 f:41:5 f:45:0 f:4d:5 f:51:0 t102 pi:37:0 pi:3e:0 pi:41:0 "
      "pi:43:0 pi:45:0 pi:48:0 pi:4a:0 pi:4d:0 pi:4f:0 pi:51:0 pi:54:0 pi:56:0 "
      "t19 s:3e:5 s:41:0 s:4a:5 s:4d:0 f:3e:5 f:41:0 f:4a:5 f:4d:0 t121 v:3a:5 "
      "t121 v:39:7 t15 v:3a:0 t106 v:35:8 t10 v:39:0 t111 v:30:8 v:35:0 t125 "
      "t117 v:32:8 t10 v:30:0 t125 t125 t103 v:5b:0 v:5b:0 t9 pi:4a:7";

  std::cout << input;

  std::vector<int> input_ids = tokenizer.encode(input);
  input_ids.insert(input_ids.begin(), tokenizer.bos_token_id);
  static const int N_TRIAL = 1;
  for (int t = 0; t < N_TRIAL; t++) {
    std::string result = "pi:4a:7";
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

      auto output_id = sampler.Sample(output_tensor, 1.f, 8, 0.8f);
                                      // output_tensor, 1.f, 1, 1.0f);

      // translated from ChatRWKV
      for (const auto &[id, occurence] : occurences) {
        occurences[id] *= 0.997;
      }
      if (output_id >= 128 || output_id == 127) {
        occurences[output_id] += 1;
      } else {
        occurences[output_id] += 0.3;
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
    const std::string filename = "midi_" + std::to_string(t) + ".txt";
    std::ofstream ofs(filename);
    // the str_to_midi.py requires <start> & <end> as separator
    result = "<start> " + result + " <end>";
    ofs << result;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "Saved to " << filename << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
  }

  return 0;
}
