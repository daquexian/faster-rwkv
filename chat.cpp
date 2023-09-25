#include <chrono>
#include <iostream>
#include <map>

#include <model.h>
#include <sampler.h>
#include <tokenizer.h>

static const std::string kUserPrefix = "Question: ";
// no space after kAssistantPrefix
static const std::string kAssistantPrefix = "Answer:";
// for debug propose
static const int kMaxOutputLength =
    std::getenv("FR_MAX_OUTPUT_LEN") != nullptr
        ? std::stoi(std::getenv("FR_MAX_OUTPUT_LEN"))
        : 999;
static const int kEndOfSentence = 0;
static const std::string kDoubleNewLine = "\n\n";
static const float kTopP = 0.0;
static const float kPresencePenalty = 0.8;
static const float kFrequencyPenalty = 0.8;
static const float kPenaltyDecay = 0.996;
static const bool kGlobalPenalty = std::getenv("FR_GLOBAL_PENALTY") != nullptr;

static const bool kShowSpeed = std::getenv("FR_SHOW_SPEED") != nullptr;

int main(int argc, char **argv) {
  std::cout.setf(std::ios::unitbuf);

  rwkv::Tokenizer tokenizer(argv[1]);
  rwkv::Sampler sampler;
  rwkv::Model model(argv[2], argv[3]);
  if (argc == 5) {
    model.LoadStateFile(argv[4]);
  }
  std::map<int, float> occurences;
  while (true) {
    std::cout << kUserPrefix;
    std::string input;
    std::getline(std::cin, input);
    std::cout << kAssistantPrefix;
    const std::string prompt =
        kUserPrefix + input + kDoubleNewLine + kAssistantPrefix;
    auto start = std::chrono::system_clock::now();
    auto tmp = start;
    auto prompt_ids = tokenizer.encode(prompt);
    auto encode_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now() - tmp);
    tmp = std::chrono::system_clock::now();
    auto output = Copy(model.Run(prompt_ids), rwkv::Device::kCPU);
    std::string response;
    int num_new_tokens = 0;
    for (; num_new_tokens < kMaxOutputLength; num_new_tokens++) {
      for (auto &[id, occurence] : occurences) {
        output.data_ptr<float>()[id] -=
            kFrequencyPenalty * occurence + kPresencePenalty;
        occurence *= kPenaltyDecay;
      }
      auto output_id =
          sampler.Sample(output, /*temperature=*/1.f, /*top_k=*/1, kTopP);
      occurences[output_id]++;
      if (output_id == kEndOfSentence) {
        break;
      }
      auto output_str = tokenizer.decode(output_id);
      std::cout << output_str;
      response += output_str;
      if (response.size() >= 2 &&
          response.substr(response.size() - 2) == "\n\n") {
        break;
      }
      // if (response.size() >= kUserPrefix.size() &&
      // response.substr(response.size() - 7) == "\nUser: ") {
      //   break;
      // }
      output = Copy(model.Run(output_id), rwkv::Device::kCPU);
    }
    auto model_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now() - tmp);
    auto end = std::chrono::system_clock::now();
    auto total_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if (kShowSpeed) {
      std::cout << "-- time: " << total_time.count() << "ms" << std::endl;
      std::cout << "-- num tokens: " << prompt_ids.size() + num_new_tokens
                << std::endl;
      std::cout << "-- ms per token: "
                << 1. * total_time.count() /
                       (prompt_ids.size() + num_new_tokens)
                << std::endl;
      std::cout << "-- tokens per second: "
                << 1. * (prompt_ids.size() + num_new_tokens) /
                       total_time.count() * 1000
                << std::endl;
      std::cout << std::endl;
    }
    if (!kGlobalPenalty) {
      occurences.clear();
    }

    // std::cout << std::endl;
    // model.Run(tokenizer.encode("\n"), states);
  }
  return 0;
}
