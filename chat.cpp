#include <iostream>
#include <map>
#include <chrono>

#include "model.h"
#include "sampler.h"
#include "tokenizer.h"

const std::string kUserPrefix = "User: ";
const std::string kAssistantPrefix = "Assistant:";
const int kMaxOutputLength = 999;
const int kEndOfSentence = 0;
const std::string kDoubleNewLine = "\n\n";
const int kNewLineId = 11;
const int kChatLenShort = 40;
const int kChatLenLong = 150;
const float kPresencePenalty = 0.4;
const float kFrequencyPenalty = 0.4;
const float kPenaltyDecay = 0.996;

const bool kQAMode = true;
const bool kDebug = std::getenv("FR_DEBUG") != nullptr;
const bool kShowSpeed = std::getenv("FR_SHOW_SPEED") != nullptr;

int main(int argc, char **argv) {
  std::cout.setf(std::ios::unitbuf);

  rwkv::Tokenizer tokenizer(argv[1]);
  rwkv::GreedySampler sampler;
  rwkv::Model model(argv[2], argv[3]);
  std::map<int, float> occurences;
  auto states = model.CreateInitialStates();
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
    if (kDebug) {
      std::cout << "prompt_ids: [";
      for (auto id : prompt_ids) {
        std::cout << id << ", ";
      }
      std::cout << "]" << std::endl;
    }
    auto output = Copy(model.Run(prompt_ids, states), rwkv::Device::kCPU);
    std::string response;
    int num_new_tokens = 0;
    for (; num_new_tokens < kMaxOutputLength; num_new_tokens++) {
      for (auto &[id, occurence] : occurences) {
        output.data_ptr<float>()[id] -=
            kFrequencyPenalty * occurence + kPresencePenalty;
        occurence *= kPenaltyDecay;
      }
      if (kQAMode) {
        output.data_ptr<float>()[kEndOfSentence] = -1e30;
        if (num_new_tokens == 0) {
          output.data_ptr<float>()[kNewLineId] += -1e-30;
        } else if (num_new_tokens <= kChatLenShort) {
          output.data_ptr<float>()[kNewLineId] += (num_new_tokens - kChatLenShort) / 10.;
        } else if (num_new_tokens <= kChatLenLong) {
          output.data_ptr<float>()[kNewLineId] += 0;
        } else {
          output.data_ptr<float>()[kNewLineId] +=
              std::min(3.0f, (num_new_tokens - kChatLenLong) * 0.25f);
        }
      }
      auto output_id = sampler.Sample(output.data_ptr<float>(), output.numel());
      if (kDebug) {
        std::cout << "output_id: " << output_id << std::endl;
        static const int kDebugOutputLength = 10;
        std::cout << "output: [";
        for (int i = 0; i < kDebugOutputLength; i++) {
          std::cout << output.data_ptr<float>()[i] << ", ";
        }
      }
      occurences[output_id]++;
      if (output_id == kEndOfSentence && !kQAMode) {
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
      output = Copy(model.Run(output_id, states), rwkv::Device::kCPU);
    }
    auto model_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now() - tmp);
    auto end = std::chrono::system_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start);
    if (kShowSpeed) {
      std::cout << "time: " << total_time.count() << "ms" << std::endl;
      std::cout << "num tokens: " << prompt_ids.size() + num_new_tokens << std::endl;
      std::cout << "ms per token: " << 1. * total_time.count() / (prompt_ids.size() + num_new_tokens) << std::endl;
      std::cout << "tokens per second: " << 1. * (prompt_ids.size() + num_new_tokens) / total_time.count() * 1000 << std::endl;
    }

    // std::cout << std::endl;
    // model.Run(tokenizer.encode("\n"), states);
  }
  return 0;
}
