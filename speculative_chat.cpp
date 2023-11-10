#include <chrono>
#include <iostream>
#include <map>

#include <model.h>
#include <sampler.h>
#include <tokenizer.h>
#include <tuple>

static const std::string kUserPrefix = "User: ";
// no space after "Assistant:"
static const std::string kAssistantPrefix = "Assistant:";
static const int kMaxOutputLength =
    std::getenv("FR_MAX_OUTPUT_LEN") != nullptr
        ? std::stoi(std::getenv("FR_MAX_OUTPUT_LEN"))
        : 999;
static const int kEndOfSentence = 0;
static const std::string kDoubleNewLine = "\n\n";
static const int kNewLineId = 11;
static const int kChatLenShort = 40;
static const int kChatLenLong = 150;
static const float kTopP = 0.0;
static const float kPresencePenalty = 0.8;
static const float kFrequencyPenalty = 0.8;
static const float kPenaltyDecay = 0.996;
static const bool kGlobalPenalty = std::getenv("FR_GLOBAL_PENALTY") != nullptr;

static const bool kQAMode = true;
static const bool kShowSpeed = std::getenv("FR_SHOW_SPEED") != nullptr;

int main(int argc, char **argv) {
  std::cout.setf(std::ios::unitbuf);

  rwkv::Tokenizer tokenizer(argv[1]);
  rwkv::Sampler sampler;
  rwkv::Model main_model(argv[2], argv[4]);
  rwkv::Model assistant_model(argv[3], argv[4]);

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

    int num_new_tokens = 0;
    auto sample_func = [&sampler, &occurences,
                        &num_new_tokens](rwkv::Tensor &tensor) {
      return sampler.Sample(tensor, /*temperature=*/1.f, /*top_k=*/1, kTopP);
    };
    auto decode_func = [&tokenizer](int id) { return tokenizer.decode(id); };

    // prefill the models
    auto output = main_model.Run(prompt_ids);
    assistant_model.Run(prompt_ids);

    std::string response;
    int id = sample_func(output);
    num_new_tokens++;
    auto first_str = tokenizer.decode(id);
    std::cout << first_str;
    response += first_str;
    int accept_length;
    std::vector<int> outputs;
    std::tie(accept_length, outputs) =
        main_model.AssistedRun(id, assistant_model, 5, sample_func, decode_func,
                               {kUserPrefix, kDoubleNewLine}, response);
    for (; num_new_tokens < kMaxOutputLength;) {
      bool should_break = false;
      for (auto output_id : outputs) {
        if (output_id == kEndOfSentence) {
          should_break = true;
          std::cout << std::endl;
          break;
        }
        auto output_str = tokenizer.decode(output_id);
        std::cout << output_str;
        response += output_str;
      }
      if (should_break)
        break;
      id = outputs.back();
      if (response.find(kUserPrefix) != std::string::npos) {
        break; // anti-prompt
      }
      if (response.size() >= 2 &&
          response.substr(response.size() - 2) == "\n\n") {
        // main_model.Run(id);
        // assistant_model.Run(id);
        break;
      }
      // std::cout << "[output] size = " << outputs.size() << "  tokens = ";
      // for(auto outpu)
      std::tie(accept_length, outputs) = main_model.AssistedRun(
          id, assistant_model, 5, sample_func, decode_func,
          {kUserPrefix, kDoubleNewLine}, response);
      num_new_tokens += outputs.size();
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
