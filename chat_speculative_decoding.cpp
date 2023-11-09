#include "kernels/kernels.h"
#include "tensor.h"
#include <chrono>
#include <functional>
#include <iostream>
#include <map>
#include <random>

#include <model.h>
#include <sampler.h>
#include <string>
#include <tokenizer.h>
#include <tuple>
#include <vector>

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

static const std::string main_model_path =
    "/home/rinne/code/llm/models/rwkv4/"
    "RWKV-4-World-1.5B-v1-fixed-20230612-ctx4096-cuda-fp16.fr";
static const std::string assistant_model_path =
    "/home/rinne/code/llm/models/rwkv4/"
    "RWKV-4-World-CHNtuned-0.4B-v1-fp16-cuda.fr";

std::tuple<rwkv::Tensor, int> speculative_decode(
    rwkv::Model &main_model, rwkv::Model &assistant_model,
    const std::string &input_text, const std::vector<int> &input_ids,
    rwkv::Tokenizer &tokenizer,
    std::function<int(const std::vector<rwkv::Tensor> &)> &sample_func,
    int speculative_length, std::vector<rwkv::States> &assistant_states_record,
    std::vector<rwkv::States> &main_states_record) {
  rwkv::States assistent_states;
  rwkv::States main_states;
  std::vector<int> assistant_model_input_ids(input_ids);
  std::vector<int> main_model_input_ids(input_ids);
  if (assistant_model_input_ids.size() > 1) {
    assistant_model.Run(assistant_model_input_ids.back(), &assistent_states);
    main_model.Run(assistant_model_input_ids.back(), &main_states);
  }

  std::vector<rwkv::Tensor> assistant_probs;
  std::vector<int> speculative_ids;
  for (int i = 0; i < speculative_length; i++) {
    auto assistent_logits =
        assistant_model.Run(assistant_model_input_ids, false, assistent_states);
    assistant_probs.push_back(rwkv::softmax(assistent_logits, 1.0f));
    assistant_states_record.push_back(
        std::move(rwkv::States(assistent_states))); // copy
    auto token = sample_func(assistant_probs);
    speculative_ids.push_back(token);
    main_model_input_ids.push_back(token);
    assistant_model_input_ids.clear();
    assistant_model_input_ids.push_back(token);
  }

  // auto speculative_text = result + tokenizer.decode(speculative_ids);

  auto main_logits = main_model.Run(main_model_input_ids, true, main_states);
  auto main_probs = rwkv::softmax(main_logits, 1.0f);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0, 1.0);
  int accepet_length = -1;
  for (int i = 0; i < speculative_length; i++) {
    float rand = dis(gen);
    auto candidate_id = speculative_ids[i];
    auto assistant_prob = assistant_probs[i].data_ptr<float>()[candidate_id];
    auto main_prob = main_probs.data_ptr<float>()[candidate_id];
    if (rand > main_prob / assistant_prob) { // reject
      accepet_length = i;
    }
  }
  if (accepet_length == -1) {
    accepet_length = speculative_length;
    assistant_model.Run(assistant_model_input_ids, assistent_states);
    assistant_states_record.push_back(
        std::move(rwkv::States(assistent_states))); // copy
  }

  int token;
  if (accepet_length < speculative_length) { // rejected or partially rejected
    auto probs = rwkv::relu(main_probs[accepet_length] -
                            assistant_probs[accepet_length]);
    probs = rwkv::div(probs, rwkv::sum(probs));
    token = sample_func(probs);
  } else {
    token = sample_func(main_probs[accepet_length]);
  }

  std::vector<int> new_tokens;
  new_tokens.insert(new_tokens.begin(), speculative_ids.begin(),
                    speculative_ids.begin() + accepet_length);
  new_tokens.push_back(token);
  auto new_text = tokenizer.decode(new_tokens);
}

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
      if (kQAMode) {
        output.data_ptr<float>()[kEndOfSentence] = -1e30;
        if (num_new_tokens == 0) {
          output.data_ptr<float>()[kNewLineId] += -1e-30;
        } else if (num_new_tokens <= kChatLenShort) {
          output.data_ptr<float>()[kNewLineId] +=
              (num_new_tokens - kChatLenShort) / 10.;
        } else if (num_new_tokens <= kChatLenLong) {
          output.data_ptr<float>()[kNewLineId] += 0;
        } else {
          output.data_ptr<float>()[kNewLineId] +=
              std::min(3.0f, (num_new_tokens - kChatLenLong) * 0.25f);
        }
      }
      auto output_id =
          sampler.Sample(output, /*temperature=*/1.f, /*top_k=*/1, kTopP);
      occurences[output_id]++;
      if (output_id == kEndOfSentence && !kQAMode) {
        break;
      }
      auto output_str = tokenizer.decode(output_id);
      std::cout << output_str;
      response += output_str;
      // it is important to pass the stop word (\n\n) to the model,
      // or it will stop incorrectly in the next iteration.
      output = Copy(model.Run(output_id), rwkv::Device::kCPU);
      if (response.size() >= 2 &&
          response.substr(response.size() - 2) == "\n\n") {
        break;
      }
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
