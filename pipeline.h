#pragma once

#include <map>
#include <memory>
#include <model.h>
#include <sampler.h>
#include <tokenizer.h>

namespace rwkv {

static const std::string kDoubleNewLine = "\n\n";

class ChatPipeline {
public:
  ChatPipeline(const std::shared_ptr<Model> &model,
               const std::shared_ptr<Tokenizer> &tokenizer)
      : _model(model), _tokenizer(tokenizer),
        _sampler(std::make_shared<GreedySampler>()) {}
  std::string Run(const std::string &input) {
    std::vector<int> input_ids = _tokenizer->encode(input);
    auto output_tensor = _model->Run(input_ids);
    for (auto &[id, occurence] : _occurences) {
      output_tensor.data_ptr<float>()[id] -=
          kFrequencyPenalty * occurence + kPresencePenalty;
      occurence *= kPenaltyDecay;
    }
    output_tensor.data_ptr<float>()[kEndOfSentence] = -1e30;
    if (num_new_tokens == 0) {
      output_tensor.data_ptr<float>()[kNewLineId] += -1e-30;
    } else if (num_new_tokens <= kChatLenShort) {
      output_tensor.data_ptr<float>()[kNewLineId] +=
          (num_new_tokens - kChatLenShort) / 10.;
    } else if (num_new_tokens <= kChatLenLong) {
      output_tensor.data_ptr<float>()[kNewLineId] += 0;
    } else {
      output_tensor.data_ptr<float>()[kNewLineId] +=
          std::min(3.0f, (num_new_tokens - kChatLenLong) * 0.25f);
    }
    auto output_id = _sampler->Sample(output_tensor.data_ptr<float>(),
                                      output_tensor.numel());
    return _tokenizer->decode(output_id);
  }

private:
  const std::shared_ptr<Model> _model;
  const std::shared_ptr<Tokenizer> _tokenizer;
  const std::shared_ptr<Sampler> _sampler;
  std::map<int, float> _occurences;
  static const int kMaxOutputLength = 999;
  static const int kEndOfSentence = 0;
  static const int kNewLineId = 11;
  static const int kChatLenShort = 40;
  static const int kChatLenLong = 150;
  constexpr static const float kPresencePenalty = 0.4;
  constexpr static const float kFrequencyPenalty = 0.4;
  constexpr static const float kPenaltyDecay = 0.996;
};
} // namespace rwkv
