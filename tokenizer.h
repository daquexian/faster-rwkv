#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace rwkv {
class TokenizerBase {
public:
  TokenizerBase(int pad_token_id, int bos_token_id, int eos_token_id)
      : pad_token_id(pad_token_id), bos_token_id(bos_token_id),
        eos_token_id(eos_token_id) {}
  virtual ~TokenizerBase() = default;
  virtual std::vector<int> encode(std::string_view str) const = 0;
  virtual std::string decode(const std::vector<int> &ids) const = 0;
  virtual std::string decode(int id) const = 0;
  const int pad_token_id;
  const int bos_token_id;
  const int eos_token_id;
};

class Tokenizer {
public:
  Tokenizer(const std::string &path);
  std::vector<int> encode(std::string_view str) const {
    return _impl->encode(str);
  }
  std::string decode(const std::vector<int> &ids) const {
    return _impl->decode(ids);
  }
  std::string decode(int id) const { return _impl->decode(id); }

  int pad_token_id() const { return _impl->pad_token_id; }
  int bos_token_id() const { return _impl->bos_token_id; }
  int eos_token_id() const { return _impl->eos_token_id; }

private:
  std::shared_ptr<TokenizerBase> _impl;
};

class NormalTokenizer : public TokenizerBase {
public:
  NormalTokenizer(const std::string &path);
  std::vector<int> encode(std::string_view str) const;
  std::string decode(const std::vector<int> &ids) const;
  std::string decode(int id) const;

private:
  std::unordered_map<std::string, int> _word2idx;
  std::unordered_map<int, std::string> _idx2word;
  std::string _normalizer;
  std::string _pre_tokenizer;
};

class ABCTokenizer : public TokenizerBase {
public:
  ABCTokenizer() : TokenizerBase(/*pad_token_id*/ 0, /*bos_token_id*/ 2, /*eos_token_id*/ 3) {}
  std::vector<int> encode(std::string_view str) const;
  std::string decode(const std::vector<int> &ids) const;
  std::string decode(int id) const;
};

} // namespace rwkv
