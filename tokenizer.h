#pragma once

#include <unordered_map>
#include <string>
#include <vector>

namespace rwkv {
class Tokenizer {
  public:
    virtual ~Tokenizer() = default;
    virtual std::vector<int> encode(std::string_view str) const = 0;
    virtual std::string decode(const std::vector<int> &ids) const = 0;
    virtual std::string decode(int id) const = 0;
};

class WorldTokenizer : public Tokenizer {
public:
  WorldTokenizer(const std::string &path);
  std::vector<int> encode(std::string_view str) const;
  std::string decode(const std::vector<int> &ids) const;
  std::string decode(int id) const;

private:
  std::unordered_map<std::string, int> _word2idx;
  std::unordered_map<int, std::string> _idx2word;
};

class ABCTokenizer : public Tokenizer {
public:
  ABCTokenizer() = default;
  std::vector<int> encode(std::string_view str) const;
  std::string decode(const std::vector<int> &ids) const;
  std::string decode(int id) const;
  const int pad_token_id = 0;
  const int bos_token_id = 2;
  const int eos_token_id = 3;
};

}
