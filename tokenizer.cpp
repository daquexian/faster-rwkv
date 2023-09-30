#include "tokenizer.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string_view>
#include <optional>

#include <msgpack.hpp>

#include <utils.h>

namespace rwkv {

struct TrieTree {
  std::unordered_map<int, std::unique_ptr<TrieTree>> children;
  std::string word;
  std::optional<int> token_id;
  
  TrieTree(const std::unordered_map<std::string, int>& word2id) {
    for (auto& pair : word2id) {
      add_word(pair.first, pair.second);
    }
  }

  std::pair<std::string, int> find_longest_prefix(const std::string& str) const {
    std::string prefix;
    int token_id = -1;
    const TrieTree* node = this;
    for (int i = 0; i < str.size(); ++i) {
      auto it = node->children.find(str[i]);
      if (it == node->children.end()) {
        break;
      }
      node = it->second.get();
      RV_CHECK(node != nullptr);
      if (node->token_id.has_value()) {
        prefix = node->word;
        token_id = node->token_id.value();
      }
    }
    RV_CHECK(!prefix.empty());
    RV_CHECK(token_id != -1);
    return {prefix, token_id};
  }

  private:
  TrieTree() = default;
  void add_word(const std::string& word, int token_id) {
    return _add_word(word, token_id, 0);
  }
  void _add_word(const std::string& word, int token_id, int idx) {
    if (idx == word.size()) {
      this->word = word;
      this->token_id = token_id;
      return;
    }
    auto& child = children[word[idx]];
    if (!child) {
      child = std::unique_ptr<TrieTree>(new TrieTree());
    }
    child->_add_word(word, token_id, idx + 1);
  }
};

Tokenizer::Tokenizer(std::filesystem::path path, void *asset_manager) {
  if (path.empty()) {
    _impl = std::make_shared<ABCTokenizer>();
    return;
  }
  if (std::filesystem::is_directory(path)) {
    path /= "tokenizer";
  }
  const std::string data = read_file(path, asset_manager);

  auto unpacker = msgpack::unpack(data.data(), data.size());
  auto obj = unpacker.get();
  const std::string type = [&]() -> std::string {
    try {
      auto map = obj.as<std::unordered_map<std::string, msgpack::object>>();
      if (map.find("type") != map.end()) {
        return map["type"].as<std::string>();
      }
    } catch (const std::exception &e) {
      // do nothing
    }
    // legacy format, NormalTokenizer
    return "NormalTokenizer";
  }();
  if (type == "NormalTokenizer") {
    _impl = std::make_shared<NormalTokenizer>(obj);
  } else if (type == "SimpleABCTokenizer") {
    _impl = std::make_shared<ABCTokenizer>();
  } else {
    RV_UNIMPLEMENTED() << "Unsupported tokenizer type: " << type;
  }
}

NormalTokenizer::NormalTokenizer(msgpack::object obj) : TokenizerBase(0, 0, 0) {
  try {
    auto dict = obj.as<std::unordered_map<std::string, msgpack::object>>();
    if (dict.find("type") != dict.end()) {
      RV_CHECK(dict["type"].as<std::string>() == "NormalTokenizer");
    }
    _idx2word = dict["idx2word"].as<std::unordered_map<int, std::string>>();
    if (dict.find("normalizer") != dict.end()) {
      _normalizer = dict["normalizer"].as<std::string>();
    }
    if (dict.find("pre_tokenizer") != dict.end()) {
      _pre_tokenizer = dict["pre_tokenizer"].as<std::string>();
    }
  } catch (const std::exception &e) {
    // legacy world tokenizer format
    _idx2word = obj.as<std::unordered_map<int, std::string>>();
  }
  for (auto &pair : _idx2word) {
    _word2idx[pair.second] = pair.first;
  }
  _tree = std::make_unique<TrieTree>(_word2idx);
}

std::vector<int> NormalTokenizer::encode(std::string_view _str) const {
  std::string str;
  if (_normalizer == "Lowercase") {
    for (int i = 0; i < _str.size(); ++i) {
      str += std::tolower(_str[i]);
    }
  } else if (_normalizer.empty()) {
    str = _str;
  } else {
    RV_UNIMPLEMENTED() << "Unknown normalizer: " << _normalizer;
  }
  std::vector<std::string> pieces;
  if (_pre_tokenizer == "WhitespaceSplit") {
    std::string buf;
    std::stringstream ss{std::string(str)};
    while (ss >> buf) {
      pieces.push_back(buf);
    }
    std::vector<int> ids;
    for (auto &piece : pieces) {
      auto it = _word2idx.find(piece);
      if (it == _word2idx.end()) {
        RV_UNIMPLEMENTED();
      } else {
        ids.push_back(it->second);
      }
    }
    return ids;
  } else if (_pre_tokenizer.empty()) {
    std::vector<int> ids;
    int str_idx = 0;

    while (str_idx < str.size()) {
      auto [prefix, token_id] = _tree->find_longest_prefix(str.substr(str_idx));
      ids.push_back(token_id);
      str_idx += prefix.size();
    }
    return ids;
  } else {
    RV_UNIMPLEMENTED() << "Unknown pre_tokenizer: " << _pre_tokenizer;
  }
}

std::string NormalTokenizer::decode(int id) const {
  auto it = _idx2word.find(id);
  if (it == _idx2word.end()) {
    return "<unk>";
  } else {
    return it->second;
  }
}

std::string NormalTokenizer::decode(const std::vector<int> &ids) const {
  std::string str;
  for (auto id : ids) {
    str += decode(id);
  }
  return str;
}

std::vector<int> ABCTokenizer::encode(std::string_view str) const {
  std::vector<int> ids;
  for (int i = 0; i < str.size(); ++i) {
    ids.push_back(str[i]);
  }
  return ids;
}

std::string ABCTokenizer::decode(int id) const {
  if (id <= eos_token_id) {
    return "";
  } else {
    return std::string(1, static_cast<char>(id));
  }
}

std::string ABCTokenizer::decode(const std::vector<int> &ids) const {
  std::string str;
  for (auto id : ids) {
    str += decode(id);
  }
  return str;
}

} // namespace rwkv
