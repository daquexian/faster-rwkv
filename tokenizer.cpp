#include "tokenizer.h"

#include "check.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string_view>

#include <msgpack.hpp>

namespace rwkv {
WorldTokenizer::WorldTokenizer(const std::string &path) {
  std::ifstream infile;
  infile.open(path, std::ios::binary | std::ios::in);
  infile.seekg(0, std::ios::end);
  int64_t length = infile.tellg();
  infile.seekg(0, std::ios::beg);
  char *data = new char[length];
  infile.read(data, length);
  infile.close();

  auto unpacker = msgpack::unpack(data, length);
  auto obj = unpacker.get();
  _idx2word = obj.as<std::unordered_map<int, std::string>>();
  for (auto &pair : _idx2word) {
    _word2idx[pair.second] = pair.first;
  }
}

std::vector<int> WorldTokenizer::encode(std::string_view str) const {
  std::vector<int> ids;
  int str_idx = 0;
  int word_len = 1;
  int id = 0;
  while (str_idx < str.size()) {
    if (str_idx + word_len > str.size()) {
      ids.push_back(id);
      break;
    }
    auto substr = str.substr(str_idx, word_len);
    auto it = _word2idx.find(std::string(substr));
    if (it == _word2idx.end()) {
      ids.push_back(id);
      str_idx += (word_len - 1);
      word_len = 1;
    } else {
      id = it->second;
      word_len++;
    }
  }
  return ids;
}

std::string WorldTokenizer::decode(int id) const {
  auto it = _idx2word.find(id);
  if (it == _idx2word.end()) {
    return "<unk>";
  } else {
    return it->second;
  }
}

std::string WorldTokenizer::decode(const std::vector<int> &ids) const {
  std::string str;
  for (auto id : ids) {
    str += decode(id);
  }
  return str;
}

MIDITokenizer::MIDITokenizer(const std::string &path) {
  std::ifstream infile;
  infile.open(path, std::ios::binary | std::ios::in);
  infile.seekg(0, std::ios::end);
  int64_t length = infile.tellg();
  infile.seekg(0, std::ios::beg);
  char *data = new char[length];
  infile.read(data, length);
  infile.close();

  auto unpacker = msgpack::unpack(data, length);
  auto obj = unpacker.get();
  auto dict = obj.as<std::unordered_map<std::string, msgpack::object>>();
  _idx2word = dict["idx2word"].as<std::unordered_map<int, std::string>>();
  for (auto &pair : _idx2word) {
    _word2idx[pair.second] = pair.first;
  }
  _normalizer = dict["normalizer"].as<std::string>();
  _pre_tokenizer = dict["pre_tokenizer"].as<std::string>();
}

std::vector<int> MIDITokenizer::encode(std::string_view str) const {
  std::string tmp;
  if (_normalizer == "Lowercase") {
    for (int i = 0; i < str.size(); ++i) {
      tmp += std::tolower(str[i]);
    }
  } else {
    RV_UNIMPLEMENTED();
  }
  std::vector<std::string> pieces;
  if (_pre_tokenizer == "WhitespaceSplit") {
    std::string buf;
    std::stringstream ss{std::string(str)};
    while (ss >> buf) {
      pieces.push_back(buf);
    }
  } else {
    RV_UNIMPLEMENTED();
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
}

std::string MIDITokenizer::decode(int id) const {
  auto it = _idx2word.find(id);
  if (it == _idx2word.end()) {
    return "<unk>";
  } else {
    return it->second;
  }
}

std::string MIDITokenizer::decode(const std::vector<int> &ids) const {
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
