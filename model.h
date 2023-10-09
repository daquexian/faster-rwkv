#pragma once

#include <any>
#include <cassert>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensor.h"

namespace rwkv {
using States = std::vector<std::vector<Tensor>>;
struct Model {
  Model(const std::string &path, const std::string &strategy);
  Model(const std::string &path, const std::string &strategy, std::any extra);
  Tensor Run(const std::vector<int> &id);
  Tensor Run(int id);
  void LoadStateFile(const std::string &path);
  void ResetStates();
  void set_states(const States &states);
  const States &states() const { return _states; }
  const int head_size() const { return _head_size; }
  const int n_layer() const { return _n_layer; }
  const int n_embd() const { return _n_embd; }
  const int n_att() const { return _n_att; }
  const int n_ffn() const { return _n_ffn; }
  const std::string &version() const { return _version; }
  const std::any &extra() const { return _extra; }

  // TODO:
  std::vector<Tensor> _embd_weights;

  bool should_print;

private:
  Tensor _Run(int id);
  Tensor _Run(const std::vector<int> &id);
  // _params is not a map because we know the exact order of the parameters
  std::vector<Tensor> _params;
  Device _act_device;
  DType _act_dtype;
  DType _weight_dtype;
  // inited in `init_model` and checked in constructor
  int _n_layer = 0;
  int _n_embd = 0;
  int _n_att = 0;
  int _n_ffn = 0;
  int _head_size = 0;
  std::string _version;
  std::any _extra;
  States _states;
};
} // namespace rwkv
