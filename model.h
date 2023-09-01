#pragma once

#include <cassert>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <any>

#include "tensor.h"

namespace rwkv {
using States = std::vector<std::vector<Tensor>>;
struct Model {
  Model(const std::string &path, const std::string &strategy);
  Tensor Run(const std::vector<int>& id);
  Tensor Run(int id);
  void ResetStates();
  void set_states(const States& states);
  States get_states() const;

  std::vector<Tensor> _embd_weights;
private:
  // std::unordered_map<std::string, Tensor> _params;
  // _params is not a map because we know the exact order of the parameters
  std::vector<Tensor> _params;
  Device _act_device;
  DType _act_dtype;
  // inited in `init_model` and checked in constructor
  int _n_layer = 0;
  int _n_embd = 0;
  std::any _extra;
  States _states;
};
} // namespace rwkv
