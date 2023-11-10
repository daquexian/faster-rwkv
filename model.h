#pragma once

#include <any>
#include <cassert>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "tensor.h"

namespace rwkv {
using States = std::vector<std::vector<Tensor>>;
struct Model {
  Model(const std::string &path, const std::string &strategy);
  Model(const std::string &path, const std::string &strategy, std::any extra);
  Tensor Run(const std::vector<int> &id, bool full_output = false,
             States *states = nullptr, bool full_state = false);
  Tensor Run(int id, States *states = nullptr);
  /*
   * The return values are {accept_length, output_ids}
   */
  std::tuple<int, std::vector<int>>
  AssistedRun(int id, Model &assistant_model, int speculative_length,
              const std::function<int(rwkv::Tensor &)> &sample_func,
              const std::function<std::string(int)> &decode_func,
              const std::vector<std::string> &antiprompts,
              const std::string &prev_response);
  void LoadStateFile(const std::string &path);
  void LoadStateFile(const std::string &path, void *asset_manager);
  void SaveStateFile(const std::string &path);
  std::shared_ptr<States> GenerateStates();
  void ResetStates();
  void set_states(const States &states);
  const States &states() const { return _states; }
  States &states() { return _states; }
  const int head_size() const { return _head_size; }
  const int n_layer() const { return _n_layer; }
  const int n_embd() const { return _n_embd; }
  const int n_att() const { return _n_att; }
  const int n_ffn() const { return _n_ffn; }
  const std::string &version() const { return _version; }
  const std::any &extra() const { return _extra; }

  DType weight_dtype() const { return _weight_dtype; }

  void RestrictStates();

  // TODO:
  std::vector<Tensor> _embd_weights;

public:
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
