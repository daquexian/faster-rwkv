#include "model.h"

#include "check.h"
#include "kernels/kernels.h"
#include <memory>
#include <tensor.h>
#include <utils.h>

#ifdef FR_ENABLE_CUDA
#include <cuda_runtime.h>
#endif
#include <fstream>
#include <iostream>
#include <msgpack.hpp>
#include <random>
#include <string>

namespace rwkv {

static const bool kDebug = std::getenv("FR_DEBUG") != nullptr;

Model::Model(const std::string &path, const std::string &strategy)
    : Model(path, strategy, std::any()) {}

Model::Model(const std::string &path, const std::string &strategy,
             std::any extra) {
  auto dev_str = strategy.substr(0, strategy.find(" "));
  Device act_device = [&]() {
    if (dev_str == "export-ncnn") {
      return Device::kNCNNMeta;
    } else if (dev_str == "export-onnx") {
      return Device::kONNXMeta;
    } else if (dev_str == "cuda") {
      return Device::kCUDA;
    } else if (dev_str == "cpu") {
      return Device::kCPU;
    } else if (dev_str == "ncnn") {
      return Device::kNCNN;
    } else if (dev_str == "onnx") {
      return Device::kONNX;
    } else {
      RV_UNIMPLEMENTED();
    }
  }();
  _act_device = act_device;
  std::tie(_act_dtype, _weight_dtype) = [&]() -> std::pair<DType, DType> {
    std::string dtype_str = strategy.substr(strategy.find(" ") + 1);
    if (dtype_str == "int4") {
      return {DType::kFloat32, DType::kInt4};
    } else if (dtype_str == "int8") {
      return {DType::kFloat32, DType::kInt8};
    } else if (dtype_str == "fp16") {
      return {DType::kFloat16, DType::kFloat16};
    } else if (dtype_str == "fp32") {
      return {DType::kFloat32, DType::kFloat32};
    } else if (dtype_str == "auto") {
      // init them in backend
      return {DType::kUndefined, DType::kUndefined};
    } else {
      RV_UNIMPLEMENTED();
    }
  }();

  init_model(this, act_device, path, strategy, extra);
  if (kDebug) {
    std::cout << "Model inited" << std::endl;
    std::cout << "version: " << _version << std::endl;
    std::cout << "activation dtype: " << dtype_to_string(_act_dtype)
              << std::endl;
    std::cout << "weight dtype: " << dtype_to_string(_weight_dtype)
              << std::endl;
    std::cout << "head_size: " << _head_size << std::endl;
    std::cout << "n_embd: " << _n_embd << std::endl;
    std::cout << "n_layer: " << _n_layer << std::endl;
    std::cout << "n_att: " << _n_att << std::endl;
    std::cout << "n_ffn: " << _n_ffn << std::endl;
  }
  RV_CHECK(!_version.empty());
  RV_CHECK(_act_dtype != DType::kUndefined);
  RV_CHECK(_weight_dtype != DType::kUndefined);
  RV_CHECK(_n_layer > 0);
  RV_CHECK(_n_embd > 0);
  RV_CHECK(_n_att > 0);
  if (_version.substr(0, 1) == "5") {
    RV_CHECK(_head_size > 0);
    RV_CHECK(_n_ffn > 0);
  }
  ResetStates();
}

void Model::SaveStateFile(const std::string &path) {
  std::vector<std::vector<std::unordered_map<std::string, msgpack::object>>>
      mp_states;

  auto dtype_to_string_in_msgpack = [](DType dtype) {
    if (dtype == DType::kFloat32) {
      return "torch.float32";
    } else if (dtype == DType::kFloat16) {
      return "torch.float16";
    } else if (dtype == DType::kInt8) {
      return "torch.int8";
    } else {
      RV_UNIMPLEMENTED();
    }
  };
  msgpack::zone z;
  for (const auto &state : _states) {
    std::vector<std::unordered_map<std::string, msgpack::object>> mp_state;
    for (const auto &s : state) {
      std::unordered_map<std::string, msgpack::object> mp_s;
      std::vector<char> data_vec;
      data_vec.resize(s.numel() * s.elem_size());
      memcpy(data_vec.data(), s.data_ptr(), s.numel() * s.elem_size());
      mp_s["dtype"] = msgpack::object(dtype_to_string_in_msgpack(s.dtype()), z);
      mp_s["data"] = msgpack::object(data_vec, z);
      mp_s["shape"] = msgpack::object(s.shape(), z);
      mp_state.push_back(mp_s);
    }
    mp_states.push_back(mp_state);
  }

  std::ofstream ofs(path);
  msgpack::pack(ofs, mp_states);
}

void Model::LoadStateFile(const std::string &path) {
  return LoadStateFile(path, nullptr);
}

void Model::LoadStateFile(const std::string &path, void *asset_manager) {
  const std::string data = read_file(path, asset_manager);

  auto unpacker = msgpack::unpack(data.data(), data.length());
  auto obj = unpacker.get();
  auto states_mp = obj.as<std::vector<std::vector<msgpack::object>>>();
  RV_CHECK(states_mp.size() == _states.size());
  for (int i = 0; i < states_mp.size(); i++) {
    RV_CHECK(states_mp[i].size() == _states[i].size());
    for (int j = 0; j < states_mp[i].size(); j++) {
      const auto &state_mp = states_mp[i][j];
      auto new_state = Tensor::FromMsgPack(state_mp);
      RV_CHECK(new_state.shape() == _states[i][j].shape());
      _states[i][j] = new_state;
    }
  }
}

std::shared_ptr<States> Model::GenerateStates() {
  std::shared_ptr<States> states = std::make_shared<States>();
  auto device = (_act_device == Device::kNCNN || _act_device == Device::kONNX)
                    ? Device::kCPU
                    : _act_device;
  if (this->_version == "4") {
    for (int i = 0; i < _n_layer; i++) {
      states->push_back({});
      auto s1 = Tensor::Empty(Shape{_n_embd}, _act_dtype, Device::kCPU);
      states->back().push_back(Copy(fill_(s1, 0), device));
      auto s2 = Tensor::Empty(Shape{_n_att}, DType::kFloat32, Device::kCPU);
      states->back().push_back(Copy(fill_(s2, 0), device));
      auto s3 = Tensor::Empty(Shape{_n_att}, DType::kFloat32, Device::kCPU);
      states->back().push_back(Copy(fill_(s3, 0), device));
      auto s4 = Tensor::Empty(Shape{_n_att}, DType::kFloat32, Device::kCPU);
      states->back().push_back(Copy(fill_(s4, -1e30), device));
      auto s5 = Tensor::Empty(Shape{_n_embd}, _act_dtype, Device::kCPU);
      states->back().push_back(Copy(fill_(s5, 0), device));
    }
  } else {
    RV_CHECK(_version.substr(0, 1) == "5");
    for (int i = 0; i < _n_layer; i++) {
      states->push_back({});
      auto s1 = Tensor::Empty(Shape{_n_embd}, _act_dtype, Device::kCPU);
      states->back().push_back(Copy(fill_(s1, 0), device));
      auto s2 = Tensor::Empty(Shape{this->_head_size, _n_att / this->_head_size,
                                    _n_embd / this->_head_size},
                              DType::kFloat32, Device::kCPU);
      states->back().push_back(Copy(fill_(s2, 0), device));
      auto s3 = Tensor::Empty(Shape{_n_embd}, _act_dtype, Device::kCPU);
      states->back().push_back(Copy(fill_(s3, 0), device));
    }
  }
  return states;
}

void Model::ResetStates() {
  _states.clear();
  _states = std::move(*GenerateStates());
}

static Tensor CopyToCPUIfAvailable(Tensor x) {
  // TODO: more elegant
  try {
    return Copy(x, Device::kCPU);
  } catch (std::exception &e) {
    return x;
  }
}

Tensor Model::Run(const std::vector<int> &ids, States *states) {
  if (kDebug) {
    std::cout << "[seq mode]Model::Run(";
    for (auto id : ids) {
      std::cout << id << ", ";
    }
    std::cout << ")" << std::endl;
  }
  if (ids.size() == 1) {
    return CopyToCPUIfAvailable(
        ModelForward(this, this->_act_device, ids[0], states));
  } else {
    return CopyToCPUIfAvailable(
        ModelForwardSeq(this, this->_act_device, ids, false, states));
  }
}

Tensor Model::Run(int id, States *states) {
  return CopyToCPUIfAvailable(
      ModelForward(this, this->_act_device, id, states));
}

std::tuple<int, std::vector<int>>
Model::AssistedRun(int id, rwkv::Model &assistant_model, int speculative_length,
                   std::function<int(const rwkv::Tensor &)> &sample_func) {
  RV_CHECK(this->version() == "5.2" && assistant_model.version() == "5.2");
  std::vector<rwkv::States> assistant_states_record;
  auto assistent_states = assistant_model.states();
  auto main_states = this->states();

  std::vector<rwkv::Tensor> assistant_probs;
  std::vector<int> speculative_ids;
  std::vector<int> main_model_input_ids;
  for (int i = 0; i < speculative_length; i++) {
    auto assistent_logits = assistant_model.Run(id, &assistent_states);
    auto assistant_prob = rwkv::softmax(assistent_logits, 1.0f);
    assistant_probs.push_back(assistant_prob);
    assistant_states_record.push_back(
        std::move(rwkv::States(assistent_states))); // copy
    auto token = sample_func(assistant_prob);
    speculative_ids.push_back(token);
    main_model_input_ids.push_back(token);
    id = token;
  }

  auto main_logits =
      this->Run(main_model_input_ids, true, &main_states); // full output
  std::vector<Tensor> main_probs;
  for (int i = 0; i < main_logits.size(0); i++) {
    main_probs.push_back(rwkv::softmax(
        main_logits.slice({Range(i, 1, i + 1), Range::All}).squeeze(0), 1.0f));
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0, 1.0);
  int accepet_length = -1;
  for (int i = 0; i < speculative_length; i++) {
    float rand = dis(gen);
    auto candidate_id = speculative_ids[i];
    auto assistant_prob = assistant_probs[i].get_item<float>({candidate_id});
    auto main_prob = main_probs[i].get_item<float>({candidate_id});
    if (rand > main_prob / assistant_prob) { // reject
      accepet_length = i;
    }
  }
  if (accepet_length == -1) {
    accepet_length = speculative_length;
    assistant_model.Run(id, &assistent_states);
    assistant_states_record.push_back(
        std::move(rwkv::States(assistent_states))); // copy
  }

  int token;
  if (accepet_length < speculative_length) { // rejected or partially rejected
    auto probs = rwkv::relu(main_probs[accepet_length] -
                            assistant_probs[accepet_length]);
    probs = rwkv::div(probs, rwkv::reduce(probs, "sum"));
    token = sample_func(probs);
  } else {
    token = sample_func(main_probs[accepet_length]);
  }

  std::vector<int> new_tokens;
  new_tokens.insert(new_tokens.begin(), speculative_ids.begin(),
                    speculative_ids.begin() + accepet_length);
  new_tokens.push_back(token);

  auto reset_states = [&accepet_length](States &states) {
    for (int i = 0; i < states.size(); i++) {
      auto &state = states[i];
      if (state[0].sizes().size() > 1) {
        state[0] = state[0]
                       .slice({Range(accepet_length - 1, 1, accepet_length),
                               Range::All})
                       .squeeze(0);
      }
      if (state[1].sizes().size() > 3) {
        state[1] = state[1]
                       .slice({Range(accepet_length - 1, 1, accepet_length),
                               Range::All, Range::All, Range::All})
                       .squeeze(0);
      }
      if (state[2].sizes().size() > 1) {
        state[2] = state[0]
                       .slice({Range(accepet_length - 1, 1, accepet_length),
                               Range::All})
                       .squeeze(0);
      }
    }
  };
  reset_states(main_states);
  reset_states(assistent_states);
  return {accepet_length, new_tokens};
}

} // namespace rwkv
