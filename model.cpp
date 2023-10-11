#include "model.h"

#include "kernels/kernels.h"
#include <tensor.h>

#ifdef FR_ENABLE_CUDA
#include <cuda_runtime.h>
#endif
#include <fstream>
#include <iostream>
#include <msgpack.hpp>
#include <string>

namespace rwkv {

static const bool kDebug = std::getenv("FR_DEBUG") != nullptr;

Model::Model(const std::string &path, const std::string &strategy): Model(path, strategy, std::any()) {}

Model::Model(const std::string &path, const std::string &strategy, std::any extra) {
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
    std::cout << "activation dtype: " << dtype_to_string(_act_dtype) << std::endl;
    std::cout << "weight dtype: " << dtype_to_string(_weight_dtype) << std::endl;
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

void Model::LoadStateFile(const std::string &path) {
  std::ifstream infile;
  infile.open(path, std::ios::binary | std::ios::in);
  infile.seekg(0, std::ios::end);
  int64_t length = infile.tellg();
  infile.seekg(0, std::ios::beg);
  std::vector<char> data;
  data.resize(length);
  infile.read(data.data(), length);
  infile.close();

  auto unpacker = msgpack::unpack(data.data(), length);
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

void Model::ResetStates() {
  _states.clear();
  // TODO:
  auto device = (_act_device == Device::kNCNN || _act_device == Device::kONNX) ? Device::kCPU : _act_device;
  if (this->_version == "4") {
    for (int i = 0; i < _n_layer; i++) {
      _states.push_back({});
      auto s1 = Tensor::Empty(Shape{_n_embd}, _act_dtype, Device::kCPU);
      _states.back().push_back(Copy(fill_(s1, 0), device));
      auto s2 = Tensor::Empty(Shape{_n_att}, DType::kFloat32, Device::kCPU);
      _states.back().push_back(Copy(fill_(s2, 0), device));
      auto s3 = Tensor::Empty(Shape{_n_att}, DType::kFloat32, Device::kCPU);
      _states.back().push_back(Copy(fill_(s3, 0), device));
      auto s4 = Tensor::Empty(Shape{_n_att}, DType::kFloat32, Device::kCPU);
      _states.back().push_back(Copy(fill_(s4, -1e30), device));
      auto s5 = Tensor::Empty(Shape{_n_embd}, _act_dtype, Device::kCPU);
      _states.back().push_back(Copy(fill_(s5, 0), device));
    }
  } else {
    RV_CHECK(_version.substr(0, 1) == "5");
    for (int i = 0; i < _n_layer; i++) {
      _states.push_back({});
      auto s1 = Tensor::Empty(Shape{_n_embd}, _act_dtype, Device::kCPU);
      _states.back().push_back(Copy(fill_(s1, 0), device));
      auto s2 =
          Tensor::Empty(Shape{this->_head_size, _n_att / this->_head_size,
                              _n_embd / this->_head_size},
                        DType::kFloat32, Device::kCPU);
      _states.back().push_back(Copy(fill_(s2, 0), device));
      auto s3 = Tensor::Empty(Shape{_n_embd}, _act_dtype, Device::kCPU);
      _states.back().push_back(Copy(fill_(s3, 0), device));
    }
  }
}

Tensor CopyToCPUIfAvailable(Tensor x) {
  // TODO: more elegant
  try {
    return Copy(x, Device::kCPU);
  } catch (std::exception &e) {
    return x;
  }
}

Tensor Model::Run(const std::vector<int> &ids) {
  if (kDebug) {
    std::cout << "Model::Run([";
    for (auto id : ids) {
      std::cout << id << ", ";
    }
    std::cout << "])" << std::endl;
  }
  for (int i = 0; i < ids.size(); ++i) {
    auto id = ids[i];
    auto out = _Run(id);
    if (i == ids.size() - 1) {
      return CopyToCPUIfAvailable(out);
    }
  }
  RV_UNIMPLEMENTED();
}

Tensor Model::Run(int id) {
  if (kDebug) {
    std::cout << "Model::Run(" << id << ")" << std::endl;
  }
  return CopyToCPUIfAvailable(_Run(id));
}

Tensor Model::_Run(int id) {
  return ModelForward(this, this->_act_device, id, _states);
}

} // namespace rwkv
