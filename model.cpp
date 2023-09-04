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

Model::Model(const std::string &path, const std::string &strategy) {
  auto dev_str = strategy.substr(0, strategy.find(" "));
  Device act_device = [&]() {
    if (dev_str == "ncnn-meta") {
      return Device::kNCNNMeta;
    } else if (dev_str == "onnx-meta") {
      return Device::kONNXMeta;
    } else if (dev_str == "cuda") {
      return Device::kCUDA;
    } else if (dev_str == "cpu") {
      return Device::kCPU;
    } else if (dev_str == "ncnn") {
      return Device::kNCNN;
    } else {
      RV_UNIMPLEMENTED();
    }
  }();
  _act_device = act_device;
  std::string atype_str = strategy.substr(strategy.find(" ") + 1);
  DType atype = [&]() {
    if (atype_str == "fp16") {
      return DType::kFloat16;
    } else if (atype_str == "fp32") {
      return DType::kFloat32;
    } else {
      RV_UNIMPLEMENTED();
    }
  }();
  _act_dtype = atype;

  init_model(this, act_device, path, strategy);
  RV_CHECK(_n_layer > 0);
  RV_CHECK(_n_embd > 0);
  RV_CHECK(_version > 0);
  if (_version == 5) {
    RV_CHECK(_head_size > 0);
  }
  ResetStates();
}

void Model::ResetStates() {
  _states.clear();
  auto device = _act_device == Device::kNCNN ? Device::kCPU : _act_device;
  if (this->_version == 4) {
    for (int i = 0; i < _n_layer; i++) {
      _states.push_back({});
      auto s1 = Tensor::Empty(Shape{_n_embd}, _act_dtype, Device::kCPU);
      _states.back().push_back(Copy(fill_(s1, 0), device));
      auto s2 = Tensor::Empty(Shape{_n_embd}, DType::kFloat32, Device::kCPU);
      _states.back().push_back(Copy(fill_(s2, 0), device));
      auto s3 = Tensor::Empty(Shape{_n_embd}, DType::kFloat32, Device::kCPU);
      _states.back().push_back(Copy(fill_(s3, 0), device));
      auto s4 = Tensor::Empty(Shape{_n_embd}, DType::kFloat32, Device::kCPU);
      _states.back().push_back(Copy(fill_(s4, -1e30), device));
      auto s5 = Tensor::Empty(Shape{_n_embd}, _act_dtype, Device::kCPU);
      _states.back().push_back(Copy(fill_(s5, 0), device));
    }
  } else {
    for (int i = 0; i < _n_layer; i++) {
      _states.push_back({});
      auto s1 = Tensor::Empty(Shape{_n_embd}, _act_dtype, Device::kCPU);
      _states.back().push_back(Copy(fill_(s1, 0), device));
      auto s2 =
          Tensor::Empty(Shape{this->_head_size, _n_embd / this->_head_size,
                              _n_embd / this->_head_size},
                        DType::kFloat32, Device::kCPU);
      _states.back().push_back(Copy(fill_(s2, 0), device));
      auto s3 = Tensor::Empty(Shape{_n_embd}, _act_dtype, Device::kCPU);
      _states.back().push_back(Copy(fill_(s3, 0), device));
    }
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
      return out;
    }
  }
  RV_UNIMPLEMENTED();
}

Tensor Model::Run(int id) {
  if (kDebug) {
    std::cout << "Model::Run(" << id << ")" << std::endl;
  }
  return _Run(id);
}

Tensor Model::_Run(int id) {
  return ModelForward(this, this->_act_device, id, _states);
}

} // namespace rwkv
