#include <algorithm>
#include <any>
#include <fstream>

#include <msgpack.hpp>

#include <kernels/kernels.h>
#include <kernels/registry.h>
#include <string>
#include <tensor.h>
#define private public
#include <model.h>
#undef private

namespace rwkv {
namespace def {

inline void init_model(Model *model, Device device, const std::string &path,
                       const std::string &strategy, const std::any& extra) {
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
  auto map = obj.as<std::unordered_map<std::string, msgpack::object>>();
  auto weights = map["weights"].as<std::map<std::string, msgpack::object>>();
  auto embd_weights = map["embd_weights"].as<std::vector<msgpack::object>>();

  auto from_mp_dtype = [](const std::string &mp_dtype) -> DType {
    if (mp_dtype == "torch.int8") {
      return DType::kInt8;
    } else if (mp_dtype == "torch.float16") {
      return DType::kFloat16;
    } else if (mp_dtype == "torch.float32") {
      return DType::kFloat32;
    } else {
      throw std::runtime_error("unknown dtype " + mp_dtype);
    }
  };

  Device weight_device = device == Device::kCUDA ? Device::kCUDA : Device::kCPU;

  auto from_mp_tensor = [from_mp_dtype,
                         weight_device](msgpack::object mp_tensor,
                                        const std::string &name) -> Tensor {
    auto mp_tensor_map =
        mp_tensor.as<std::unordered_map<std::string, msgpack::object>>();
    // NOTE: `mp_tensor_data` will be destroyed after this function returns
    auto mp_tensor_data = mp_tensor_map["data"].as<std::vector<char>>();
    auto mp_tensor_shape = mp_tensor_map["shape"].as<std::vector<int64_t>>();
    auto mp_tensor_dtype = mp_tensor_map["dtype"].as<std::string>();
    auto fr_cpu_tensor =
        Tensor::FromPtr(mp_tensor_data.data(), Shape(mp_tensor_shape),
                        from_mp_dtype(mp_tensor_dtype), Device::kCPU);
    auto ret = Copy(fr_cpu_tensor, weight_device, true);
    ret.name = name;
    ret.is_constant = true;
    return ret;
  };

  auto push_param = [model, &from_mp_tensor, &weights](const std::string &key) {
    // copy the weights so that it will not be released
    model->_params.push_back(from_mp_tensor(weights[key], key));
  };
  model->_n_layer = map["n_layer"].as<int>();
  model->_n_embd = map["n_embd"].as<int>();

  if (map.find("version") == map.end()) {
    model->_version = "4";
  } else {
    model->_version = map["version"].as<std::string>();
  }
  if (model->_version.substr(0, 1) == "5") {
    model->_head_size = map["n_head"].as<int>();
    model->_n_att = map["n_att"].as<int>();
    model->_n_ffn = map["n_ffn"].as<int>();
  } else {
    RV_CHECK(model->_version == "4");
    model->_n_att = model->_n_embd;
  }

  for (int i = 0; i < model->_n_layer; i++) {
    std::string bbb_pf = "blocks." + std::to_string(i) + ".";
    std::string att_pf = "blocks." + std::to_string(i) + ".att.";
    std::string ffn_pf = "blocks." + std::to_string(i) + ".ffn.";
    // auto &kw = _params[att_pf + "key.weight"];
    // auto &vw = _params[att_pf + "value.weight"];
    // auto &rw = _params[att_pf + "receptance.weight"];
    // auto &ow = _params[att_pf + "output.weight"];
    // std::tie(x, state[0], state[1], state[2], state[3]) =
    //     att(x, state[0], state[1], state[2], state[3],
    //         _params[bbb_pf + "ln1.weight"], _params[bbb_pf + "ln1.bias"],
    //         _params[att_pf + "time_mix_k"], _params[att_pf + "time_mix_v"],
    //         _params[att_pf + "time_mix_r"], _params[att_pf + "time_decay"],
    //         _params[att_pf + "time_first"], kw, vw, rw, ow);
    //

    push_param(bbb_pf + "ln1.weight");
    push_param(bbb_pf + "ln1.bias");
    if (model->_version.substr(0, 1) == "5") {
      push_param(att_pf + "ln_x.weight");
      push_param(att_pf + "ln_x.bias");
    }
    push_param(att_pf + "time_mix_k");
    push_param(att_pf + "time_mix_v");
    push_param(att_pf + "time_mix_r");
    if (model->_version == "5.1") {
      push_param(att_pf + "time_mix_g");
    }
    push_param(att_pf + "time_decay");
    push_param(att_pf + "time_first");
    push_param(att_pf + "key.weight");
    push_param(att_pf + "value.weight");
    push_param(att_pf + "receptance.weight");
    if (model->_version == "5.1") {
      push_param(att_pf + "gate.weight");
    }
    push_param(att_pf + "output.weight");
    // std::tie(x, state[offset]) =
    //     ffn(x, state[offset], _params[bbb_pf + "ln2.weight"],
    //         _params[bbb_pf + "ln2.bias"], _params[ffn_pf + "time_mix_k"],
    //         _params[ffn_pf + "time_mix_r"], kw, vw, rw);
    push_param(bbb_pf + "ln2.weight");
    push_param(bbb_pf + "ln2.bias");
    push_param(ffn_pf + "time_mix_k");
    push_param(ffn_pf + "time_mix_r");
    push_param(ffn_pf + "key.weight");
    push_param(ffn_pf + "value.weight");
    push_param(ffn_pf + "receptance.weight");
  }
  push_param("ln_out.weight");
  push_param("ln_out.bias");
  push_param("head.weight");

  for (int i = 0; i < embd_weights.size(); i++) {
    auto mp_tensor = embd_weights[i];
    model->_embd_weights.push_back(
        from_mp_tensor(mp_tensor, std::string("embd_") + std::to_string(i)));
    if (model->_act_device == Device::kNCNNMeta) {
      // add embd_weights to ncnn bin
      Copy(model->_embd_weights.back(), Device::kNCNNMeta);
    }
  }
}

KernelRegister init_model_reg_1("init_model", Device::kCPU, init_model);
KernelRegister init_model_reg_2("init_model", Device::kCUDA, init_model);
KernelRegister init_model_reg_3("init_model", Device::kNCNNMeta, init_model);

} // namespace def
} // namespace rwkv
