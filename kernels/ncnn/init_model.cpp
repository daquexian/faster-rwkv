#include <fstream>
#include <iostream>

#include <cpu.h>
#include <net.h>

#include <kernels/kernels.h>
#include <kernels/registry.h>
#include <tensor.h>
#include "extra.h"
#define private public
#include <model.h>
#undef private

namespace rwkv {
namespace _ncnn {

void init_model(Model *model, Device device, const std::string &path,
                const std::string &strategy) {
  // use all big cores
#ifdef __ANDROID__
  ncnn::set_cpu_powersave(2);
#endif
  
  auto param_path = path + ".param";
  auto bin_path = path + ".bin";

  {
    auto n_layer = 0;
    std::ifstream param_file(param_path);
    int i;
    for (std::string line; std::getline(param_file, line); i++) {
      if (line.find("Input") == 0 && line.find("state_") != std::string::npos) {
        auto tmp = line.substr(line.find("state_"));
        auto name = tmp.substr(0, tmp.find(" "));
        auto layer_id = std::stoi(name.substr(6, name.find("_", 6) - 6));
        n_layer = std::max(n_layer, layer_id + 1);
        model->_n_embd = std::stoi(tmp.substr(tmp.rfind("=") + 1));
      }
    }
    model->_n_layer = n_layer;
  }
  model->_version = 4;
  auto net = std::make_shared<ncnn::Net>();
  if (model->_act_dtype == DType::kFloat16) {
    net->opt.use_fp16_packed = false;
    net->opt.use_fp16_arithmetic = false;
    net->opt.use_fp16_storage = false;
    net->opt.use_bf16_storage = true;
  } else {
    RV_CHECK(model->_act_dtype == DType::kFloat32);
    net->opt.use_fp16_packed = false;
    net->opt.use_fp16_arithmetic = false;
    net->opt.use_fp16_storage = false;
    net->opt.use_bf16_storage = false;
  }
  RV_CHECK(!net->load_param(param_path.c_str()));
  RV_CHECK(!net->load_model(bin_path.c_str()));
  int input_blob_id;
  std::vector<std::vector<int>> state_ids;
  int output_blob_id;
  std::vector<std::vector<int>> output_state_ids;
  for (int i = 0; i < net->input_names().size(); i++) {
    auto name = std::string(net->input_names()[i]);
    if (name == "input") {
      input_blob_id = net->input_indexes()[i];
    } else if (name.find("state_") != std::string::npos) {
      auto tmp = name.substr(name.find("state_"));
      auto layer_id = std::stoi(tmp.substr(6, tmp.find("_", 6) - 6));
      state_ids.resize(layer_id + 1);
      state_ids[layer_id].push_back(net->input_indexes()[i]);
    }
  }
  for (int i = 0; i < net->output_names().size(); i++) {
    auto name = std::string(net->output_names()[i]);
    if (name == "output") {
      output_blob_id = net->output_indexes()[i];
    } else if (name.find("output_state_") != std::string::npos) {
      auto tmp = name.substr(name.find("output_state_"));
      auto layer_id = std::stoi(tmp.substr(13, tmp.find("_", 13) - 13));
      output_state_ids.resize(layer_id + 1);
      output_state_ids[layer_id].push_back(net->output_indexes()[i]);
    }
  }

  model->_extra = std::make_shared<NcnnExtra>(net, input_blob_id, state_ids, output_blob_id, output_state_ids);
}

KernelRegister init_model_reg("init_model", Device::kNCNN, init_model);

} // namespace _ncnn
} // namespace rwkv
