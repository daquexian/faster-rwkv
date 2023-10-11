#include <fstream>
#include <iostream>
#include <sstream>

#include <experimental_onnxruntime_cxx_api.h>
#ifdef __ANDROID__
#include <onnxruntime/core/providers/nnapi/nnapi_provider_factory.h>
#endif

#include <kernels/kernels.h>
#include <kernels/onnx/extra.h>
#include <kernels/registry.h>
#include <tensor.h>
#define private public
#include <model.h>
#undef private

namespace rwkv {
namespace onnx {

static const bool kDebug = std::getenv("FR_DEBUG") != nullptr;

void init_model(Model *model, Device device, const std::string &path,
                const std::string &strategy, const std::any &extra) {
  auto env = std::make_shared<Ort::Env>();
  Ort::SessionOptions session_options;
  if (std::getenv("VERBOSE") != nullptr) {
    session_options.SetLogSeverityLevel(OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE);
  }
#ifdef __ANDROID__
  if (std::getenv("NNAPI") != nullptr) {
    uint32_t nnapi_flags = 0;
    if (std::getenv("NNAPI_FP16") != nullptr) {
      nnapi_flags |= NNAPI_FLAG_USE_FP16;
    }
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(session_options, nnapi_flags));
  }
#endif

  std::string _path(path);
  auto session = std::make_shared<Ort::Experimental::Session>(*env, _path,
                                                              session_options);

  model->_extra = std::make_shared<OnnxExtra>(env, session);
  std::string config;
  const auto config_path = path + ".config";
  std::ifstream config_file(config_path);
  if (config_file.good()) {
    std::stringstream ss;
    ss << config_file.rdbuf();
    config = ss.str();
  }
  const auto get_value = [&config](const std::string &key,
                                   std::optional<std::string> default_value =
                                       std::nullopt) {
    const std::string key_with_colon = key + ": ";
    auto pos = config.find(key_with_colon);
    if (pos == std::string::npos) {
      if (default_value.has_value()) {
        return default_value.value();
      }
      RV_UNIMPLEMENTED() << "cannot find key: " << key
                         << " and default value is not provided";
    }
    pos += key_with_colon.size();
    auto pos2 = config.find("\n", pos);
    if (pos2 == std::string::npos) {
      pos2 = config.size();
    }
    return config.substr(pos, pos2 - pos);
  };
  const auto str_to_dtype = [](const std::string &str) {
    if (str == "fp32") {
      return DType::kFloat32;
    } else if (str == "fp16") {
      return DType::kFloat16;
    } else if (str == "int8") {
      return DType::kInt8;
    } else if (str == "int4") {
      return DType::kInt4;
    } else {
      RV_UNIMPLEMENTED() << "unsupported dtype: " << str;
    }
  };

  model->_version = get_value("version");
  model->_act_dtype = str_to_dtype(get_value("act_dtype", "fp32"));
  model->_weight_dtype = str_to_dtype(get_value("weight_dtype", "fp32"));
  model->_head_size = std::stoi(get_value("head_size"));
  // overwrite these fields if it is new model (having config file)
  model->_n_embd = std::stoi(get_value("n_embd"));
  model->_n_layer = std::stoi(get_value("n_layer"));
  model->_n_att = std::stoi(get_value("n_att"));
  model->_n_ffn = std::stoi(get_value("n_ffn"));
}

KernelRegister init_model_reg("init_model", Device::kONNX, init_model);

} // namespace onnx
} // namespace rwkv
