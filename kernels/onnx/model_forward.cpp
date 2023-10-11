#include <fstream>
#include <iostream>

#include <experimental_onnxruntime_cxx_api.h>

#include <kernels/graph_backend.h>
#include <kernels/kernels.h>
#include <kernels/registry.h>
#include <kernels/onnx/extra.h>
#include <model.h>
#include <tensor.h>

namespace rwkv {

// Implement three functions for graph-grained backends like ncnn:
// 1. T Tensor::FromTensor() const
// 2. Tensor Tensor::ToTensor(const T &backend_tensor)
// 3. std::pair<T, std::vector<std::vector<T>>> GraphBackendForwardInternal(
//         const Model *model, int id,
//         const std::vector<std::vector<T>> &states)
// 
// And register the backend with:
// KernelRegister xxxxx_model_forward_reg("model_forward", Device::kXXX,
//         GraphBackendForward<T>);

// NOTE: the memory is shared here. You can also copy it if you want.
template <> Ort::Value Tensor::FromTensor() const {
  RV_CHECK(device() == Device::kCPU);
  DType fr_dtype = dtype();
  ONNXTensorElementDataType onnx_dtype;
  if (fr_dtype == DType::kFloat32) {
    onnx_dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  } else {
    RV_UNIMPLEMENTED() << "Unsupported dtype: " << fr_dtype;
  }
  return Ort::Experimental::Value::CreateTensor(const_cast<void*>(data_ptr()), numel() * elem_size(), shape(), onnx_dtype);
}

// NOTE: the memory is not shared, otherwise the data may be released
template <> Tensor Tensor::ToTensor(const Ort::Value &ort_value) {
  const void* dptr = ort_value.GetTensorRawData();
  auto onnx_dtype = ort_value.GetTensorTypeAndShapeInfo().GetElementType();
  DType fr_dtype;
  if (onnx_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    fr_dtype = DType::kFloat32;
  } else {
    RV_UNIMPLEMENTED() << "Unsupported onnx dtype: " << onnx_dtype;
  }
  auto shape = ort_value.GetTensorTypeAndShapeInfo().GetShape();
  auto fr_tensor = Tensor::FromPtr(const_cast<void*>(dptr), shape, fr_dtype, Device::kCPU);
  return Copy(fr_tensor, Device::kCPU, true);
}

template <>
std::pair<Ort::Value, std::vector<std::vector<Ort::Value>>>
GraphBackendForwardInternal(const Model *model, int id,
                            std::vector<std::vector<Ort::Value>> &&states) {
  auto &extra = *std::any_cast<std::shared_ptr<OnnxExtra>>(model->extra());
  auto &session = *extra.session;

  int64_t id_int64 = id;
  auto input_id_t = Ort::Experimental::Value::CreateTensor(&id_int64, 1, {});
  auto input_names = session.GetInputNames();
  std::vector<Ort::Value> input_values;
  input_values.push_back(std::move(input_id_t));
  for (auto& tmp : states) {
    for (auto& s : tmp) {
      input_values.push_back(std::move(s));
    }
  }
  // states are invalid after this line
  auto output_names = session.GetOutputNames();
  auto output_values = session.Run(input_names, input_values, output_names);

  auto output = std::move(output_values[output_values.size() - 1]);
  int state_num_per_layer = states[0].size();
  std::vector<std::vector<Ort::Value>> new_states;
  for (int i = 0; i < states.size(); i++) {
    new_states.emplace_back();
    for (int j = 0; j < state_num_per_layer; j++) {
      new_states[i].push_back(std::move(output_values[i * state_num_per_layer + j]));
    }
  }

  return {std::move(output), std::move(new_states)};
}

KernelRegister onnx_model_forward_reg("model_forward", Device::kONNX,
                                      GraphBackendForward<Ort::Value>);

} // namespace rwkv

