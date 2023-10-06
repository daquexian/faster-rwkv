#include <fstream>
#include <iostream>

#include <net.h>

#include "extra.h"
#include <kernels/graph_backend.h>
#include <kernels/kernels.h>
#include <kernels/registry.h>
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
template <> ncnn::Mat Tensor::FromTensor() const {
  RV_CHECK(device() == Device::kCPU);

  int nranks = shape().size();
  if (nranks == 3) {
    return ncnn::Mat(size(2), size(1), size(0), const_cast<void *>(data_ptr()),
                     elem_size());
  } else if (nranks == 2) {
    return ncnn::Mat(size(1), size(0), const_cast<void *>(data_ptr()),
                     elem_size());
  } else if (nranks == 1) {
    return ncnn::Mat(size(0), const_cast<void *>(data_ptr()), elem_size());
  } else {
    RV_UNIMPLEMENTED();
  }
}

// NOTE: the memory is not shared, otherwise the data may be released
template <> Tensor Tensor::ToTensor(const ncnn::Mat &ncnn_mat) {
  Shape shape;
  if (ncnn_mat.dims == 1) {
    shape = {ncnn_mat.w};
  } else if (ncnn_mat.dims == 2) {
    shape = {ncnn_mat.h, ncnn_mat.w};
  } else if (ncnn_mat.dims == 3) {
    shape = {ncnn_mat.c, ncnn_mat.h, ncnn_mat.w};
  } else {
    RV_UNIMPLEMENTED();
  }
  return Copy(
      Tensor::FromPtr(ncnn_mat.data, shape, DType::kFloat32, Device::kCPU),
      Device::kCPU, /*always_copy=*/true);
}

template <>
std::pair<ncnn::Mat, std::vector<std::vector<ncnn::Mat>>>
GraphBackendForwardInternal(const Model *model, int id,
                            const std::vector<std::vector<ncnn::Mat>> &states) {
  // Retrieve the NcnnExtra object from the model. It is created in
  // kernel/ncnn/init_model.cpp
  auto &extra = *std::any_cast<std::shared_ptr<NcnnExtra>>(model->extra());
  auto &net = *extra.net;
  auto input_blob_id = extra.input_blob_id;
  auto &state_ids = extra.state_ids;
  auto output_blob_id = extra.output_blob_id;
  auto &output_state_ids = extra.output_state_ids;
  int ncnn_impl_version = extra.ncnn_impl_version;
  ncnn::Extractor ex = net.create_extractor();
  ncnn::Mat input;
  RV_CHECK(ncnn_impl_version == 1 || ncnn_impl_version == 2) << "Invalid ncnn_impl_version: " << ncnn_impl_version;
  if (ncnn_impl_version == 1) {
    // In ncnn model we generated, blob with id `n` is the embedding weights for
    // token with id `n`
    ex.extract(id, input);
  } else {
    input = ncnn::Mat(1, &id, /*_elemsize=*/4u);
  }

  // Set ncnn input and states
  ex.input(input_blob_id, input);
  RV_CHECK(!states.empty());
  for (int i = 0; i < states.size(); i++) {
    for (int j = 0; j < states[i].size(); j++) {
      auto &state = states[i][j];
      ex.input(state_ids[i][j], state);
    }
  }

  // Run ncnn inference and get output and new states
  ncnn::Mat output;
  ex.extract(output_blob_id, output);
  std::vector<std::vector<ncnn::Mat>> new_states(states.size());
  for (int i = 0; i < states.size(); i++) {
    new_states[i].reserve(states[i].size());
    for (int j = 0; j < states[i].size(); j++) {
      ncnn::Mat output_state;
      ex.extract(output_state_ids[i][j], output_state);
      new_states[i].push_back(output_state);
    }
  }
  RV_CHECK(output.c == 1 && output.d == 1 && output.h == 1);
  return {output, new_states};
}

KernelRegister ncnn_model_forward_reg("model_forward", Device::kNCNN,
                                      GraphBackendForward<ncnn::Mat>);

} // namespace rwkv
