#include <fstream>
#include <iostream>

#include <net.h>

#include "extra.h"
#include <kernels/kernels.h>
#include <kernels/registry.h>
#include <tensor.h>
#define private public
#include <model.h>
#undef private

namespace rwkv {
namespace _ncnn {

Tensor ModelForward(const Model *model, Device device, int id,
                    std::vector<std::vector<Tensor>> &states) {

  auto &extra = *std::any_cast<std::shared_ptr<NcnnExtra>>(model->_extra);
  auto &net = *extra.net;
  auto input_blob_id = extra.input_blob_id;
  auto &state_ids = extra.state_ids;
  auto output_blob_id = extra.output_blob_id;
  auto &output_state_ids = extra.output_state_ids;
  ncnn::Extractor ex = net.create_extractor();
  ncnn::Mat input;
  // In ncnn model we generated, blob with id `n` is the embedding weights for
  // token with id `n`
  ex.extract(id, input);
  ex.input(input_blob_id, input);
  RV_CHECK(!states.empty());
  for (int i = 0; i < states.size(); i++) {
    for (int j = 0; j < states[i].size(); j++) {
      auto &state_tensor = states[i][j];
      RV_CHECK(state_tensor.shape().size() == 1);
      RV_CHECK(state_tensor.device() == Device::kCPU);
      ncnn::Mat state_mat(state_tensor.numel(), state_tensor.data_ptr(),
                          state_tensor.elem_size());
      ex.input(state_ids[i][j], state_mat);
    }
  }
  ncnn::Mat output;
  ex.extract(output_blob_id, output);
  for (int i = 0; i < states.size(); i++) {
    for (int j = 0; j < states[i].size(); j++) {
      ncnn::Mat output_state;
      ex.extract(output_state_ids[i][j], output_state);
      auto output_state_tensor =
          Copy(Tensor::FromPtr(output_state.data, {output_state.w},
                               DType::kFloat32, Device::kCPU),
               Device::kCPU, true);
      states[i][j] = output_state_tensor;
    }
  }
  RV_CHECK(output.c == 1 && output.d == 1 && output.h == 1);
  auto ret = Copy(
      Tensor::FromPtr(output.data, {output.w}, DType::kFloat32, Device::kCPU),
      Device::kCPU, true);
  return ret;
}

KernelRegister model_forward_reg("model_forward", Device::kNCNN, ModelForward);

} // namespace _ncnn
} // namespace rwkv
