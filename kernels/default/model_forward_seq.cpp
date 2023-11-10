#include "check.h"
#include <cstdlib>
#include <fstream>
#include <iostream>

#include <msgpack.hpp>

#include <kernels/export-ncnn/kernels.h>
#include <kernels/kernels.h>
#include <vector>
#ifdef FR_ENABLE_ONNX
#include <kernels/export-onnx/kernels.h>
#endif
#include <kernels/registry.h>
#include <string>
#include <tensor.h>
#define private public
#include <model.h>
#undef private

namespace rwkv {

namespace def {

static Tensor CopyToCPUIfAvailable(Tensor x) {
  // TODO: more elegant
  try {
    return Copy(x, Device::kCPU);
  } catch (std::exception &e) {
    return x;
  }
}

Tensor ModelForwardSeqFallback(Model *model, Device device,
                               const std::vector<int> &ids, bool full_output,
                               States *input_states) {
  RV_CHECK(!full_output) << "full_output is not supported in fallback mode";
  for (int i = 0; i < ids.size(); ++i) {
    auto id = ids[i];
    auto out = ModelForward(model, model->_act_device, id, input_states);
    if (i == ids.size() - 1) {
      return CopyToCPUIfAvailable(out);
    }
  }
  RV_UNIMPLEMENTED();
}

/*
 * full_state should be an empty vector and will be filled when return.
 */
Tensor ModelForwardSeq(Model *model, Device device, const std::vector<int> &id,
                       bool full_output, States *input_states,
                       bool full_state) {
  auto &states = input_states != nullptr ? *input_states : model->states();
  RV_CHECK(input_states != nullptr || !full_state);
  Tensor x = [&]() -> Tensor {
#ifdef FR_ENABLE_ONNX
    RV_UNIMPLEMENTED();
    // if (model->_act_device == Device::kONNXMeta) {
    //   Tensor input_id = onnxmeta::add_input({}, DType::kInt64, "input_id");
    //   Tensor embd_weights_cpu =
    //       Tensor::Empty({static_cast<long>(model->_embd_weights.size()),
    //                      model->_embd_weights[0].shape()[0]},
    //                     DType::kFloat32, Device::kCPU);
    //   {
    //     float *ptr = embd_weights_cpu.data_ptr<float>();
    //     for (int i = 0; i < model->_embd_weights.size(); i++) {
    //       for (int j = 0; j < model->_n_embd; j++) {
    //         // embd weights in .fr are always fp16
    //         *ptr++ = static_cast<float>(
    //             model->_embd_weights[i].data_ptr<float16>()[j]);
    //       }
    //     }
    //   }

    //   Tensor embd_weights = onnxmeta::possible_initializer(embd_weights_cpu);
    //   return onnxmeta::gather(embd_weights, input_id);
    //   }
#endif
    return vgather(model->_embd_weights, id);
  }();

  auto &params = model->_params;
  if (model->_act_device == Device::kNCNNMeta) {
    x = ncnnmeta::add_input(x.shape(), "input");
    for (int i = 0; i < states.size(); i++) {
      for (int j = 0; j < states[i].size(); j++) {
        auto state_name =
            "state_" + std::to_string(i) + "_" + std::to_string(j);
        auto &state_tensor = states[i][j];
        state_tensor = ncnnmeta::add_input(state_tensor.shape(), state_name);
      }
    }
  }
#ifdef FR_ENABLE_ONNX
  if (model->_act_device == Device::kONNXMeta) {
    for (int i = 0; i < states.size(); i++) {
      for (int j = 0; j < states[i].size(); j++) {
        auto state_name =
            "state_" + std::to_string(i) + "_" + std::to_string(j);
        auto &state_tensor = states[i][j];
        state_tensor = onnxmeta::add_input(state_tensor.shape(),
                                           DType::kFloat32, state_name);
      }
    }
  }
#endif

  int param_idx = 0;

  for (int i = 0; i < states.size(); ++i) {
    auto &state = states[i];

    {
      if (model->_version == "4") {
        RV_UNIMPLEMENTED();
      } else if (model->_version == "5") {
        if (full_state)
          RV_UNIMPLEMENTED();
        std::tie(x, state[0], state[1]) = att_seq_v5(
            x, state[0], state[1], params[param_idx], params[param_idx + 1],
            params[param_idx + 2], params[param_idx + 3], params[param_idx + 4],
            params[param_idx + 5], params[param_idx + 6], params[param_idx + 7],
            params[param_idx + 8], params[param_idx + 9],
            params[param_idx + 10], params[param_idx + 11],
            params[param_idx + 12]);
        if (device == Device::kNCNNMeta || device == Device::kONNXMeta) {
          mark_as_output(state[0], "output_state_" + std::to_string(i) + "_0");
          mark_as_output(state[1], "output_state_" + std::to_string(i) + "_1");
        }
        param_idx += 13;
      } else if (model->_version == "5.1") {
        RV_UNIMPLEMENTED();
      } else if (model->_version == "5.2") {
        Tensor &state0 = state[0];
        Tensor &state1 = state[1];
        if (state0.sizes().size() > 1) {
          state0 = state0.slice({Range(-1, 1, state0.size(0)), Range::All})
                       .squeeze(0);
        }
        if (state1.sizes().size() > 3) {
          state0 = state1
                       .slice({Range(-1, 1, state1.size(0)), Range::All,
                               Range::All, Range::All})
                       .squeeze(0);
        }
        std::tie(x, state[0], state[1]) = att_seq_v5_2(
            x, state0, state1, params[param_idx], params[param_idx + 1],
            params[param_idx + 2], params[param_idx + 3], params[param_idx + 4],
            params[param_idx + 5], params[param_idx + 6], params[param_idx + 7],
            params[param_idx + 8], params[param_idx + 9],
            params[param_idx + 10], params[param_idx + 11],
            params[param_idx + 12], params[param_idx + 13],
            params[param_idx + 14], model->n_att(), full_state);
        if (device == Device::kNCNNMeta || device == Device::kONNXMeta) {
          mark_as_output(state[0], "output_state_" + std::to_string(i) + "_0");
          mark_as_output(state[1], "output_state_" + std::to_string(i) + "_1");
        }
        param_idx += 15;
      } else {
        RV_UNIMPLEMENTED();
      }
    }
    {
      int offset = 4;
      if (model->_version.substr(0, 1) == "5") {
        offset = 2;
      }

      Tensor &ffn_state = state[offset];
      if (ffn_state.sizes().size() > 1) {
        ffn_state =
            ffn_state.slice({Range(-1, 1, ffn_state.size(0)), Range::All})
                .squeeze(0);
      }

      std::tie(x, state[offset]) = ffn_seq(
          x, ffn_state, params[param_idx], params[param_idx + 1],
          params[param_idx + 2], params[param_idx + 3], params[param_idx + 4],
          params[param_idx + 5], params[param_idx + 6], full_state);
      if (device == Device::kNCNNMeta || device == Device::kONNXMeta) {
        mark_as_output(state[offset], "output_state_" + std::to_string(i) +
                                          "_" + std::to_string(offset));
      }
      param_idx += 7;
    }

    if (x.dtype() == DType::kFloat16 && (i + 1) % 6 == 0) {
      scalar_div_(x, 2);
    }
  }

  //             x = F.layer_norm(x, (args.n_embd,),
  //             weight=w['ln_out.weight'], bias=w['ln_out.bias'])
  if (!full_output) {
    x = x.slice({Range(-1, 1, x.size(0)), Range::All}).squeeze(0);
  }
  x = layernorm(x, params[param_idx], params[param_idx + 1]);

  if (params[param_idx + 2].dtype() == DType::kInt8) {
    RV_UNIMPLEMENTED();
  }

  //                 x = x @ w['head.weight']
  x = matmul(x, params[param_idx + 2]);
  if (x.dtype() == DType::kFloat16) {
    x = cast_dtype(x, DType::kFloat32);
  }
  if (device == Device::kNCNNMeta || device == Device::kONNXMeta) {
    mark_as_output(x, "output");
  }
  return x;
} // namespace def

KernelRegister model_forward_seq_reg_1("model_forward_seq", Device::kCPU,
                                       ModelForwardSeqFallback);
KernelRegister model_forward_seq_reg_2("model_forward_seq", Device::kCUDA,
                                       ModelForwardSeq);
KernelRegister model_forward_seq_reg_3("model_forward_seq", Device::kNCNN,
                                       ModelForwardSeqFallback);
KernelRegister model_forward_seq_reg_4("model_forward_seq", Device::kONNX,
                                       ModelForwardSeqFallback);

} // namespace def
} // namespace rwkv
