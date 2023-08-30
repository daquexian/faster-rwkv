#include <fstream>

#include <msgpack.hpp>

#include <kernels/kernels.h>
#include <kernels/ncnn-meta/kernels.h>
#include <kernels/registry.h>
#include <tensor.h>
#define private public
#include <model.h>
#undef private

namespace rwkv {

namespace def {

Tensor ModelForward(const Model *model, Device device, int id,
                    std::vector<std::vector<Tensor>> &states) {
  Tensor x = model->_embd_weights[id];
  auto &params = model->_params;
#ifdef FR_ENABLE_NCNN
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
#endif

  int param_idx = 0;

  for (int i = 0; i < states.size(); ++i) {
    auto &state = states[i];

    {
      // x, state[i*5+0], state[i*5+1], state[i*5+2], state[i*5+3] = ATT(
      //   x, state[i*5+0], state[i*5+1], state[i*5+2], state[i*5+3],
      //   w[f'{bbb}ln1.weight'], w[f'{bbb}ln1.bias'],
      //   w[f'{att}time_mix_k'], w[f'{att}time_mix_v'], w[f'{att}time_mix_r'],
      //   w[f'{att}time_decay'], w[f'{att}time_first'],
      //   kw, vw, rw, ow,
      //   kmx, krx, kmy, kry,
      //   vmx, vrx, vmy, vry,
      //   rmx, rrx, rmy, rry,
      //   omx, orx, omy, ory,
      // )
      std::tie(x, state[0], state[1], state[2], state[3]) = att(
          x, state[0], state[1], state[2], state[3], params[param_idx],
          params[param_idx + 1], params[param_idx + 2], params[param_idx + 3],
          params[param_idx + 4], params[param_idx + 5], params[param_idx + 6],
          params[param_idx + 7], params[param_idx + 8], params[param_idx + 9],
          params[param_idx + 10]);
      if (device == Device::kNCNNMeta) {
        mark_as_output(state[0], "output_state_" + std::to_string(i) + "_0");
        mark_as_output(state[1], "output_state_" + std::to_string(i) + "_1");
        mark_as_output(state[2], "output_state_" + std::to_string(i) + "_2");
        mark_as_output(state[3], "output_state_" + std::to_string(i) + "_3");
      }
      param_idx += 11;
    }
    {
      int offset = 4;

      // x, state[offset] =
      //     FFN(x, state[offset], w[f '{bbb}ln2.weight'], w[f '{bbb}ln2.bias'],
      //         w[f '{ffn}time_mix_k'], w[f '{ffn}time_mix_r'], kw, vw, rw,
      //         kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, )
      std::tie(x, state[offset]) = ffn(
          x, state[offset], params[param_idx], params[param_idx + 1],
          params[param_idx + 2], params[param_idx + 3], params[param_idx + 4],
          params[param_idx + 5], params[param_idx + 6]);
      if (device == Device::kNCNNMeta) {
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
  x = layernorm(x, params[param_idx], params[param_idx + 1]);

  //                 x = x @ w['head.weight']
  x = matmul(x, params[param_idx + 2]);
  if (x.dtype() == DType::kFloat16) {
    x = cast_dtype(x, DType::kFloat32);
  }
  return x;
}

KernelRegister model_forward_reg_1("model_forward", Device::kCPU, ModelForward);
KernelRegister model_forward_reg_2("model_forward", Device::kCUDA,
                                   ModelForward);
KernelRegister model_forward_reg_3("model_forward", Device::kNCNNMeta,
                                   ModelForward);

} // namespace def
} // namespace rwkv
