#include <model.h>
#include <tensor.h>

namespace rwkv {

template <typename T>
std::pair<T, std::vector<std::vector<T>>>
GraphBackendForwardInternal(const Model *model, int id,
                     std::vector<std::vector<T>> &&states);

template <typename T>
Tensor GraphBackendForward(Model *model, Device device, int id) {
  auto &states = model->states();

  std::vector<std::vector<T>> backend_states(states.size());
  for (int i = 0; i < states.size(); i++) {
    backend_states[i].reserve(states[i].size());
    for (int j = 0; j < states[i].size(); j++) {
      backend_states[i].push_back(states[i][j].FromTensor<T>());
    }
  }

  auto [backend_output, new_backend_states] =
      GraphBackendForwardInternal(model, id, std::move(backend_states));

  for (int i = 0; i < states.size(); i++) {
    for (int j = 0; j < states[i].size(); j++) {
      states[i][j] = Tensor::ToTensor(new_backend_states[i][j]);
    }
  }

  return Tensor::ToTensor(backend_output);
}

} // namespace rwkv
