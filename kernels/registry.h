#ifndef _KERNELS_REGISTRY_H_
#define _KERNELS_REGISTRY_H_
#include <any>
#include <map>
#include <stdexcept>
#include <string>
#include <tensor.h>
#include <utility>

namespace rwkv {
class KernelRegistry {
public:
  static KernelRegistry &Instance() {
    static KernelRegistry instance;
    return instance;
  }
  void Register(const std::string &name, Device device, std::any kernel, int priority) {
    _kernels[std::make_pair(name, device)] = kernel;
  }
  template <typename T> T Get(const std::string &name, Device device) {
    if (_kernels.find(std::make_pair(name, device)) == _kernels.end()) {
      throw std::runtime_error("kernel " + name + " not found");
    }
    return std::any_cast<T>(_kernels[std::make_pair(name, device)]);
  }

private:
  std::map<std::pair<std::string, Device>, std::any> _kernels;
};

struct KernelRegister {
  KernelRegister(const std::string &name, Device device, std::any kernel, int priority = 1) {
    KernelRegistry::Instance().Register(name, device, kernel, priority);
  }
};

} // namespace rwkv
#endif // _KERNELS_REGISTRY_H_
