#pragma once

#include <mutex>
#include <tensor.h>
#include <tests/benchmark/random.h>
#include <unordered_map>
#include <vector>

namespace rwkv {
namespace test {
class TensorCache {
private:
  std::unordered_map<std::string, std::vector<Tensor>> dataMap;
  std::mutex mtx; // 添加互斥锁

  TensorCache() {} // 私有化构造函数，防止外部实例化

public:
  static TensorCache &get_instance() {
    static TensorCache instance; // 使用静态局部变量确保实例只被创建一次
    return instance;
  }

  TensorCache &register_tensors(const std::string &key,
                                const std::vector<Tensor> &vec) {
    std::lock_guard<std::mutex> lock(mtx); // 互斥锁保护关键部分
    dataMap[key] = vec;
    return *this;
  }

  const std::vector<Tensor> &get_tensors(const std::string &key) {
    std::lock_guard<std::mutex> lock(mtx); // 互斥锁保护关键部分
    return dataMap[key];
  }

  void unregister_tensors(const std::string &key) {
    std::lock_guard<std::mutex> lock(mtx); // 互斥锁保护关键部分
    if (dataMap.find(key) != dataMap.end()) {
      dataMap.erase(key);
    }
  }
};
} // namespace test
} // namespace rwkv
