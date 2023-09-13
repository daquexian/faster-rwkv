#pragma once

#include <cstddef>

namespace rwkv {
class Allocator {
public:
  void *Allocate(size_t size) {
    size = (size + kAlignSize - 1) / kAlignSize * kAlignSize;
    return DoAllocate(size);
  }
  virtual void *DoAllocate(size_t size) = 0;
  virtual void Deallocate(void *ptr) = 0;
  static const int kAlignSize = 512;
};

class NullAllocator : public rwkv::Allocator {
public:
  void *DoAllocate(size_t size) { return nullptr; }
  void Deallocate(void *ptr) {}
};

inline rwkv::Allocator &null_allocator() {
  static NullAllocator allocator;
  return allocator;
}

} // namespace rwkv
