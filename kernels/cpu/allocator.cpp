#include <cstdlib>

#include "kernels/allocator.h"
#include <kernels/registry.h>

namespace rwkv {
namespace cpu {

class Allocator : public rwkv::Allocator {
public:
  void *DoAllocate(size_t size) {
#ifdef __ANDROID__
    return malloc(size);
#elif defined(_MSC_VER)
    return _aligned_malloc(size, kAlignSize);
#else
    return aligned_alloc(kAlignSize, size);
#endif
  }
  void Deallocate(void *ptr) { free(ptr); }
};

rwkv::Allocator& allocator() {
  static Allocator allocator;
  return allocator;
}

KernelRegister allocator_reg("allocator", Device::kCPU, allocator);

} // namespace cuda
} // namespace rwkv


