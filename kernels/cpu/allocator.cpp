#include <cstdlib>

#include "kernels/allocator.h"
#include <kernels/registry.h>

namespace rwkv {
namespace cpu {

class Allocator : public rwkv::Allocator {
public:
  void *DoAllocate(size_t size) {
    return aligned_alloc(kAlignSize, size);
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


