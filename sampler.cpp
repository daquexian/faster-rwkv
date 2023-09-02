#include "sampler.h"

#include <string>

#include "check.h"

namespace rwkv {
int Sampler::Sample(const float *ptr, int len) {
  RV_CHECK(len >= 1);
  float max = ptr[0];
  int max_idx = 0;
  for (int i = 1; i < len; i++) {
    if (ptr[i] > max) {
      max = ptr[i];
      max_idx = i;
    }
  }
  return max_idx;
}
} // namespace rwkv
