

#define FR_LAUNCH_CUDA_KERNEL_BASE_256(kernel, type, ndim, total, ...)         \
  if (total % 256 == 0) {                                                      \
    kernel<type, ndim><<<total / 256, 256>>>(total, 0, __VA_ARGS__);           \
  } else {                                                                     \
    auto pieces = total / 256;                                                 \
    if (pieces == 0) {                                                         \
      kernel<type, ndim><<<1, total>>>(total, 0, __VA_ARGS__);                 \
    } else {                                                                   \
      kernel<type, ndim><<<pieces, 256>>>(total, 0, __VA_ARGS__);              \
      kernel<type, ndim>                                                       \
          <<<1, total - pieces * 256>>>(total, pieces * 256, __VA_ARGS__);     \
    }                                                                          \
  }

#define FR_LAUNCH_CUDA_KERNEL_NO_ALLOC_BASE_256(kernel, total, ...)            \
  if (total % 256 == 0) {                                                      \
    kernel<<<total / 256, 256>>>(total, 0, __VA_ARGS__);                       \
  } else {                                                                     \
    auto pieces = total / 256;                                                 \
    if (pieces == 0) {                                                         \
      kernel<<<1, total>>>(total, 0, __VA_ARGS__);                             \
    } else {                                                                   \
      kernel<<<pieces, 256>>>(total, 0, __VA_ARGS__);                          \
      kernel<<<1, total - pieces * 256>>>(total, pieces * 256, __VA_ARGS__);   \
    }                                                                          \
  }
