#include "tensor.h"

namespace rwkv {
namespace cuda {
Tensor gemm_cublas_tensor(const Tensor &a, const Tensor &b);
void gemm_cublas_tensor(const Tensor &a, const Tensor &b, Tensor &c);

void gemm_cublas(const void *a, const void *b, void *c, int batch, int ori_m,
                 int ori_n, int ori_k, DType input_dtype, DType output_dtype);
void gemm_cublas(const half *a, const half *b, float *c, int batch, int ori_m,
                 int ori_n, int ori_k);
void gemm_cublas(const half *a, const half *b, half *c, int batch, int ori_m,
                 int ori_n, int ori_k);
void gemm_cublas(const float *a, const float *b, float *c, int batch, int ori_m,
                 int ori_n, int ori_k);
} // namespace cuda
} // namespace rwkv
