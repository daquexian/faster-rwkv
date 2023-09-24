#pragma once

#include <exception>
#include <stdexcept>

#define CUBLAS_CHECK(condition)                                                \
  for (cublasStatus_t _cublas_check_status = (condition);                      \
       _cublas_check_status != CUBLAS_STATUS_SUCCESS;)                         \
    throw std::runtime_error("cuBLAS error " +                                 \
                             std::to_string(_cublas_check_status) + " at " +   \
                             std::to_string(__LINE__) + " in " __FILE__);

#define CUDA_CHECK(condition)                                                  \
  for (cudaError_t _cuda_check_status = (condition);                           \
       _cuda_check_status != cudaSuccess;)                                     \
    throw std::runtime_error(                                                  \
        "CUDA error " + std::string(cudaGetErrorString(_cuda_check_status)) +  \
        " at " + std::to_string(__LINE__) + " in " __FILE__);

#define RV_CHECK(...)                                                          \
  for (bool _rv_check_status = (__VA_ARGS__); !_rv_check_status;)              \
    throw std::runtime_error("Error at " + std::to_string(__LINE__) +          \
                             " in " __FILE__);

#define RV_UNIMPLEMENTED()                                                     \
  throw std::runtime_error("Unimplemented at " + std::to_string(__LINE__) +    \
                           " in " __FILE__);

#define FR_DISALLOW_COPY(ClassName)                                            \
  ClassName(const ClassName &) = delete;                                       \
  ClassName &operator=(const ClassName &) = delete

#define FR_DISALLOW_MOVE(ClassName)                                            \
  ClassName(ClassName &&) = delete;                                            \
  ClassName &operator=(ClassName &&) = delete

#define FR_DISALLOW_COPY_AND_MOVE(ClassName)                                   \
  FR_DISALLOW_COPY(ClassName);                                                 \
  FR_DISALLOW_MOVE(ClassName)
