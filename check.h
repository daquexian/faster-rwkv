#pragma once

#include <exception>
#include <stdexcept>
#include <string>

struct FRException : public std::runtime_error {
  FRException() : std::runtime_error("") {}
  const char *what() const noexcept override { return msg.c_str(); }
  FRException &operator<<(const std::string &s) {
    msg += s;
    return *this;
  }
  std::string msg;
};

#define CUBLAS_CHECK(condition)                                                \
  for (cublasStatus_t _cublas_check_status = (condition);                      \
       _cublas_check_status != CUBLAS_STATUS_SUCCESS;)                         \
  throw FRException() << ("cuBLAS error " +                                    \
                          std::to_string(_cublas_check_status) + " at " +      \
                          std::to_string(__LINE__) +                           \
                          " in " __FILE__ "\n  > Error msg: ")

#define CUDA_CHECK(condition)                                                  \
  for (cudaError_t _cuda_check_status = (condition);                           \
       _cuda_check_status != cudaSuccess;)                                     \
  throw FRException() << ("CUDA error " +                                      \
                          std::string(                                         \
                              cudaGetErrorString(_cuda_check_status)) +        \
                          " at " + std::to_string(__LINE__) +                  \
                          " in " __FILE__ "\n  > Error msg: ")

#define RV_CHECK(...)                                                          \
  for (bool _rv_check_status = (__VA_ARGS__); !_rv_check_status;)              \
  throw FRException() << ("Error at " + std::to_string(__LINE__) +             \
                          " in " __FILE__ "\n  > Error msg: ")

#define RV_UNIMPLEMENTED()                                                     \
  throw FRException() << ("Unimplemented at " + std::to_string(__LINE__) +     \
                          " in " __FILE__ "\n  > Error msg: ")

#define FR_DISALLOW_COPY(ClassName)                                            \
  ClassName(const ClassName &) = delete;                                       \
  ClassName &operator=(const ClassName &) = delete

#define FR_DISALLOW_MOVE(ClassName)                                            \
  ClassName(ClassName &&) = delete;                                            \
  ClassName &operator=(ClassName &&) = delete

#define FR_DISALLOW_COPY_AND_MOVE(ClassName)                                   \
  FR_DISALLOW_COPY(ClassName);                                                 \
  FR_DISALLOW_MOVE(ClassName)
