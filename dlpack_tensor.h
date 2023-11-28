#pragma once
#include <dlpack/dlpack.h>
#include "tensor.h"
#include <type_traits>

namespace rwkv {

DLManagedTensor* toDLPack(const Tensor& src);
Tensor fromDLPack(const ::DLManagedTensor* src);

}