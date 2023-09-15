#include <fstream>
#include <iostream>
#include <sstream>

#include <kernels/kernels.h>
#include <kernels/registry.h>
#include <tensor.h>
#include "extra.h"
#define private public
#include <model.h>
#undef private

namespace rwkv {
namespace onnx {

static const bool kDebug = std::getenv("FR_DEBUG") != nullptr;

void init_model(Model *model, Device device, const std::string &path,
                const std::string &strategy, const std::any& extra) {
  
}

}
}
