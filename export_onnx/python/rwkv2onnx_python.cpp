#include <filesystem>

#include <pybind11/pybind11.h>

#include <kernels/export-onnx/kernels.h>

namespace py = pybind11;

using namespace rwkv;

PYBIND11_MODULE(rwkv2onnx_python, m) {
  m.doc() = "rwkv2onnx python binding";

  m.def("convert", &onnxmeta::ExportModel, "Convert a model to ONNX format");
}

