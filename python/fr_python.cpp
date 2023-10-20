#include <filesystem>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include <model.h>
#include <sampler.h>
#include <tensor.h>
#include <tokenizer.h>
#include <kernels/kernels.h>

namespace py = pybind11;
using namespace pybind11::literals;

using namespace rwkv;

// shared memory
Tensor FRTensorFromPyBuffer(py::buffer b) {
  py::buffer_info info = b.request();
  DType dtype;
  if (info.format == py::format_descriptor<float>::format()) {
    dtype = DType::kFloat32;
  } else if (info.format == py::format_descriptor<int8_t>::format()) {
    dtype = DType::kInt8;
  } else if (info.format == py::format_descriptor<int16_t>::format()) {
    dtype = DType::kFloat16;
  } else {
    RV_UNIMPLEMENTED() << "Unsupported dtype: " << info.format;
  }
  Shape shape(info.shape.begin(), info.shape.end());
  return Tensor::FromPtr(info.ptr, shape, dtype, Device::kCPU);
}

PYBIND11_MODULE(fr_python, m) {
  m.doc() = "faster-rwkv python binding";

  m.def("layernorm", [](const Tensor &x, const Tensor &weight,
                        const Tensor &bias) {
    return layernorm(Copy(x, Device::kCUDA), Copy(weight, Device::kCUDA), Copy(bias, Device::kCUDA));
    }
        , "x"_a, "weight"_a, "bias"_a);

  py::class_<Tokenizer, std::shared_ptr<Tokenizer>>(m, "Tokenizer")
      .def(py::init<std::filesystem::path>())
      .def("encode", &Tokenizer::encode)
      .def("decode", py::overload_cast<int>(&Tokenizer::decode, py::const_))
      .def("decode", py::overload_cast<const std::vector<int> &>(
                         &Tokenizer::decode, py::const_));

  py::class_<Sampler, std::shared_ptr<Sampler>>(m, "Sampler")
      .def(py::init<>())
      .def("_sample", &Sampler::Sample, "logits"_a, "temperature"_a = 1.0,
           "top_k"_a = 50, "top_p"_a = 1.0)
      .def("set_seed", &Sampler::set_seed);

  py::class_<Model, std::shared_ptr<Model>>(m, "Model")
      .def(py::init(
          [](const std::filesystem::path &path, const std::string &strategy) {
            return std::make_shared<Model>(std::string(path), strategy);
          }))
      .def("_run", py::overload_cast<const std::vector<int> &>(&Model::Run))
      .def("_run", py::overload_cast<int>(&Model::Run))
      .def("states", py::overload_cast<>(&Model::states, py::const_))
      .def("reset_states", &Model::ResetStates);

  py::class_<Tensor, std::shared_ptr<Tensor>>(m, "_Tensor",
                                              py::buffer_protocol())
      .def(py::init([](py::buffer b) -> std::shared_ptr<Tensor> {
        return std::make_shared<Tensor>(FRTensorFromPyBuffer(b));
      }))
      .def("cpu", [](Tensor &t) { return Copy(t, Device::kCPU, false); })
      .def_buffer([](Tensor &t) -> py::buffer_info {
        auto format_descriptor = [&]() {
          switch (t.dtype()) {
          case DType::kFloat32:
            return py::format_descriptor<float>::format();
          case DType::kFloat16:
            // NOTE: we export fp16 tensor as int16 and view it as float16 in
            // python,
            // because py::format_descriptor doesn't support float16
            return py::format_descriptor<int16_t>::format();
          case DType::kInt8:
            return py::format_descriptor<int8_t>::format();
          default:
            RV_UNIMPLEMENTED() << "Unsupported dtype: "
                               << std::to_string(static_cast<int>(t.dtype()));
          }
        }();
        std::vector<int> strides(t.shape().size(), elem_size(t.dtype()));
        for (int i = t.shape().size() - 2; i >= 0; i--) {
          strides[i] = strides[i + 1] * t.shape()[i + 1];
        }
        return py::buffer_info(t.data_ptr(), elem_size(t.dtype()),
                               format_descriptor, t.shape().size(), t.shape(),
                               strides);
      });
}
