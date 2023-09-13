#include <fstream>

#include <kernels/allocator.h>
#include <kernels/registry.h>
#include <kernels/shape/shape_inference.h>
#include <model.h>
#include <tensor.h>

#include <onnx/onnx_pb.h>

namespace rwkv {
namespace onnxmeta {

using onnx::GraphProto;
using onnx::ModelProto;
using onnx::NodeProto;
using onnx::TensorProto;
using onnx::ValueInfoProto;

static const int kOpsetVersion = 18;

GraphProto graph;

void init() {}

ModelProto destory() {
  ModelProto model;
  model.set_ir_version(7);
  model.set_producer_name("faster-rwkv");
  model.set_producer_version("0.0.1");
  *model.mutable_graph() = graph;
  graph.Clear();
  return model;
}

void ExportModel(const std::string &input_path,
                 const std::string &output_path) {

  // NOTE: fp32 here is just a placeholder. The dtype used by ncnn is determined
  // by the weight_dtype parameter.
  Model model(input_path, "onnx-meta fp32");
  model.Run(0);
  ModelProto model_proto = destory();
  // save model_proto to output_path
  std::ofstream ofs(output_path, std::ios::binary);
  RV_CHECK(ofs.good());
  model_proto.SerializeToOstream(&ofs);
}

DType dtype = DType::kFloat32;

Tensor add_input(const Shape &shape, const std::string &name) {
  Tensor output = Tensor::Empty(shape, dtype, Device::kONNXMeta);
  output.name = name;
  ValueInfoProto *input = graph.add_input();
  input->set_name(name);
  input->mutable_type()->mutable_tensor_type()->set_elem_type(
      static_cast<int>(dtype));
  for (auto dim : shape) {
    input->mutable_type()
        ->mutable_tensor_type()
        ->mutable_shape()
        ->add_dim()
        ->set_dim_value(dim);
  }
  return output;
}

Tensor constant_scalar(float x) {
  Tensor output = Tensor::Empty({1}, DType::kFloat32, Device::kONNXMeta);
  NodeProto *node = graph.add_node();
  node->set_op_type("ConstantOfShape");
  node->add_output(output.name);
  node->add_attribute()->set_name("value");
  node->mutable_attribute(0)->mutable_t()->set_data_type(TensorProto::FLOAT);
  node->mutable_attribute(0)->mutable_t()->add_float_data(x);
  return output;
}

Tensor possible_initializer(const Tensor &x) {
  if (x.device() == Device::kONNXMeta) {
    return x;
  }
  RV_CHECK(x.device() == Device::kCPU);
  Tensor output = Tensor::Empty(x.shape(), x.dtype(), Device::kONNXMeta);
  NodeProto *node = graph.add_node();
  node->set_op_type("Constant");
  node->add_output(output.name);
  node->add_attribute()->set_name("value");
  if (x.dtype() == DType::kFloat32) {
    node->mutable_attribute(0)->mutable_t()->set_data_type(TensorProto::FLOAT);
    node->mutable_attribute(0)->mutable_t()->mutable_raw_data()->assign(
        x.data_ptr<float>(), x.data_ptr<float>() + x.numel());
  } else if (x.dtype() == DType::kFloat16) {
    node->mutable_attribute(0)->mutable_t()->set_data_type(
        TensorProto::FLOAT16);
    node->mutable_attribute(0)->mutable_t()->mutable_raw_data()->assign(
        x.data_ptr<float16>(), x.data_ptr<float16>() + x.numel());
  } else {
    RV_UNIMPLEMENTED();
  }
  return output;
}

Tensor gather(const Tensor& x, const Tensor& index) {
  RV_CHECK(x.shape().size() == 2);
  RV_CHECK(index.shape().size() == 1);
  auto output = Tensor::Empty({index.shape()[1]}, x.dtype(), x.device());
  NodeProto *node = graph.add_node();
  node->set_op_type("Gather");
  node->add_input(x.name);
  node->add_input(index.name);
  node->add_output(output.name);
  return output;
}

Tensor layernorm(const Tensor &x, const Tensor &weight, const Tensor &bias) {
  RV_CHECK(x.shape().size() == 2);
  auto output = Tensor::Empty(x.shape(), x.dtype(), x.device());
  NodeProto *node = graph.add_node();
  node->set_op_type("LayerNormalization");
  node->add_input(x.name);
  node->add_input(weight.name);
  node->add_input(bias.name);
  node->add_output(output.name);
  node->add_attribute()->set_name("axis");
  node->mutable_attribute(0)->set_i(1);
  node->add_attribute()->set_name("epsilon");
  node->mutable_attribute(1)->set_f(1e-5f);
  return output;
}

Tensor matmul(const Tensor &_x, const Tensor &_y) {
  auto x = possible_initializer(_x);
  auto y = possible_initializer(_y);
  auto output =
      Tensor::Empty(shape::matmul(x.shape(), y.shape()), x.dtype(), x.device());
  NodeProto *node = graph.add_node();
  node->set_op_type("MatMul");
  node->add_input(x.name);
  node->add_input(y.name);
  node->add_output(output.name);
  return output;
}

// TODO: add shape inference

#define BROADCAST_BINARY_OP(op, onnx_type)                                     \
  Tensor op(const Tensor &_x, const Tensor &_y) {                              \
    auto x = possible_initializer(_x);                                         \
    auto y = possible_initializer(_y);                                         \
    Tensor output = Tensor::Empty(                                             \
        shape::broadcast_binary(x.shape(), y.shape()), x.dtype(), x.device()); \
    NodeProto *node = graph.add_node();                                        \
    node->set_op_type(onnx_type);                                              \
    node->add_input(x.name);                                                   \
    node->add_input(y.name);                                                   \
    node->add_output(output.name);                                             \
    return output;                                                             \
  }

BROADCAST_BINARY_OP(add, "Add")
BROADCAST_BINARY_OP(sub, "Sub")
BROADCAST_BINARY_OP(mul, "Mul")
BROADCAST_BINARY_OP(div, "Div")
BROADCAST_BINARY_OP(maximum, "Max")

Tensor rsub_scalar(float x, const Tensor &_y) {
  auto y = possible_initializer(_y);
  Tensor x_t = constant_scalar(x);
  Tensor output = Tensor::Empty(y.shape(), y.dtype(), y.device());
  NodeProto *node = graph.add_node();
  node->set_op_type("Sub");
  node->add_input(y.name);
  node->add_input(x_t.name);
  node->add_output(output.name);
  return output;
}

Tensor exp(const Tensor &x) {
  Tensor output = Tensor::Empty(x.shape(), x.dtype(), x.device());
  NodeProto *node = graph.add_node();
  node->set_op_type("Exp");
  node->add_input(x.name);
  node->add_output(output.name);
  return output;
}

Tensor relu(const Tensor &x) {
  Tensor output = Tensor::Empty(x.shape(), x.dtype(), x.device());
  NodeProto *node = graph.add_node();
  node->set_op_type("Relu");
  node->add_input(x.name);
  node->add_output(output.name);
  return output;
}

Tensor sigmoid(const Tensor &x) {
  Tensor output = Tensor::Empty(x.shape(), x.dtype(), x.device());
  NodeProto *node = graph.add_node();
  node->set_op_type("Sigmoid");
  node->add_input(x.name);
  node->add_output(output.name);
  return output;
}

Tensor reshape(const Tensor &x, const Shape &shape) {
  Tensor output = Tensor::Empty(shape, x.dtype(), x.device());
  NodeProto *node = graph.add_node();
  node->set_op_type("Reshape");
  node->add_input(x.name);
  Tensor shape_cpu_tensor = Tensor::FromPtr(const_cast<int64_t *>(shape.data()),
                                            {static_cast<long>(shape.size())},
                                            DType::kInt64, Device::kCPU);
  auto shape_tensor = possible_initializer(shape_cpu_tensor);
  node->add_input(shape_tensor.name);
  node->add_output(output.name);
  return output;
}

Tensor mark_as_output(const Tensor &x, const std::string &name) {
  ValueInfoProto *output = graph.add_output();
  output->set_name(name);
  output->mutable_type()->mutable_tensor_type()->set_elem_type(
      static_cast<int>(x.dtype()));
  for (auto dim : x.shape()) {
    output->mutable_type()
        ->mutable_tensor_type()
        ->mutable_shape()
        ->add_dim()
        ->set_dim_value(dim);
  }
  return x;
}

KernelRegister allocator_reg("allocator", Device::kONNXMeta, null_allocator);

KernelRegister layernorm_reg("layernorm", Device::kONNXMeta, layernorm);
// KernelRegister groupnorm_reg("groupnorm", Device::kONNXMeta, groupnorm);
KernelRegister matmul_reg("matmul", Device::kONNXMeta, matmul);
KernelRegister add_reg("add", Device::kONNXMeta, add);
KernelRegister sub_reg("sub", Device::kONNXMeta, sub);
KernelRegister mul_reg("mul", Device::kONNXMeta, mul);
KernelRegister div_reg("div", Device::kONNXMeta, div);
KernelRegister maximum_reg("maximum", Device::kONNXMeta, maximum);
KernelRegister rsub_reg("rsub_scalar", Device::kONNXMeta, rsub_scalar);
KernelRegister exp_reg("exp", Device::kONNXMeta, exp);
KernelRegister relu_reg("relu", Device::kONNXMeta, relu);
KernelRegister sigmoid_reg("sigmoid", Device::kONNXMeta, sigmoid);
KernelRegister reshape_reg("reshape", Device::kONNXMeta, reshape);
KernelRegister mark_as_output_reg("mark_as_output", Device::kONNXMeta,
                                  mark_as_output);

} // namespace onnxmeta
} // namespace rwkv
