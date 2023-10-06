#include <fstream>

#include <kernels/allocator.h>
#include <kernels/registry.h>
#include <kernels/shape/shape_inference.h>
#include <model.h>
#include <tensor.h>

#include <onnx/checker.h>
#include <onnx/onnx_pb.h>

namespace rwkv {
namespace onnxmeta {

using onnx::GraphProto;
using onnx::ModelProto;
using onnx::NodeProto;
using onnx::TensorProto;
using onnx::ValueInfoProto;

static const int kOpsetVersion = 18;

int *get_unique_op_id_ptr() {
  static int _unique_id = 0;
  return &_unique_id;
}

int unique_op_id() { return (*get_unique_op_id_ptr())++; }

void reset_unique_op_id() { *get_unique_op_id_ptr() = 0; }

GraphProto graph;

NodeProto *new_node() {
  auto node = graph.add_node();
  node->set_name(std::to_string(unique_op_id()));
  return node;
}

void init() {}

ModelProto destory() {
  ModelProto model;
  model.set_ir_version(7);
  model.add_opset_import()->set_domain("");
  model.mutable_opset_import(0)->set_version(kOpsetVersion);
  model.set_producer_name("faster-rwkv");
  model.set_producer_version("0.0.1");
  graph.set_name("main");
  *model.mutable_graph() = graph;
  graph.Clear();
  onnx::checker::check_model(model);
  return model;
}

void ExportModel(const std::string &input_path,
                 const std::string &output_path) {

  // NOTE: fp32 here is just a placeholder. The dtype used by ncnn is determined
  // by the weight_dtype parameter.
  Model model(input_path, "export-onnx fp32");
  model.Run(0);
  ModelProto model_proto = destory();
  // save model_proto to output_path
  std::ofstream ofs(output_path, std::ios::binary);
  RV_CHECK(ofs.good());
  model_proto.SerializeToOstream(&ofs);
}

DType _dtype = DType::kFloat32;

int fr_dtype_to_onnx_dtype(DType fr_dtype) {
  if (fr_dtype == DType::kFloat32) {
    return TensorProto::FLOAT;
  } else if (fr_dtype == DType::kFloat16) {
    return TensorProto::FLOAT16;
  } else if (fr_dtype == DType::kInt64) {
    return TensorProto::INT64;
  } else {
    RV_UNIMPLEMENTED();
  }
}

Tensor add_input(const Shape &shape, DType dtype, const std::string &name) {
  Tensor output = Tensor::Empty(shape, dtype, Device::kONNXMeta);
  output.name = name;
  ValueInfoProto *input = graph.add_input();
  input->set_name(name);
  input->mutable_type()->mutable_tensor_type()->set_elem_type(
      fr_dtype_to_onnx_dtype(dtype));
  auto *onnx_shape =
      input->mutable_type()->mutable_tensor_type()->mutable_shape();
  for (auto dim : shape) {
    onnx_shape->add_dim()->set_dim_value(dim);
  }
  return output;
}

Tensor constant_scalar(float x) {
  Tensor output = Tensor::Empty({1}, DType::kFloat32, Device::kONNXMeta);
  NodeProto *node = new_node();
  node->set_op_type("Constant");
  node->add_output(output.name);
  node->add_attribute()->set_name("value");
  node->mutable_attribute(0)->set_type(onnx::AttributeProto::TENSOR);
  node->mutable_attribute(0)->mutable_t()->set_data_type(TensorProto::FLOAT);
  node->mutable_attribute(0)->mutable_t()->add_float_data(x);
  node->mutable_attribute(0)->mutable_t()->mutable_dims()->Add(1);
  return output;
}

Tensor possible_initializer(const Tensor &x) {
  if (x.device() == Device::kONNXMeta) {
    return x;
  }
  RV_CHECK(x.device() == Device::kCPU);
  Tensor output = Tensor::Empty(x.shape(), x.dtype(), Device::kONNXMeta);
  NodeProto *node = new_node();
  node->set_name(std::to_string(unique_op_id()));
  node->set_op_type("Constant");
  node->add_output(output.name);
  node->add_attribute()->set_name("value");
  node->mutable_attribute(0)->set_type(onnx::AttributeProto::TENSOR);
  node->mutable_attribute(0)->mutable_t()->set_data_type(
      fr_dtype_to_onnx_dtype(x.dtype()));
  node->mutable_attribute(0)->mutable_t()->mutable_raw_data()->assign(
      static_cast<const char *>(x.data_ptr<>()), x.numel() * x.elem_size());
  node->mutable_attribute(0)->mutable_t()->mutable_dims()->CopyFrom(
      {x.shape().begin(), x.shape().end()});
  return output;
}

Tensor gather(const Tensor &x, const Tensor &index) {
  RV_CHECK(x.shape().size() == 2);
  RV_CHECK(index.shape().size() == 0);
  auto output = Tensor::Empty({x.shape()[1]}, x.dtype(), x.device());
  NodeProto *node = new_node();
  node->set_op_type("Gather");
  node->add_input(x.name);
  node->add_input(index.name);
  node->add_output(output.name);
  return output;
}

Tensor layernorm(const Tensor &x, const Tensor &_weight, const Tensor &_bias) {
  auto output = Tensor::Empty(x.shape(), x.dtype(), x.device());
  auto weight = possible_initializer(_weight);
  auto bias = possible_initializer(_bias);
  NodeProto *node = new_node();
  node->set_op_type("LayerNormalization");
  node->add_input(x.name);
  node->add_input(weight.name);
  node->add_input(bias.name);
  node->add_output(output.name);
  node->add_attribute()->set_name("axis");
  node->mutable_attribute(0)->set_i(1);
  node->mutable_attribute(0)->set_type(onnx::AttributeProto::INT);
  node->add_attribute()->set_name("epsilon");
  node->mutable_attribute(1)->set_f(1e-5f);
  node->mutable_attribute(1)->set_type(onnx::AttributeProto::FLOAT);
  return output;
}

Tensor matmul(const Tensor &_x, const Tensor &_y) {
  auto x = possible_initializer(_x);
  auto y = possible_initializer(_y);
  auto output =
      Tensor::Empty(shape::matmul(x.shape(), y.shape()), x.dtype(), x.device());
  NodeProto *node = new_node();
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
    NodeProto *node = new_node();                                              \
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
  NodeProto *node = new_node();
  node->set_op_type("Sub");
  node->add_input(y.name);
  node->add_input(x_t.name);
  node->add_output(output.name);
  return output;
}

Tensor exp(const Tensor &x) {
  Tensor output = Tensor::Empty(x.shape(), x.dtype(), x.device());
  NodeProto *node = new_node();
  node->set_op_type("Exp");
  node->add_input(x.name);
  node->add_output(output.name);
  return output;
}

Tensor relu(const Tensor &x) {
  Tensor output = Tensor::Empty(x.shape(), x.dtype(), x.device());
  NodeProto *node = new_node();
  node->set_op_type("Relu");
  node->add_input(x.name);
  node->add_output(output.name);
  return output;
}

Tensor sigmoid(const Tensor &x) {
  Tensor output = Tensor::Empty(x.shape(), x.dtype(), x.device());
  NodeProto *node = new_node();
  node->set_op_type("Sigmoid");
  node->add_input(x.name);
  node->add_output(output.name);
  return output;
}

Tensor reshape(const Tensor &x, const Shape &shape) {
  Tensor output = Tensor::Empty(shape, x.dtype(), x.device());
  NodeProto *node = new_node();
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
      fr_dtype_to_onnx_dtype(_dtype));
  std::cout << "output name: " << name << ", dim: " << x.shape() << std::endl;
  for (auto dim : x.shape()) {
    output->mutable_type()
        ->mutable_tensor_type()
        ->mutable_shape()
        ->add_dim()
        ->set_dim_value(dim);
  }
  Tensor output_tensor = Tensor::Empty(x.shape(), x.dtype(), x.device());
  output_tensor.name = name;
  NodeProto *node = new_node();
  node->set_op_type("Identity");
  node->add_input(x.name);
  node->add_output(name);
  return output_tensor;
}

KernelRegister allocator_reg("allocator", Device::kONNXMeta, null_allocator);

KernelRegister layernorm_reg("layernorm", Device::kONNXMeta, layernorm);
KernelRegister groupnorm_reg("groupnorm", Device::kONNXMeta, groupnorm);
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
