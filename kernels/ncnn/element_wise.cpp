#include <layer.h>
#include <layer/binaryop.h>
#include <mat.h>

#include <kernels/registry.h>
#include <tensor.h>

namespace rwkv {
namespace {

static int ncnn_scalar_div(ncnn::Mat &a, float b) {
  ncnn::ParamDict pd;
  pd.set(0, ncnn::BinaryOp::Operation_DIV);
  pd.set(1, 1); // with_scalar
  pd.set(2, b); // b

  std::vector<ncnn::Mat> weights;

  ncnn::Layer *op = ncnn::create_layer("BinaryOp");

  op->load_param(pd);

  ncnn::Option opt;
  opt.num_threads = 1;
  opt.use_vulkan_compute = false;
  opt.use_int8_inference = false;
  opt.use_packing_layout = false;
  if (a.elemsize == 2) {
    RV_CHECK(op->support_fp16_storage);
    opt.use_fp16_arithmetic = true;
    opt.use_fp16_storage = true;
  }

  ncnn::ModelBinFromMatArray mb(weights.data());

  op->load_model(mb);

  op->create_pipeline(opt);

  ((ncnn::BinaryOp *)op)->ncnn::BinaryOp::forward_inplace(a, opt);

  op->destroy_pipeline(opt);

  delete op;

  return 0;
}

Tensor& scalar_div_(Tensor &x, float divisor) {
  ncnn::Mat ncnn_x(x.numel(), const_cast<void*>(x.data_ptr()), x.elem_size());
  ncnn_scalar_div(ncnn_x, divisor);
  return x;
}

KernelRegister inplace_scalar_div_reg("scalar_div_", Device::kCPU, scalar_div_);

} // namespace
} // namespace rwkv
