#include <layer.h>
#include <layer/matmul.h>
#include <mat.h>

#include <kernels/registry.h>
#include <tensor.h>

namespace rwkv {
namespace {

static int ncnn_matmul(const ncnn::Mat &a, const ncnn::Mat &b, ncnn::Mat &c) {
  ncnn::ParamDict pd;
  pd.set(0, 0); // transB

  std::vector<ncnn::Mat> weights;

  ncnn::Layer *op = ncnn::create_layer("MatMul");

  op->load_param(pd);

  ncnn::Option opt;
  opt.num_threads = 1;
  if (a.elemsize == 2) {
    RV_CHECK(op->support_fp16_storage);
    opt.use_fp16_arithmetic = true;
    opt.use_fp16_storage = true;
  }

  ncnn::ModelBinFromMatArray mb(weights.data());

  op->load_model(mb);

  op->create_pipeline(opt);

  std::vector<ncnn::Mat> outputs{c};

  ((ncnn::MatMul *)op)->ncnn::MatMul::forward({a, b}, outputs, opt);

  op->destroy_pipeline(opt);

  delete op;

  return 0;
}

Tensor matmul(Tensor a, Tensor b) {
  ncnn::Mat ncnn_x(x->numel(), x->data_ptr(), x->elem_size());
  ncnn_scalar_div(ncnn_x, divisor);
  return x;
}

KernelRegister inplace_scalar_div_reg("scalar_div_", Device::kCPU, scalar_div_);

} // namespace
} // namespace rwkv
