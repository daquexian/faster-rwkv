#include <iostream>
#include <layer.h>
#include <mat.h>

#include <tensor.h>
#include <kernels/registry.h>

namespace rwkv {

namespace {
static int ncnn_cast(const ncnn::Mat &a, ncnn::Mat &b, int type_from,
                     int type_to) {
  ncnn::ParamDict pd;
  pd.set(0, type_from);
  pd.set(1, type_to);

  std::vector<ncnn::Mat> weights(0);

  ncnn::Option opt;
  opt.num_threads = 1;
  opt.use_vulkan_compute = false;
  opt.use_int8_inference = false;
  opt.use_packing_layout = false;

  ncnn::Layer *op = ncnn::create_layer("Cast");

  op->load_param(pd);

  ncnn::ModelBinFromMatArray mb(weights.data());

  op->load_model(mb);

  op->create_pipeline(opt);

  op->forward(a, b, opt);

  op->destroy_pipeline(opt);

  delete op;

  return 0;
}

Tensor cast_dtype(const Tensor& x, DType dtype) {
  if (x.dtype() == dtype) {
    return x;
  }
  RV_CHECK(dtype == DType::kFloat32 || dtype == DType::kFloat16);
  RV_CHECK(x.dtype() == DType::kFloat32 || x.dtype() == DType::kFloat16);
  auto y = Tensor::Empty(x.shape(), dtype, x.device());
  ncnn::Mat ncnn_x(x.numel(), const_cast<void*>(x.data_ptr()), x.elem_size());
  ncnn::Mat ncnn_y(y.numel(), y.data_ptr(), y.elem_size());
  if (dtype == DType::kFloat16) {
    ncnn_cast(ncnn_x, ncnn_y, 1, 2);
  } else if (dtype == DType::kFloat32) {
    ncnn_cast(ncnn_x, ncnn_y, 2, 1);
  } else {
    RV_UNIMPLEMENTED();
  }
  return y;
}

KernelRegister cast_dtype_reg("cast_dtype", Device::kCPU, cast_dtype);
} // namespace

} // namespace rwkv
