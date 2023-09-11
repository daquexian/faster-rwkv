#include <fstream>
#include <iostream>

#include <kernels/ncnn-meta/kernels.h>

int main(int argc, char **argv) {
  if (argc != 3 || argc != 4) {
    std::cerr
        << "Usage: ./export_ncnn <input path> <output prefix> [<weight_dtype>]"
        << std::endl;
    return 1;
  }
  if (std::ifstream ifs(argv[1]); !ifs.good()) {
    std::cerr << "Failed to open " << argv[1] << std::endl;
    std::cerr
        << "Usage: ./export_ncnn <input path> <output prefix> [<weight_dtype>]"
        << std::endl;
    return 1;
  }
  rwkv::DType weight_dtype;
  if (argc == 3) {
    std::cout
        << "Using fp16 weight dtype... You can also specify weight dtype by "
           "adding a third argument. For example, ./export_ncnn <input path> "
           "<output prefix> int8, which generates a faster and smaller model."
        << std::endl;
    weight_dtype = rwkv::DType::kFloat16;
  } else {
    std::string weight_dtype_str(argv[3]);
    if (weight_dtype_str == "int8" || weight_dtype_str == "i8") {
      weight_dtype = rwkv::DType::kInt8;
    } else if (weight_dtype_str == "fp16") {
      weight_dtype = rwkv::DType::kFloat16;
    } else {
      RV_UNIMPLEMENTED();
    }
  }
  rwkv::ncnnmeta::ExportModel(argv[1], weight_dtype, argv[2]);
  return 0;
}
