#include <fstream>
#include <iostream>

#include <kernels/onnx-meta/kernels.h>

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr
        << "Usage: ./export_ncnn <input path> <output prefix>"
        << std::endl;
    return 1;
  }
  if (std::ifstream ifs(argv[1]); !ifs.good()) {
    std::cerr << "Failed to open " << argv[1] << std::endl;
    std::cerr
        << "Usage: ./export_ncnn <input path> <output prefix>"
        << std::endl;
    return 1;
  }
  rwkv::onnxmeta::ExportModel(argv[1], argv[2]);
  return 0;
}

