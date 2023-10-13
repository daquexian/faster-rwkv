#include <fstream>
#include <iostream>

#include <kernels/export-onnx/kernels.h>

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr
        << "Usage: " << argv[0] << " <input path> <output prefix> <dtype>"
        << std::endl;
    return 1;
  }
  if (std::ifstream ifs(argv[1]); !ifs.good()) {
    std::cerr << "Failed to open " << argv[1] << std::endl;
    std::cerr
        << "Usage: " << argv[0] << " <input path> <output prefix> <dtype>"
        << std::endl;
    return 1;
  }
  rwkv::onnxmeta::ExportModel(argv[1], argv[2], argv[3]);
  return 0;
}

