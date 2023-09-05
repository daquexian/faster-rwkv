#include <kernels/ncnn-meta/kernels.h>

int main(int argc, char **argv) {
  // ./export_ncnn <output prefix> <input path>
  rwkv::ncnnmeta::ExportModel(argv[2], argv[1]);
  return 0;
}
