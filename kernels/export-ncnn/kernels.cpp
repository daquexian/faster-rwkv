#include <kernels/export-ncnn/kernels.h>

#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <string>

#include <kernels/allocator.h>
#include <kernels/registry.h>
#include <kernels/shape/shape_inference.h>
#include <model.h>
#include <tensor.h>

namespace rwkv {
namespace cpu {
Tensor cast_dtype(const Tensor &x, DType dtype);
}
namespace ncnnmeta {

int *get_unique_layer_id_ptr() {
  static int _unique_id = 0;
  return &_unique_id;
}

int unique_layer_id() { return (*get_unique_layer_id_ptr())++; }

void reset_unique_layer_id() { *get_unique_layer_id_ptr() = 0; }

int *get_blob_num_ptr() {
  static int _blob_num = 0;
  return &_blob_num;
}

int add_and_get_blob_num(int num) {
  int *ptr = get_blob_num_ptr();
  *ptr += num;
  return *ptr;
}

void reset_blob_num() { *get_blob_num_ptr() = 0; }

bool* get_disable_int4() {
  static bool _disable_int4 = false;
  return &_disable_int4;
}

void disable_int4(bool flag) {
  *get_disable_int4() = flag;
}

FILE *bp, *pp;
std::string _pp_path;
std::string _config_path;
DType _weight_dtype;

void init(DType weight_dtype, const std::string &bp_path,
          const std::string &pp_path, const std::string &config_path) {
  reset_blob_num();
  reset_unique_layer_id();
  bp = fopen(bp_path.c_str(), "wb");
  pp = fopen(pp_path.c_str(), "wb");
  _pp_path = pp_path;
  _config_path = config_path;
  _weight_dtype = weight_dtype;
}

void destroy(const Model &model) {
  fclose(bp);
  fclose(pp);
  {
    std::ifstream t(_pp_path);
    std::stringstream buffer;
    buffer << t.rdbuf();
    t.close();
    std::string pp_str = buffer.str();
    int layer_num = unique_layer_id();
    int blob_num = add_and_get_blob_num(0);
    pp_str = "7767517\n" + std::to_string(layer_num) + " " +
             std::to_string(blob_num) + "\n" + pp_str;
    std::ofstream out(_pp_path);
    out << pp_str;
    out.close();
  }
  {
    std::ofstream config_file(_config_path);
    config_file << "version: " << model.version() << std::endl;
    config_file << "act_dtype: " << dtype_to_string(DType::kFloat32)
                << std::endl;
    config_file << "weight_dtype: " << dtype_to_string(_weight_dtype)
                << std::endl;
    config_file << "head_size: " << model.head_size() << std::endl;
    config_file << "n_layer: " << model.n_layer() << std::endl;
    config_file << "n_embd: " << model.n_embd() << std::endl;
    config_file << "n_att: " << model.n_att() << std::endl;
    config_file << "n_ffn: " << model.n_ffn() << std::endl;
    config_file.close();
  }
}

void ExportModel(const std::string &input_path, DType weight_dtype,
                 const std::string &output_prefix) {
  RV_CHECK(weight_dtype == DType::kFloat16 || weight_dtype == DType::kInt8 ||
           weight_dtype == DType::kInt4);

  rwkv::ncnnmeta::init(weight_dtype, output_prefix + ".bin",
                       output_prefix + ".param", output_prefix + ".config");

  // NOTE: fp32 here is just a placeholder. The dtype used by ncnn is determined
  // by the weight_dtype parameter.
  rwkv::Model model(input_path, "export-ncnn fp32");
  model.Run(0);
  rwkv::ncnnmeta::destroy(model);
}

void append_data_to_bin_file(const Tensor &tensor, bool write_tag) {
  RV_CHECK(tensor.device() == Device::kCPU);
  // RV_CHECK(tensor.is_constant);
  if (write_tag) {
    if (tensor.dtype() == DType::kInt8) {
      unsigned int int8_flag = 0x000D4B38;
      fwrite((const char *)&int8_flag, sizeof(int8_flag), 1, bp);
    } else if (tensor.dtype() == DType::kFloat16) {
      unsigned int fp16_flag = 0x01306B47;
      fwrite((const char *)&fp16_flag, sizeof(fp16_flag), 1, bp);
    } else {
      RV_CHECK(tensor.dtype() == DType::kFloat32);
      unsigned int fp32_flag = 0;
      fwrite((const char *)&fp32_flag, sizeof(fp32_flag), 1, bp);
    }
  }
  fwrite(tensor.data_ptr(), tensor.elem_size(), tensor.numel(), bp);
}

#define PRINT_OP_TYPE_AND_NAME(op_type, input_num, output_num)                 \
  fprintf(pp, "%-16s", op_type);                                               \
  {                                                                            \
    std::string tmp = std::to_string(unique_layer_id());                       \
    fprintf(pp, " %-24s", tmp.c_str());                                        \
  }                                                                            \
  add_and_get_blob_num(output_num);                                            \
  fprintf(pp, " %d %d", input_num, output_num);

Tensor add_input(const Shape &shape, const std::string &name) {
  PRINT_OP_TYPE_AND_NAME("Input", 0, 1);
  auto output = Tensor::Empty(shape, DType::kFloat32, Device::kNCNNMeta);
  output.name = name;
  fprintf(pp, " %s", output.name.c_str());
  if (shape.size() == 4) {
    fprintf(pp, " 0=%d", (int)shape[3]);
    fprintf(pp, " 1=%d", (int)shape[2]);
    fprintf(pp, " 2=%d", (int)shape[1]);
    fprintf(pp, " 3=%d", (int)shape[0]);
  } else if (shape.size() == 3) {
    fprintf(pp, " 0=%d", (int)shape[2]);
    fprintf(pp, " 1=%d", (int)shape[1]);
    fprintf(pp, " 2=%d", (int)shape[0]);
  } else if (shape.size() == 2) {
    fprintf(pp, " 0=%d", (int)shape[1]);
    fprintf(pp, " 1=%d", (int)shape[0]);
  } else if (shape.size() == 1) {
    fprintf(pp, " 0=%d", (int)shape[0]);
  } else {
    RV_UNIMPLEMENTED();
  }
  fprintf(pp, "\n");
  return output;
}

Tensor layernorm(const Tensor &x, const Tensor &weight, const Tensor &bias) {
  PRINT_OP_TYPE_AND_NAME("LayerNorm", 1, 1);
  auto output = Tensor::Empty(x.shape(), DType::kFloat32, Device::kNCNNMeta);
  fprintf(pp, " %s %s", x.name.c_str(), output.name.c_str());

  fprintf(pp, " 0=%d", static_cast<int>(weight.numel()));
  fprintf(pp, " 1=%e", 1e-5f);
  fprintf(pp, " 2=1");
  fprintf(pp, "\n");
  append_data_to_bin_file(cpu::cast_dtype(weight, DType::kFloat32), false);
  append_data_to_bin_file(cpu::cast_dtype(bias, DType::kFloat32), false);
  return output;
}

Tensor groupnorm(const Tensor &x, int num_groups, const Tensor &weight,
                 const Tensor &bias) {
  PRINT_OP_TYPE_AND_NAME("GroupNorm", 1, 1);
  auto output = Tensor::Empty(x.shape(), DType::kFloat32, Device::kNCNNMeta);
  fprintf(pp, " %s %s", x.name.c_str(), output.name.c_str());

  fprintf(pp, " 0=%d", num_groups);
  fprintf(pp, " 1=%d", static_cast<int>(weight.numel()));
  fprintf(pp, " 2=%e", 1e-5f);
  fprintf(pp, " 3=1");
  fprintf(pp, "\n");
  append_data_to_bin_file(cpu::cast_dtype(weight, DType::kFloat32), false);
  append_data_to_bin_file(cpu::cast_dtype(bias, DType::kFloat32), false);
  return output;
}

Tensor batch_matmul(const Tensor &a, const Tensor &b) {
  Shape output_shape = shape::matmul(a.shape(), b.shape());

  Tensor a_meta = a;
  if (a.device() == Device::kCPU) {
    a_meta = MemoryData(a);
  }
  Tensor b_meta = b;
  if (b.device() == Device::kCPU) {
    b_meta = MemoryData(b);
  }
  PRINT_OP_TYPE_AND_NAME("MatMul", 2, 1);
  auto output = Tensor::Empty(output_shape, DType::kFloat32, Device::kNCNNMeta);
  fprintf(pp, " %s %s %s", a_meta.name.c_str(), b_meta.name.c_str(),
          output.name.c_str());
  // transB
  fprintf(pp, " 0=0");
  fprintf(pp, "\n");
  return output;
}

Tensor reshape(const Tensor &x, const Shape &shape) {
  RV_CHECK(x.numel() == num_elements(shape));
  PRINT_OP_TYPE_AND_NAME("Reshape", 1, 1);
  auto output = Tensor::Empty(shape, DType::kFloat32, Device::kNCNNMeta);
  fprintf(pp, " %s %s", x.name.c_str(), output.name.c_str());
  if (shape.size() == 4) {
    fprintf(pp, " 0=%d", (int)shape[3]);
    fprintf(pp, " 1=%d", (int)shape[2]);
    fprintf(pp, " 2=%d", (int)shape[1]);
    fprintf(pp, " 11=%d", (int)shape[0]);
  } else if (shape.size() == 3) {
    fprintf(pp, " 0=%d", (int)shape[2]);
    fprintf(pp, " 1=%d", (int)shape[1]);
    fprintf(pp, " 2=%d", (int)shape[0]);
  } else if (shape.size() == 2) {
    fprintf(pp, " 0=%d", (int)shape[1]);
    fprintf(pp, " 1=%d", (int)shape[0]);
  } else if (shape.size() == 1) {
    fprintf(pp, " 0=%d", (int)shape[0]);
  } else {
    RV_UNIMPLEMENTED();
  }
  fprintf(pp, "\n");
  return output;
}

uint8_t quantize_nf4(float x) {
  RV_CHECK(x <= 1 && x >= -1);
  // It is copied from https://github.com/TimDettmers/bitsandbytes/blob/main/csrc/kernels.cu#L278
  // the values for this tree was generated by test_normal_map_tree
  // in the file tests/test_functional.py
  if(x > 0.03979014977812767f)
    if(x > 0.3893125355243683f) // 1
      if(x > 0.6427869200706482f) // 11
        if(x > 0.8614784181118011f) // 111
          return 0b1111;
        else
          return 0b1110;
      else
        if(x > 0.5016634166240692f) // 110
          return 0b1101;
        else
          return 0b1100;
    else
      if(x > 0.2035212516784668f) // 10
        if(x > 0.2920137718319893f) // 101
          return 0b1011;
        else
          return 0b1010;
      else
        if(x > 0.1202552504837513f) // 100
          return 0b1001;
        else
          return 0b1000;
  else
    if(x > -0.33967943489551544f) // 0
      if(x > -0.13791173323988914f) // 01
        if(x > -0.045525018125772476f) // 011
          return 0b0111;
        else
          return 0b0110;
      else
        if(x > -0.23460740596055984f) // 010
          return 0b0101;
        else
          return 0b0100;
    else
      if(x > -0.6106329262256622f) // 00
        if(x > -0.4599952697753906f) // 001
          return 0b0011;
        else
          return 0b0010;
      else
        if(x > -0.8480964004993439f) // 000
          return 0b0001;
        else
          return 0b0000;
}

// The code here is highly coupled with the kernel implementation.
Tensor gemv_a32w4(const Tensor &a, const Tensor &b) {
  RV_CHECK(b.device() == Device::kCPU);
  const float *const ptr0 = b.data_ptr<float>();
  const int K = b.shape()[0];
  RV_CHECK(K % 2 == 0);
  const int N = b.shape()[1];
  static const int KT = 64;
  // static const int effective_KT = KT / 2;
  Tensor B_int4_t = Tensor::Empty({K / 2, N}, DType::kInt8, Device::kCPU);
  constexpr int kGroupSize = 8;
  const bool double_quant = true;
  RV_CHECK(64 % kGroupSize == 0);
  constexpr int kGroupNum = 64 / kGroupSize;
  // a column --> kGroupNum scales
  Tensor scales_t =
      Tensor::Empty({K * N * kGroupNum / KT}, DType::kInt8, Device::kCPU);
  std::vector<int8_t> scales_vec;
  Tensor dq_scales_t =
      Tensor::Empty({scales_t.numel() / 16}, DType::kFloat16, Device::kCPU);
  std::vector<float16> dq_scales_vec;

  const int kBlockCols = 8;

  std::vector<std::array<float, kBlockCols>> unquanted_scales_buffer;
  int kScaleGroupSize = 16;

  // (2 comes from a int8 has two int4)
  // (K / 2, N)
  // (K / KT, N / kBlockCols, KT / 2, kBlockCols)
  // int4
  for (int a = 0; a < K / KT; a++) {
    uint8_t *ptr = static_cast<uint8_t *>(B_int4_t.data_ptr()) + a * (KT / 2) * N;
    int block_id = a * (N / kBlockCols);
    for (int b = 0; b < N / kBlockCols; b++) {
      std::array<float, KT * kBlockCols> block_data;
      int index = 0;
      // every (64, 1) block in (64, 8) superblock has one or more scales
      for (int c = 0; c < KT; c++) {
        for (int d = 0; d < kBlockCols; d++) {
          int k = a * KT + c;
          int n = b * kBlockCols + d;
          block_data[index++] = ptr0[k * N + n];
        }
      }

      const auto col_scales =
          [&]() -> std::array<std::array<float, kBlockCols>, kGroupNum> {
        std::array<std::array<float, KT>, kBlockCols> col_datas;
        std::array<std::array<float, kBlockCols>, kGroupNum> col_scales;

        for (int i = 0; i < KT * kBlockCols; i++) {
          col_datas[i % kBlockCols][i / kBlockCols] = block_data[i];
        }

        // a column --> two scales/zero_points
        // calculate scale and zero point
        // float[i] = int[i] * scale + zero_point
        // int[i] = (float[i] - zero_point) / scale
        // scale = (max - min) / 255
        for (int col = 0; col < kBlockCols; col++) {
          const auto &col_data = col_datas[col];
          RV_CHECK(col_data.size() == 64);
          for (int group = 0; group < kGroupNum; group++) {
            float scale;
            float zero_point;
            float max = col_data[group * kGroupSize];
            float min = col_data[group * kGroupSize];
            for (int i = group * kGroupSize + 1; i < group * kGroupSize + kGroupSize; i++) {
              if (col_data[i] > max) {
                max = col_data[i];
              }
              if (col_data[i] < min) {
                min = col_data[i];
              }
            }
            if (max == min) {
              scale = 1.f;
            } else {
              scale = std::max(std::abs(max), std::abs(min));
            }
            col_scales[group][col] = scale;
          }
        }

        return col_scales;
      }();

      for (const auto& x : col_scales) {
        unquanted_scales_buffer.push_back(x);
      }
      // NOTE(daquexian): we do not need zero_point because we assume the distribution is 
      // already normal distribution, needed by nf4.
      if (unquanted_scales_buffer.size() == kScaleGroupSize) {
        std::array<float16, kBlockCols> this_dq_scales;
        for (int i = 0; i < kBlockCols; i++) {
          float max = std::abs(unquanted_scales_buffer[0][i]);
          for (int j = 1; j < kScaleGroupSize; j++) {
            if (std::abs(unquanted_scales_buffer[j][i]) > max) {
              max = std::abs(unquanted_scales_buffer[j][i]);
            }
          }
          this_dq_scales[i] = max / 127.f;
          // we maintain nf4 table as int8, [-127, 127], instead of [-1, 1], so we want the scales
          // to be 1/127 of the original scales
          dq_scales_vec.push_back(this_dq_scales[i] / float16(127.f));
        }

        for (int i = 0; i < kScaleGroupSize; i++) {
          for (int j = 0; j < kBlockCols; j++) {
            float dq_scale = this_dq_scales[j];
            float unquanted_scale = unquanted_scales_buffer[i][j];
            int quantized_scale = std::lround(unquanted_scale / dq_scale);
            RV_CHECK(quantized_scale <= 127 && quantized_scale >= -127) << "unquanted_scale = " << unquanted_scale << ", dq_scale = " << dq_scale;
            scales_vec.push_back(quantized_scale);
          }
        }

        unquanted_scales_buffer.clear();
      }

      block_id++;

      const int kSubBlockSize = 16;
      // In block_data (traversed by i) which has (KT, kBlockCols) unpacked int8 elems,
      // a ((16 / kBlockCols), kBlockCols) subblock (traversed by j) and another ((16 / kBlockCols), kBlockCols) subblock under it are packed elementwisely
      // 16 comes from 128 (simd reg bits) / 4 (int4 bits) / 2 (two subblocks)
      for (int i = 0; i < KT / (kSubBlockSize / kBlockCols * 2); i++) {
        for (int j = 0; j < kSubBlockSize; j++) {
          int idx1 = i * kSubBlockSize * 2 + j;
          int idx2 = idx1 + kSubBlockSize;
          RV_CHECK(idx1 < KT * kBlockCols);
          float scale = col_scales[(idx1 / kBlockCols) / kGroupSize][idx1 % kBlockCols];
          block_data[idx1] = block_data[idx1] / scale;
          uint8_t tmp1 = quantize_nf4(block_data[idx1]);
          RV_CHECK(tmp1 >= 0 && tmp1 <= 15);

          RV_CHECK(idx2 < KT * kBlockCols);
          scale = col_scales[(idx2 / kBlockCols) / kGroupSize][idx2 % kBlockCols];
          block_data[idx2] = block_data[idx2] / scale;
          uint8_t tmp2 = quantize_nf4(block_data[idx2]);
          RV_CHECK(tmp2 >= 0 && tmp2 <= 15);

          RV_CHECK((ptr - B_int4_t.data_ptr<uint8_t>()) <
                   B_int4_t.numel());
          *ptr++ = (tmp2 << 4) + tmp1;
        }
      }
    }
  }

  RV_CHECK(unquanted_scales_buffer.empty());
  RV_CHECK(scales_vec.size() == scales_t.numel());
  memcpy(scales_t.data_ptr(), scales_vec.data(), scales_t.numel() * scales_t.elem_size());

  RV_CHECK(dq_scales_vec.size() == dq_scales_t.numel());
  memcpy(dq_scales_t.data_ptr(), dq_scales_vec.data(), dq_scales_t.numel() * dq_scales_t.elem_size());

  PRINT_OP_TYPE_AND_NAME("GemvA32W4", 1, 1);
  append_data_to_bin_file(B_int4_t, true);
  append_data_to_bin_file(scales_t, false);
  append_data_to_bin_file(dq_scales_t, false);
  auto output = Tensor::Empty({N}, DType::kFloat32, Device::kNCNNMeta);
  fprintf(pp, " %s", a.name.c_str());
  fprintf(pp, " %s", output.name.c_str());
  fprintf(pp, " 0=%d", N);
  fprintf(pp, " 1=%d", K);
  fprintf(pp, " 11=%d", kGroupSize);
  fprintf(pp, " 22=%d\n", kScaleGroupSize);
  return output;
}

// The code here is highly coupled with the kernel implementation.
Tensor gemv_a32w8(const Tensor &a, const Tensor &b) {
  RV_CHECK(b.device() == Device::kCPU);
  const float *const ptr0 = b.data_ptr<float>();
  const int K = b.shape()[0];
  const int N = b.shape()[1];
  static const int KT = 64;
  Tensor B_int8_t = Tensor::Empty({K, N}, DType::kInt8, Device::kCPU);
  Tensor scales_t = Tensor::Empty({K * N / KT}, DType::kFloat32, Device::kCPU);
  float *scales = scales_t.data_ptr<float>();
  Tensor zero_points_t =
      Tensor::Empty({K * N / KT}, DType::kFloat32, Device::kCPU);
  float *zero_points = zero_points_t.data_ptr<float>();

  // (K, N)
  // (K / 64, N / 4, 64, 4)
  // int8
  for (int a = 0; a < K / KT; a++) {
    uint8_t *ptr = static_cast<uint8_t *>(B_int8_t.data_ptr()) + a * KT * N;
    int block_id = a * (N / 4);
    for (int b = 0; b < N / 4; b++) {
      std::array<float, KT * 4> block_data;
      int index = 0;
      // every (64, 1) block in (64, 4) superblock has a scale and a zero_point
      for (int c = 0; c < KT; c++) {
        for (int d = 0; d < 4; d++) {
          int k = a * KT + c;
          int n = b * 4 + d;
          block_data[index++] = ptr0[k * N + n];
        }
      }

      const auto [col_scales, col_zeropoints] =
          [&]() -> std::pair<std::array<float, 4>, std::array<float, 4>> {
        std::array<std::array<float, KT>, 4> col_datas;
        std::array<float, 4> col_scales;
        std::array<float, 4> col_zeropoints;

        for (int i = 0; i < KT * 4; i++) {
          col_datas[i % 4][i / 4] = block_data[i];
        }

        // calculate scale and zero point
        // float[i] = int[i] * scale + zero_point
        // int[i] = (float[i] - zero_point) / scale
        // scale = (max - min) / 255
        for (int col = 0; col < 4; col++) {
          const auto &col_data = col_datas[col];
          float scale;
          float zero_point;
          float max = col_data[0];
          float min = col_data[0];
          for (int i = 1; i < static_cast<int>(col_data.size()); i++) {
            if (col_data[i] > max) {
              max = col_data[i];
            }
            if (col_data[i] < min) {
              min = col_data[i];
            }
          }
          // std::cout << "max = " << max << std::endl;
          // std::cout << "min = " << min << std::endl;
          if (max == min) {
            scale = 1.f;
          } else {
            scale = (max - min) / 255.f;
          }
          zero_point = min;
          col_scales[col] = scale;
          col_zeropoints[col] = zero_point;

          scales[block_id * 4 + col] = scale;
          zero_points[block_id * 4 + col] = zero_point;
        }

        return {col_scales, col_zeropoints};
      }();

      // std::cout << "scales[" << block_id << "] = " << scale << std::endl;
      // std::cout << "zero_points[" << block_id << "] = " << zero_point <<
      // std::endl;
      block_id++;

      for (int i = 0; i < KT * 4; i++) {
        // std::cout << "pre quant, col_datas[" << i << "] = " << col_datas[i]
        // << std::endl;
        block_data[i] =
            (block_data[i] - col_zeropoints[i % 4]) / col_scales[i % 4];
        assert(block_data[i] >= 0 && block_data[i] <= 255);
        *ptr++ = std::lround(block_data[i]);
        // std::cout << "col_datas[" << i << "] = " << col_datas[i] <<
        // std::endl; std::cout << "(int)col_datas[" << i << "] = " <<
        // std::lround(col_datas[i]) << std::endl;
      }
    }
  }

  PRINT_OP_TYPE_AND_NAME("GemvA32W8", 1, 1);
  append_data_to_bin_file(B_int8_t, true);
  append_data_to_bin_file(scales_t, false);
  append_data_to_bin_file(zero_points_t, false);
  auto output = Tensor::Empty({N}, DType::kFloat32, Device::kNCNNMeta);
  fprintf(pp, " %s", a.name.c_str());
  fprintf(pp, " %s", output.name.c_str());
  fprintf(pp, " 0=%d", N);
  fprintf(pp, " 1=%d\n", K);
  return output;
}

Tensor gemm(const Tensor &a, const Tensor &b) {
  RV_CHECK(a.device() == Device::kNCNNMeta);
  auto [a_reshape, reshaped] = [&]() -> std::pair<Tensor, bool> {
    if (a.shape().size() == 1) {
      PRINT_OP_TYPE_AND_NAME("Reshape", 1, 1);
      auto output =
          Tensor::Empty({1, a.shape()[0]}, DType::kFloat32, Device::kNCNNMeta);
      fprintf(pp, " %s %s", a.name.c_str(), output.name.c_str());
      fprintf(pp, " 0=0 1=1\n");
      return {output, true};
    }
    return {a, false};
  }();
  int constantM = 0;
  int constantN = 0;
  int constantK = 0;
  RV_CHECK(a_reshape.shape().size() == 2);
  // if (a.device() == Device::kCPU) {
  //   append_data_to_bin_file(a, true);
  //   constantM = a.shape()[0];
  //   constantK = a.shape()[1];
  // }
  RV_CHECK(b.shape().size() == 2);
  if (b.device() == Device::kCPU) {
    append_data_to_bin_file(cpu::cast_dtype(b, DType::kFloat16), true);
    constantK = b.shape()[0];
    constantN = b.shape()[1];
  }
  auto output = Tensor::Empty({a_reshape.shape()[0], b.shape()[1]},
                              DType::kFloat32, Device::kNCNNMeta);
  int input_num = 0;
  if (a_reshape.device() == Device::kNCNNMeta) {
    input_num++;
  }
  if (b.device() == Device::kNCNNMeta) {
    input_num++;
  }
  PRINT_OP_TYPE_AND_NAME("Gemm", input_num, 1);
  if (a_reshape.device() == Device::kNCNNMeta) {
    fprintf(pp, " %s", a_reshape.name.c_str());
  }
  if (b.device() == Device::kNCNNMeta) {
    fprintf(pp, " %s", b.name.c_str());
  }
  fprintf(pp, " %s", output.name.c_str());
  fprintf(pp, " 4=%d 5=%d 7=%d 8=%d 9=%d", a_reshape.device() == Device::kCPU,
          b.device() == Device::kCPU, constantM, constantN, constantK);
  fprintf(pp, "\n");
  if (reshaped) {
    PRINT_OP_TYPE_AND_NAME("Reshape", 1, 1);
    auto squeezed =
        Tensor::Empty({b.shape()[1]}, DType::kFloat32, Device::kNCNNMeta);
    fprintf(pp, " %s %s", output.name.c_str(), squeezed.name.c_str());
    fprintf(pp, " 0=-1\n");
    return squeezed;
  } else {
    return output;
  }
}

Tensor matmul(const Tensor &a, const Tensor &b) {
  if (_weight_dtype == DType::kInt4 && a.shape().size() == 1 &&
      b.shape().size() == 2 && b.device() == Device::kCPU) {
    if(*get_disable_int4()) {
      return gemv_a32w8(a, b);
    } else {
      return gemv_a32w4(a, b);
    }
  } else if (_weight_dtype == DType::kInt8 && a.shape().size() == 1 &&
             b.shape().size() == 2 && b.device() == Device::kCPU) {
    return gemv_a32w8(a, b);
  } else if (a.shape().size() <= 2 && b.shape().size() <= 2) {
    return gemm(a, b);
  } else {
    return batch_matmul(a, b);
  }
}

Tensor MemoryData(const Tensor &x) {
  RV_CHECK(x.device() == Device::kCPU);
  PRINT_OP_TYPE_AND_NAME("MemoryData", 0, 1);
  auto output = Tensor::Empty(x.shape(), DType::kFloat32, Device::kNCNNMeta);
  // use x's name
  output.name = x.name;
  fprintf(pp, " %s", output.name.c_str());
  if (x.shape().size() == 3) {
    fprintf(pp, " 0=%d", (int)x.shape()[2]);
    fprintf(pp, " 1=%d", (int)x.shape()[1]);
    fprintf(pp, " 2=%d", (int)x.shape()[0]);
  } else if (x.shape().size() == 2) {
    fprintf(pp, " 0=%d", (int)x.shape()[1]);
    fprintf(pp, " 1=%d", (int)x.shape()[0]);
  } else if (x.shape().size() == 1) {
    fprintf(pp, " 0=%d", (int)x.shape()[0]);
  } else {
    RV_UNIMPLEMENTED();
  }
  // 21 for load_type arg, 0 means write_tag
  fprintf(pp, " 21=0");
  fprintf(pp, "\n");
  append_data_to_bin_file(x, true);
  return output;
}

std::map<std::string, int> binary_op_ids{{"add", 0},     {"sub", 1},
                                         {"mul", 2},     {"div", 3},
                                         {"maximum", 4}, {"rsub", 7}};

#define BINARYOP(op_type_name)                                                 \
  Tensor op_type_name(const Tensor &x, const Tensor &y) {                      \
    Tensor meta_x = x.device() == Device::kCPU ? MemoryData(x) : x;            \
    Tensor meta_y = y.device() == Device::kCPU ? MemoryData(y) : y;            \
    PRINT_OP_TYPE_AND_NAME("BinaryOp", 2, 1);                                  \
    auto output = Tensor::Empty(shape::broadcast_binary(x.shape(), y.shape()), \
                                DType::kFloat32, Device::kNCNNMeta);           \
    fprintf(pp, " %s", meta_x.name.c_str());                                   \
    fprintf(pp, " %s", meta_y.name.c_str());                                   \
    fprintf(pp, " %s", output.name.c_str());                                   \
    fprintf(pp, " 0=%d", binary_op_ids[STRINGIFY(op_type_name)]);              \
    fprintf(pp, "\n");                                                         \
    return output;                                                             \
  }

BINARYOP(add);
BINARYOP(sub);
BINARYOP(mul);
BINARYOP(div);
BINARYOP(maximum);

Tensor rsub_scalar(float x, const Tensor &y) {
  Tensor meta_y = y.device() == Device::kCPU ? MemoryData(y) : y;
  PRINT_OP_TYPE_AND_NAME("BinaryOp", 1, 1);
  auto output = Tensor::Empty({}, DType::kFloat32, Device::kNCNNMeta);
  fprintf(pp, " %s", meta_y.name.c_str());
  fprintf(pp, " %s", output.name.c_str());
  fprintf(pp, " 0=%d", binary_op_ids["rsub"]);
  fprintf(pp, " 1=1");
  fprintf(pp, " 2=%e", x);
  fprintf(pp, "\n");
  return output;
}

Tensor exp(const Tensor &x) {
  PRINT_OP_TYPE_AND_NAME("Exp", 1, 1);
  auto output = Tensor::Empty(x.shape(), DType::kFloat32, Device::kNCNNMeta);
  fprintf(pp, " %s", x.name.c_str());
  fprintf(pp, " %s", output.name.c_str());
  fprintf(pp, "\n");
  return output;
}

Tensor relu(const Tensor &x) {
  PRINT_OP_TYPE_AND_NAME("ReLU", 1, 1);
  auto output = Tensor::Empty(x.shape(), DType::kFloat32, Device::kNCNNMeta);
  fprintf(pp, " %s", x.name.c_str());
  fprintf(pp, " %s", output.name.c_str());
  fprintf(pp, "\n");
  return output;
}

Tensor sigmoid(const Tensor &x) {
  PRINT_OP_TYPE_AND_NAME("Sigmoid", 1, 1);
  auto output = Tensor::Empty(x.shape(), DType::kFloat32, Device::kNCNNMeta);
  fprintf(pp, " %s", x.name.c_str());
  fprintf(pp, " %s", output.name.c_str());
  fprintf(pp, "\n");
  return output;
}

Tensor silu(const Tensor &x) {
  PRINT_OP_TYPE_AND_NAME("Swish", 1, 1);
  auto output = Tensor::Empty(x.shape(), DType::kFloat32, Device::kNCNNMeta);
  fprintf(pp, " %s", x.name.c_str());
  fprintf(pp, " %s", output.name.c_str());
  fprintf(pp, "\n");
  return output;
}

Tensor mark_as_output(const Tensor &x, const std::string &name) {
  PRINT_OP_TYPE_AND_NAME("Split", 1, 1);
  auto output = Tensor::Empty(x.shape(), DType::kFloat32, Device::kNCNNMeta);
  output.name = name;
  fprintf(pp, " %s", x.name.c_str());
  fprintf(pp, " %s", output.name.c_str());
  fprintf(pp, "\n");
  return output;
}

std::pair<Tensor, Tensor> split2(const Tensor &x) {
  Tensor meta_x = x.device() == Device::kCPU ? MemoryData(x) : x;
  PRINT_OP_TYPE_AND_NAME("Split", 1, 2);
  auto output1 =
      Tensor::Empty(meta_x.shape(), DType::kFloat32, Device::kNCNNMeta);
  auto output2 =
      Tensor::Empty(meta_x.shape(), DType::kFloat32, Device::kNCNNMeta);
  fprintf(pp, " %s", meta_x.name.c_str());
  fprintf(pp, " %s", output1.name.c_str());
  fprintf(pp, " %s", output2.name.c_str());
  fprintf(pp, "\n");
  return {output1, output2};
}

std::tuple<Tensor, Tensor, Tensor> split3(const Tensor &x) {
  Tensor meta_x = x.device() == Device::kCPU ? MemoryData(x) : x;
  PRINT_OP_TYPE_AND_NAME("Split", 1, 3);
  auto output1 = Tensor::Empty(x.shape(), DType::kFloat32, Device::kNCNNMeta);
  auto output2 = Tensor::Empty(x.shape(), DType::kFloat32, Device::kNCNNMeta);
  auto output3 = Tensor::Empty(x.shape(), DType::kFloat32, Device::kNCNNMeta);
  fprintf(pp, " %s", meta_x.name.c_str());
  fprintf(pp, " %s", output1.name.c_str());
  fprintf(pp, " %s", output2.name.c_str());
  fprintf(pp, " %s", output3.name.c_str());
  fprintf(pp, "\n");
  return {output1, output2, output3};
}

std::tuple<Tensor, Tensor, Tensor, Tensor> split4(const Tensor &x) {
  Tensor meta_x = x.device() == Device::kCPU ? MemoryData(x) : x;
  PRINT_OP_TYPE_AND_NAME("Split", 1, 4);
  auto output1 = Tensor::Empty(x.shape(), DType::kFloat32, Device::kNCNNMeta);
  auto output2 = Tensor::Empty(x.shape(), DType::kFloat32, Device::kNCNNMeta);
  auto output3 = Tensor::Empty(x.shape(), DType::kFloat32, Device::kNCNNMeta);
  auto output4 = Tensor::Empty(x.shape(), DType::kFloat32, Device::kNCNNMeta);
  fprintf(pp, " %s", meta_x.name.c_str());
  fprintf(pp, " %s", output1.name.c_str());
  fprintf(pp, " %s", output2.name.c_str());
  fprintf(pp, " %s", output3.name.c_str());
  fprintf(pp, " %s", output4.name.c_str());
  fprintf(pp, "\n");
  return {output1, output2, output3, output4};
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> split5(const Tensor &x) {
  Tensor meta_x = x.device() == Device::kCPU ? MemoryData(x) : x;
  PRINT_OP_TYPE_AND_NAME("Split", 1, 5);
  auto output1 = Tensor::Empty(x.shape(), DType::kFloat32, Device::kNCNNMeta);
  auto output2 = Tensor::Empty(x.shape(), DType::kFloat32, Device::kNCNNMeta);
  auto output3 = Tensor::Empty(x.shape(), DType::kFloat32, Device::kNCNNMeta);
  auto output4 = Tensor::Empty(x.shape(), DType::kFloat32, Device::kNCNNMeta);
  auto output5 = Tensor::Empty(x.shape(), DType::kFloat32, Device::kNCNNMeta);
  fprintf(pp, " %s", meta_x.name.c_str());
  fprintf(pp, " %s", output1.name.c_str());
  fprintf(pp, " %s", output2.name.c_str());
  fprintf(pp, " %s", output3.name.c_str());
  fprintf(pp, " %s", output4.name.c_str());
  fprintf(pp, " %s", output5.name.c_str());
  fprintf(pp, "\n");
  return {output1, output2, output3, output4, output5};
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
att(const Tensor &x, const Tensor &sx, const Tensor &aa, const Tensor &bb,
    const Tensor &pp, const Tensor &ln_w, const Tensor &ln_b,
    const Tensor &k_mix, const Tensor &v_mix, const Tensor &r_mix,
    const Tensor &t_decay, const Tensor &t_first, const Tensor &kw,
    const Tensor &vw, const Tensor &rw, const Tensor &ow) {
  auto [x_s1, x_s2] = split2(x);
  auto xx = layernorm(x_s1, ln_w, ln_b);
  auto [xx_s1, xx_s2, xx_s3, xx_s4] = split4(xx);
  // auto [kx, vx, rx] = time_mix()
  auto [sx_s1, sx_s2, sx_s3] = split3(sx);
  auto [k_mix_s1, k_mix_s2] = split2(k_mix);
  auto [v_mix_s1, v_mix_s2] = split2(v_mix);
  auto [r_mix_s1, r_mix_s2] = split2(r_mix);
  auto kx = xx_s1 * k_mix_s1 + sx_s1 * (1 - k_mix_s2);
  auto vx = xx_s2 * v_mix_s1 + sx_s2 * (1 - v_mix_s2);
  auto rx = xx_s3 * r_mix_s1 + sx_s3 * (1 - r_mix_s2);

  auto r = sigmoid(matmul(rx, rw));
  auto k = matmul(kx, kw);
  auto [k_s1, k_s2, k_s3] = split3(k);
  auto v = matmul(vx, vw);
  auto [v_s1, v_s2] = split2(v);

  auto ww = t_first + k_s1;
  auto [ww_s1, ww_s2] = split2(ww);
  auto [pp_s1, pp_s2, pp_s3] = split3(pp);
  auto p = maximum(pp_s1, ww_s1);
  auto [p_s1, p_s2] = split2(p);
  auto e1 = exp(pp_s2 - p_s1);
  auto [e1_s1, e1_s2] = split2(e1);
  auto e2 = exp(ww_s2 - p_s2);
  auto [e2_s1, e2_s2] = split2(e2);
  auto [aa_s1, aa_s2] = split2(aa);
  auto [bb_s1, bb_s2] = split2(bb);
  auto wkv = ((e1_s1 * aa_s1 + e2_s1 * v_s1) / (e1_s2 * bb_s1 + e2_s2));
  // if (wkv->dtype() == DType::kFloat16) ...
  auto ww2 = t_decay + pp_s3;
  auto [ww2_s1, ww2_s2] = split2(ww2);
  auto p2 = maximum(ww2_s1, k_s2);
  auto [p2_s1, p2_s2, p2_s3] = split3(p2);
  auto e1n = exp(ww2_s2 - p2_s1);
  auto [e1n_s1, e1n_s2] = split2(e1n);
  auto e2n = exp(k_s3 - p2_s2);
  auto [e2n_s1, e2n_s2] = split2(e2n);

  auto out = matmul(r * wkv, ow);
  return {x_s2 + out, xx_s4, e1n_s1 * aa_s2 + e2n_s1 * v_s2,
          e1n_s2 * bb_s2 + e2n_s2, p2_s3};
}

KernelRegister att_reg("att", Device::kNCNNMeta, att);

std::tuple<Tensor, Tensor, Tensor>
att_one_v5(const Tensor &x, const Tensor &sx, const Tensor &s,
           const Tensor &ln_w, const Tensor &ln_b, const Tensor &lx_w,
           const Tensor &lx_b, const Tensor &k_mix, const Tensor &v_mix,
           const Tensor &r_mix, const Tensor &t_decay, const Tensor &t_first,
           const Tensor &kw, const Tensor &vw, const Tensor &rw,
           const Tensor &ow) {

  auto [x_s1, x_s2] = split2(x);
  auto xx = layernorm(x_s1, ln_w, ln_b);
  // auto [kx, vx, rx] = time_mix()
  auto [xx_s1, xx_s2, xx_s3, xx_s4] = split4(xx);
  auto [sx_s1, sx_s2, sx_s3] = split3(sx);
  auto [k_mix_s1, k_mix_s2] = split2(k_mix);
  auto [v_mix_s1, v_mix_s2] = split2(v_mix);
  auto [r_mix_s1, r_mix_s2] = split2(r_mix);
  auto kx = xx_s1 * k_mix_s1 + sx_s1 * (1 - k_mix_s2);
  auto vx = xx_s2 * v_mix_s1 + sx_s2 * (1 - v_mix_s2);
  auto rx = xx_s3 * r_mix_s1 + sx_s3 * (1 - r_mix_s2);

  auto H = t_decay.size(0);
  auto S = x.size(x.shape().size() - 1) / H;

  auto r = matmul(rx, rw).view({H, 1, S});
  auto k = matmul(kx, kw).view({H, S, 1});
  auto v = matmul(vx, vw).view({H, 1, S});

  auto a = matmul(k, v);
  auto [a_s1, a_s2] = split2(a);
  auto [s_s1, s_s2] = split2(s);
  auto out = matmul(r, a_s1 * t_first + s_s1);
  auto decayed_s = a_s2 + s_s2 * t_decay;

  out = out.flatten();
  // NOTE: ncnn groupnorm is different from pytorch groupnorm, so we use 1d
  // input here
  out = groupnorm(out, static_cast<int>(H), lx_w, lx_b).flatten();
  out = matmul(out, ow);

  return {x_s2 + out, xx_s4, decayed_s};
}

KernelRegister att_v5_reg("att_one_v5", Device::kNCNNMeta, att_one_v5);

std::tuple<Tensor, Tensor, Tensor>
att_one_v5_1(const Tensor &x, const Tensor &sx, const Tensor &s,
             const Tensor &ln_w, const Tensor &ln_b, const Tensor &lx_w,
             const Tensor &lx_b, const Tensor &k_mix, const Tensor &v_mix,
             const Tensor &r_mix, const Tensor &g_mix, const Tensor &t_decay,
             const Tensor &t_first, const Tensor &kw, const Tensor &vw,
             const Tensor &rw, const Tensor &gw, const Tensor &ow) {

  auto [x_s1, x_s2] = split2(x);
  auto xx = layernorm(x_s1, ln_w, ln_b);
  // auto [kx, vx, rx] = time_mix()
  auto [xx_s1, xx_s2, xx_s3, xx_s4, xx_s5] = split5(xx);
  auto [sx_s1, sx_s2, sx_s3, sx_s4] = split4(sx);
  auto [k_mix_s1, k_mix_s2] = split2(k_mix);
  auto [v_mix_s1, v_mix_s2] = split2(v_mix);
  auto [r_mix_s1, r_mix_s2] = split2(r_mix);
  auto [g_mix_s1, g_mix_s2] = split2(g_mix);
  auto kx = xx_s1 * k_mix_s1 + sx_s1 * (1 - k_mix_s2);
  auto vx = xx_s2 * v_mix_s1 + sx_s2 * (1 - v_mix_s2);
  auto rx = xx_s3 * r_mix_s1 + sx_s3 * (1 - r_mix_s2);
  auto gx = xx_s4 * g_mix_s1 + sx_s4 * (1 - g_mix_s2);

  auto H = t_decay.size(0);
  auto S = x.size(x.shape().size() - 1) / H;

  auto r = matmul(rx, rw).view({H, 1, S});
  auto k = matmul(kx, kw).view({H, S, 1});
  auto v = matmul(vx, vw).view({H, 1, S});
  auto g = silu(matmul(gx, gw));

  auto a = matmul(k, v);
  auto [a_s1, a_s2] = split2(a);
  auto [s_s1, s_s2] = split2(s);
  auto out = matmul(r, a_s1 * t_first + s_s1);
  auto decayed_s = a_s2 + s_s2 * t_decay;

  out = out.flatten();
  // NOTE: ncnn groupnorm is different from pytorch groupnorm, so we use 1d
  // input here
  out = groupnorm(out, static_cast<int>(H), lx_w, lx_b).flatten();
  out = out * g;
  out = matmul(out, ow);

  return {x_s2 + out, xx_s5, decayed_s};
}

KernelRegister att_v5_1_reg("att_one_v5_1", Device::kNCNNMeta, att_one_v5_1);

std::tuple<Tensor, Tensor> ffn(const Tensor &x, const Tensor &sx,
                               const Tensor &ln_w, const Tensor &ln_b,
                               const Tensor &k_mix, const Tensor &r_mix,
                               const Tensor &kw, const Tensor &vw,
                               const Tensor &rw) {
  auto [x_s1, x_s2] = split2(x);
  auto xx = layernorm(x_s1, ln_w, ln_b);
  auto [xx_s1, xx_s2, xx_s3] = split3(xx);
  auto [sx_s1, sx_s2] = split2(sx);
  auto [k_mix_s1, k_mix_s2] = split2(k_mix);
  auto [r_mix_s1, r_mix_s2] = split2(r_mix);
  // auto [kx, rx] = channel_mix(xx, sx, k_mix, r_mix);
  auto kx = xx_s1 * k_mix_s1 + sx_s1 * (1 - k_mix_s2);
  auto rx = xx_s2 * r_mix_s1 + sx_s2 * (1 - r_mix_s2);

  auto r = sigmoid(matmul(rx, rw));
  auto vx = relu(matmul(kx, kw));
  vx = vx * vx;
  auto out = r * matmul(vx, vw);
  return {x_s2 + out, xx_s3};
}

KernelRegister ffn_reg("ffn", Device::kNCNNMeta, ffn);

KernelRegister allocator_reg("allocator", Device::kNCNNMeta, null_allocator);

KernelRegister layernorm_reg("layernorm", Device::kNCNNMeta, layernorm);
KernelRegister groupnorm_reg("groupnorm", Device::kNCNNMeta, groupnorm);
KernelRegister matmul_reg("matmul", Device::kNCNNMeta, matmul);
KernelRegister add_reg("add", Device::kNCNNMeta, add);
KernelRegister sub_reg("sub", Device::kNCNNMeta, sub);
KernelRegister mul_reg("mul", Device::kNCNNMeta, mul);
KernelRegister div_reg("div", Device::kNCNNMeta, div);
KernelRegister maximum_reg("maximum", Device::kNCNNMeta, maximum);
KernelRegister rsub_reg("rsub_scalar", Device::kNCNNMeta, rsub_scalar);
KernelRegister exp_reg("exp", Device::kNCNNMeta, exp);
KernelRegister relu_reg("relu", Device::kNCNNMeta, relu);
KernelRegister sigmoid_reg("sigmoid", Device::kNCNNMeta, sigmoid);
KernelRegister reshape_reg("reshape", Device::kNCNNMeta, reshape);
KernelRegister mark_as_output_reg("mark_as_output", Device::kNCNNMeta,
                                  mark_as_output);

} // namespace ncnnmeta
} // namespace rwkv
