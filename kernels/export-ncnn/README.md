### int8/int4 量化

#### weight only 量化

Faster RWKV （以及市面上大部分大模型）的量化方式属于 weight only 量化，和 CNN 时代流行的量化方式有较大不同：

- 传统量化：将权重和激活值都量化为 int8，kernel 内使用 int8 定点运算。

- weight only 量化：不量化激活值，将权重量化为 int8 或者 int4，计算时反量化为 fp16，以 fp16 精度进行计算，掉点较小，且仍能加速大模型推理（因为 batch=1 时大模型推理是 memory bound 的，对权重的量化缓解了带宽瓶颈）。
  

此外，在 weight only 量化中，为了降低量化损失，会将每 G 个权重分为一组，每组计算出一个 scale（显然 G 越小，量化越精确，但 scale 所占空间会更大，kernel 也会更加复杂）。在 Faster RWKV int8 量化中 G=64，也就是每 64 个权重为一组，在 int4 量化中，G=8，同时 scale 也会以 64 个为一组再次进行 int8 量化（double quantization），降低 scale 所占的空间。

#### NF4

Faster RWKV int4 量化目前使用 NF4 量化方法（https://arxiv.org/abs/2305.14314）。NF4 是一个将 [-1, 1] 范围内的 float 数值映射为 int4 数值的映射表（例如将 0.6427869200706482 ~ 0.8614784181118011 映射为 15）。它是在权重符合正态分布的假设下，量化损失最低的量化方式。

#### 代码位置

- int4
  
  - 量化：入口是 kernels.cpp 的 gemv_a32w4 函数，NF4 映射表在 quantize_nf4 函数中。
    
  - 反量化：反量化和 gemv kernel 融合在一起（以获得推理加速），kernel 在 [daquexian/ncnn](https://github.com/daquexian/ncnn) gemv 分支的 src/layer/arm/gemva32w4_arm_asimdhp.cpp 中
    
- int8
  
  - 量化：入口是 kernels.cpp 的 gemv_a32w8 函数。
    
  - 反量化：[daquexian/ncnn](https://github.com/daquexian/ncnn) gemv 分支的 src/layer/arm/gemva32w8.cpp 中
