## Faster RWKV

### Android

#### Convert Model (Skip it if you are not a devloper)

1. Generate a ChatRWKV weight by `v2/convert_model.py` (in ChatRWKV repo) and strategy `cuda fp32` or `cpu fp32`. Note that though we use fp32 here, the real dtype is determined is the following step.

2. Generate a faster-rwkv weight by `tools/convert_weight.py`.

3. Export ncnn model by `export_ncnn.cpp`.

#### Build (Skip it if you are not a devloper)

For the path of Android NDK and toolchain file, please refer to Android NDK docs.

```
mkdir build
cd build
cmake -DFR_ENABLE_CUDA=OFF -DFR_ENABLE_ONNX=OFF -DNCNN_DISABLE_EXCEPTION=OFF -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DANDROID_NDK=xxxx -DCMAKE_TOOLCHAIN_FILE=xxxx -GNinja ..
ninja
```

#### Run

1. Copy `chat`, `libfaster_rwkv.so` (download from [releases](https://github.com/daquexian/faster-rwkv/releases) or build by yourself) into the Android phone (by using adb or Termux).

2. Copy the [tokenizer_model](https://github.com/daquexian/faster-rwkv/blob/master/tokenizer_model) and the ncnn models (.param and .bin, download from [huggingface](https://huggingface.co/daquexian/fr-models/tree/main) or convert by yourself) into the Android phone (by using adb or Termux).

3. Run ``LD_LIBRARY_PATH=`pwd` ./chat tokenizer_model ncnn_models_basename "ncnn fp16"`` in adb shell or Termux, for example, if the ncnn models are named `rwkv-4-chntuned-1.5b.param` and `rwkv-4-chntuned-1.5b.bin`, the command should be ``LD_LIBRARY_PATH=`pwd` ./chat tokenizer_model rwkv-4-chntuned-1.5b "ncnn fp16"``.

#### Requirements

* Android System >= 9.0

* RAM >= 4GB (for 1.5B model)

* No hard requirement for CPU. More powerful = faster.

### TODO

- [ ] seq mode
- [ ] v5 models support
- [ ] export ONNX
- [ ] more backends..
- [ ] simplify model convertion
