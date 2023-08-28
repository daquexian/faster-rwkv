## Faster RWKV

### CUDA

#### Convert Model

1. Generate a ChatRWKV weight file by `v2/convert_model.py` (in ChatRWKV repo) and strategy `cuda fp16`.

2. Generate a faster-rwkv weight file by `tools/convert_weight.py`. For example, `python3 tools/convert_weight.py RWKV-4-World-CHNtuned-1.5B-v1-20230620-ctx4096-converted-fp16.pth rwkv-4-1.5b-chntuned-fp16.fr`.

#### Build

```
mkdir build
cd build
cmake -DFR_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -GNinja ..
ninja
```

#### Run

`./chat tokenizer_file_path weight_file_path "cuda fp16"`

For example, `./chat ../tokenizer_model ../rwkv-4-1.5b-chntuned-fp16.fr "cuda fp16"`

### Android

#### Convert Model

1. Generate a ChatRWKV weight file by `v2/convert_model.py` (in ChatRWKV repo) and strategy `cuda fp32` or `cpu fp32`. Note that though we use fp32 here, the real dtype is determined is the following step.

2. Generate a faster-rwkv weight file by `tools/convert_weight.py`.

3. Export ncnn model by `export_ncnn.cpp`.

#### Build

For the path of Android NDK and toolchain file, please refer to Android NDK docs.

```
mkdir build
cd build
cmake -DFR_ENABLE_NCNN=ON -DNCNN_DISABLE_EXCEPTION=OFF -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DANDROID_NDK=xxxx -DCMAKE_TOOLCHAIN_FILE=xxxx -DCMAKE_BUILD_TYPE=Release -GNinja ..
ninja
```

#### Run

1. Copy `chat`, `libfaster_rwkv.so` into the Android phone (by using adb or Termux).

2. Copy the [tokenizer_model](https://github.com/daquexian/faster-rwkv/blob/master/tokenizer_model) and the ncnn models (.param and .bin) into the Android phone (by using adb or Termux).

3. Run ``LD_LIBRARY_PATH=`pwd` ./chat tokenizer_model ncnn_models_basename "ncnn fp16"`` in adb shell or Termux, for example, if the ncnn models are named `rwkv-4-chntuned-1.5b.param` and `rwkv-4-chntuned-1.5b.bin`, the command should be ``LD_LIBRARY_PATH=`pwd` ./chat tokenizer_model rwkv-4-chntuned-1.5b "ncnn fp16"``.

#### Requirements

* Android System >= 9.0

* RAM >= 4GB (for 1.5B model)

* No hard requirement for CPU. More powerful = faster.

### Android Demo

Run one of the following commands in Termux to download prebuilt executables and models automatically. The download script supports continuely downloading partially downloaded files, so feel free to ctrl-C and restart it if the speed is too slow.

Executables, 1.5B CHNtuned model and 0.1B world model:

```
curl -L -s https://raw.githubusercontent.com/daquexian/faster-rwkv/master/download_binaries_and_models_termux.sh | bash -s 2
```

Executables and 0.1B world model:

```
curl -L -s https://raw.githubusercontent.com/daquexian/faster-rwkv/master/download_binaries_and_models_termux.sh | bash -s 1
```

Executables only:

```
curl -L -s https://raw.githubusercontent.com/daquexian/faster-rwkv/master/download_binaries_and_models_termux.sh | bash -s 0
```

### TODO

- [ ] seq mode
- [ ] v5 models support
- [ ] export ONNX
- [ ] more backends..
- [ ] simplify model convertion
