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

3. Export ncnn model by `./export_ncnn <input_faster_rwkv_model_path> <output_path_prefix>`. You can download pre-built `export_ncnn` from [Releases](https://github.com/daquexian/faster-rwkv/releases) if you are a Linux users, or build it by yourself.

#### Build

##### Android App Development

Download the pre-built [Android AAR library](https://developer.android.com/studio/projects/android-library#psd-add-aar-jar-dependency) from [Releases](https://github.com/daquexian/faster-rwkv/releases), or run the `aar/build_aar.sh` to build it by yourself.

##### Android C++ Development

For the path of Android NDK and toolchain file, please refer to Android NDK docs.

```
mkdir build
cd build
cmake -DFR_ENABLE_NCNN=ON -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DANDROID_NDK=xxxx -DCMAKE_TOOLCHAIN_FILE=xxxx -DCMAKE_BUILD_TYPE=Release -GNinja ..
ninja
```

#### Run in Termux (Ignore it if you are an app developer)

1. Copy `chat` into the Android phone (by using adb or Termux).

2. Copy the [tokenizer_model](https://github.com/daquexian/faster-rwkv/blob/master/tokenizer_model) and the ncnn models (.param, .bin and .config) into the Android phone (by using adb or Termux).

3. Run ``./chat tokenizer_model ncnn_models_basename "ncnn fp16"`` in adb shell or Termux, for example, if the ncnn models are named `rwkv-4-chntuned-1.5b.param`, `rwkv-4-chntuned-1.5b.bin` and `rwkv-4-chntuned-1.5b.config`, the command should be ``./chat tokenizer_model rwkv-4-chntuned-1.5b "ncnn fp16"``.

#### Requirements

* Android System >= 9.0

* RAM >= 4GB (for 1.5B model)

* No hard requirement for CPU. More powerful = faster.

### Android Demo

Run one of the following commands in Termux to download prebuilt executables and models automatically. The download script supports continuely downloading partially downloaded files, so feel free to Ctrl-C and restart it if the speed is too slow.

Executables, 1.5B CHNtuned int8 model, 1.5B CHNtuned int4 model and 0.1B world int8 model:

```
curl -L -s https://raw.githubusercontent.com/daquexian/faster-rwkv/master/download_binaries_and_models_termux.sh | bash -s 3
```

Executables, 1.5B CHNtuned int4 model and 0.1B world int8 model:

```
curl -L -s https://raw.githubusercontent.com/daquexian/faster-rwkv/master/download_binaries_and_models_termux.sh | bash -s 2
```

Executables and 0.1B world int8 model:

```
curl -L -s https://raw.githubusercontent.com/daquexian/faster-rwkv/master/download_binaries_and_models_termux.sh | bash -s 1
```

Executables only:

```
curl -L -s https://raw.githubusercontent.com/daquexian/faster-rwkv/master/download_binaries_and_models_termux.sh | bash -s 0
```

### Export ONNX

1. Install `rwkv2onnx` python package by `pip install rwkv2onnx`.

2. Clone https://github.com/BlinkDL/ChatRWKV

3. Run `rwkv2onnx <input path> <output path> <ChatRWKV path>`. For example, `rwkv2onnx ~/RWKV-5-World-0.1B-v1-20230803-ctx4096.pth ~/RWKV-5-0.1B.onnx ~/ChatRWKV`

### TODO

- [x] JNI
- [x] v5 models support (models are published at https://huggingface.co/daquexian/fr-models/tree/main)
- [x] ABC music models support (models are published at https://huggingface.co/daquexian/fr-models/tree/main)
- [x] CI
- [x] ARM NEON int8 (~2x speedup compared to fp16)
- [x] ARM NEON int4 (>2x speedup compared to fp16)
- [x] MIDI music models support
- [x] custom initial state support
- [x] export ONNX
- [ ] seq mode
    - [x] CUDA
    - [ ] Others
- [ ] Raven models support
- [ ] more backends..
- [ ] simplify model convertion
