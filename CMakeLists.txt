cmake_minimum_required(VERSION 3.25)
project(faster-rwkv CXX)

set(CMAKE_CXX_STANDARD 17)
include(FetchContent)

option(FR_ENABLE_CUDA "Enable CUDA" OFF)
# ONNX support is a WIP
option(FR_ENABLE_ONNX "Enable ONNX" OFF)
option(FR_ENABLE_NCNN "Enable NCNN" ON)
option(FR_BUILD_PYTHON "" OFF)
if (DEFINED ANDROID_NDK)
    option(FR_BUILD_JNI "" ON)
else()
    option(FR_BUILD_JNI "" OFF)
endif()

if(NOT DEFINED CMAKE_POSITION_INDEPENDENT_CODE)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

if (FR_ENABLE_CUDA)
    enable_language(CUDA)
    ## ===================== A solution for compiling error of half caused by low cuda arch detected =======================
    ## find the largest cuda arch number compatible with the device.
    # include(FindCUDA/select_compute_arch)
    # CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
    # string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    # string(REPLACE " " "\;" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
    # string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
    # STRING(FIND "${CUDA_ARCH_LIST}" "\;" LAST_SEMICOLON_INDEX REVERSE)
    # math(EXPR SPLIT_START_INDEX "${LAST_SEMICOLON_INDEX} + 1")
    # STRING(SUBSTRING "${CUDA_ARCH_LIST}" ${SPLIT_START_INDEX} -1 FILTERED_CUDA_ARCH_LIST)
    # set(CMAKE_CUDA_ARCHITECTURES ${FILTERED_CUDA_ARCH_LIST})
    ## ======================================================================================================================
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
        message("CMAKE_CUDA_ARCHITECTURES is not set, using native arch: " ${CMAKE_CUDA_ARCHITECTURES})
    else()
        message("using arch: " ${CMAKE_CUDA_ARCHITECTURES})
    endif()

    set(cuda_kernel_srcs
        kernels/cuda/activations/silu.cu
        kernels/cuda/matmul.cpp
        kernels/cuda/cat.cu
        kernels/cuda/layer_norm.cu
        kernels/cuda/group_norm.cu
        kernels/cuda/cast_dtype.cu
        kernels/cuda/att.cu
        kernels/cuda/att_seq.cu
        kernels/cuda/ffn.cu
        kernels/cuda/ffn_seq.cu
        kernels/cuda/fill.cu
        kernels/cuda/flip.cu
        kernels/cuda/pad.cu
        kernels/cuda/repeat.cu
        kernels/cuda/slice.cu
        kernels/cuda/transpose.cu
        kernels/cuda/element_wise.cu
        kernels/cuda/allocator.cpp
    )
endif()

if (FR_ENABLE_NCNN)
    FetchContent_Declare(
        ncnn
        GIT_REPOSITORY https://github.com/daquexian/ncnn
        # This commit reduces peak memory usage
        # https://github.com/Tencent/ncnn/pull/4966
        GIT_TAG c0daa4fd
        SYSTEM
        )
    option(NCNN_BUILD_BENCHMARK "" OFF)
    option(NCNN_BUILD_TOOLS "" OFF)
    option(NCNN_BUILD_EXAMPLES "" OFF)
    option(NCNN_BUILD_TESTS "" OFF)
    option(NCNN_PIXEL "" OFF)
    option(NCNN_PIXEL_ROTATE "" OFF)
    option(NCNN_PIXEL_AFFINE "" OFF)
    option(NCNN_PIXEL_DRAWING "" OFF)
    option(NCNN_DISABLE_EXCEPTION "" OFF)
    if (FR_BUILD_PYTHON)
        # pybind11 requires RTTI
        option(NCNN_DISABLE_RTTI "" OFF)
    endif()
    if (NOT CMAKE_SYSTEM_NAME STREQUAL "Android" AND NOT CMAKE_CROSSCOMPILING)
        option(NCNN_OPENMP "" OFF)
        option(NCNN_SIMPLEOMP "" ON)
    endif()
    # Termux
    if (CMAKE_SYSTEM_NAME STREQUAL "Android" AND NOT DEFINED ANDROID_NDK)
        option(NCNN_PLATFORM_API "" OFF)
    endif()
    include(disable_unused_ncnn_layers)
    FetchContent_MakeAvailable(ncnn)
    set(ncnn_deps ncnn)
    set(ncnn_kernel_srcs
        kernels/ncnn/init_model.cpp
        kernels/ncnn/model_forward.cpp
    )
endif()

if (FR_ENABLE_ONNX)
    FetchContent_Declare(
        onnx
        GIT_REPOSITORY https://github.com/onnx/onnx
        GIT_TAG v1.14.0
        SYSTEM
    )
    FetchContent_MakeAvailable(onnx)
    # WIP
    # option(onnxruntime_USE_FULL_PROTOBUF "" ON)
    # FetchContent_Declare(
    #     onnxruntime
    #     GIT_REPOSITORY https://github.com/microsoft/onnxruntime
    #     GIT_TAG 5af6279440a0db698b0afbfc3de655400562831f
    #     SOURCE_SUBDIR cmake
    #     SYSTEM
    # )
    # FetchContent_MakeAvailable(onnxruntime)
    set(onnx_kernel_srcs
        kernels/export-ncnn/kernels.cpp
    )
endif()

FetchContent_Declare(
        msgpack
        GIT_REPOSITORY https://github.com/msgpack/msgpack-c
        GIT_TAG cpp-6.1.0
        SYSTEM
        )
option(MSGPACK_USE_BOOST "" OFF)
FetchContent_MakeAvailable(msgpack)

add_library(faster_rwkv_internal
        utils.cpp
        model.cpp 
        tensor.cpp
        tokenizer.cpp
        sampler.cpp
        kernels/shape/shape_inference.cpp
        kernels/cpu/allocator.cpp
        kernels/cpu/fill.cpp
        kernels/cpu/cast_dtype.cpp
        kernels/cpu/repeat.cpp
        kernels/cpu/transpose.cpp
        kernels/cpu/slice.cpp
        kernels/cpu/flip.cpp
        kernels/cpu/pad.cpp
        kernels/default/gather_ops.cpp
        kernels/default/view_ops.cpp
        kernels/cpu/softmax.cpp
        kernels/default/att.cpp
        kernels/default/ffn.cpp
        kernels/default/init_model.cpp
        kernels/default/model_forward.cpp
        kernels/default/model_forward_seq.cpp
        kernels/export-ncnn/kernels.cpp
        ${cuda_kernel_srcs}
        ${ncnn_kernel_srcs}
        ${onnx_kernel_srcs}
        )
target_link_libraries(faster_rwkv_internal PUBLIC msgpack-cxx)
target_link_libraries(faster_rwkv_internal PUBLIC ${ncnn_deps})
target_include_directories(faster_rwkv_internal PUBLIC ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

if (FR_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(faster_rwkv_internal PUBLIC CUDA::cudart CUDA::cublas)
    target_compile_definitions(faster_rwkv_internal PUBLIC FR_ENABLE_CUDA)
endif()
if (FR_ENABLE_NCNN)
    target_compile_definitions(faster_rwkv_internal PUBLIC FR_ENABLE_NCNN)
endif()
if (FR_ENABLE_ONNX)
    target_compile_definitions(faster_rwkv_internal PUBLIC FR_ENABLE_ONNX)
    target_link_libraries(faster_rwkv_internal PRIVATE onnx)
endif()

if (DEFINED ANDROID_NDK)
    target_compile_definitions(faster_rwkv_internal PUBLIC FR_ENABLE_ANDROID_ASSET)
endif()

add_library(faster_rwkv INTERFACE)
target_link_libraries(faster_rwkv INTERFACE "$<LINK_LIBRARY:WHOLE_ARCHIVE,faster_rwkv_internal>")

add_executable(export_ncnn export_ncnn.cpp)
target_link_libraries(export_ncnn faster_rwkv)

option(BENCHMARK_ENABLE_TESTING "Enable testing of the benchmark library." OFF)
option(BENCHMARK_ENABLE_GTEST_TESTS "Enable building the unit tests which depend on gtest" OFF)
FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark
        GIT_TAG v1.8.2
        SYSTEM
        )
FetchContent_MakeAvailable(benchmark)
add_executable(bench_model bench_model.cpp)
target_link_libraries(bench_model benchmark::benchmark faster_rwkv)

add_executable(chat chat.cpp)
target_link_libraries(chat faster_rwkv)

add_executable(abc_music abc_music.cpp)
target_link_libraries(abc_music faster_rwkv)

add_executable(midi_music midi_music.cpp)
target_link_libraries(midi_music faster_rwkv)

if (FR_ENABLE_ONNX)
    add_executable(export_onnx export_onnx.cpp)
    target_link_libraries(export_onnx faster_rwkv)
endif()

if (FR_BUILD_JNI)
    add_subdirectory(aar)
endif()

if (FR_BUILD_PYTHON)
    add_subdirectory(python)
endif()

add_subdirectory(tools)
add_subdirectory(tests)
