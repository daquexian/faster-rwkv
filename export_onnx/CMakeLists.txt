option(FR2ONNX_BUILD_PYTHON "" OFF)

add_executable(export_onnx export_onnx.cpp)
target_link_libraries(export_onnx faster_rwkv)

if (FR2ONNX_BUILD_PYTHON)
    add_subdirectory(python)
endif()
