include(FetchContent)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY
  https://github.com/pybind/pybind11
  GIT_TAG v2.11.1
  SYSTEM
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(rwkv2onnx_python rwkv2onnx_python.cpp)
target_link_libraries(rwkv2onnx_python PRIVATE faster_rwkv)

