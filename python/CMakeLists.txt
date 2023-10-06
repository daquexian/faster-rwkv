include(FetchContent)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY
  https://github.com/pybind/pybind11
  GIT_TAG v2.11.1
  SYSTEM
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(fr_python fr_python.cpp)
target_link_libraries(fr_python PRIVATE faster_rwkv)
