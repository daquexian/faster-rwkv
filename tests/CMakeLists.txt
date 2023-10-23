FetchContent_Declare(
      gtest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.14.0
      SYSTEM
      )
if (FR_MSVC_STATIC_RUNTIME)
    option(gtest_force_shared_crt "" OFF)
else()
    option(gtest_force_shared_crt "" ON)
endif()
FetchContent_MakeAvailable(gtest)
enable_testing()
add_executable(test_model test_model.cpp)
target_link_libraries(test_model gtest_main faster_rwkv)
include(GoogleTest)
if (NOT CMAKE_SYSTEM_NAME STREQUAL "Android" AND NOT CMAKE_CROSSCOMPILING)
    gtest_discover_tests(test_model)
endif()

add_executable(test_tokenizer test_tokenizer.cpp)
target_link_libraries(test_tokenizer gtest_main faster_rwkv)
if (NOT CMAKE_SYSTEM_NAME STREQUAL "Android" AND NOT CMAKE_CROSSCOMPILING)
    gtest_discover_tests(test_tokenizer)
endif()

add_executable(test_sampler test_sampler.cpp)
target_link_libraries(test_sampler gtest_main faster_rwkv)
if (NOT CMAKE_SYSTEM_NAME STREQUAL "Android" AND NOT CMAKE_CROSSCOMPILING)
    gtest_discover_tests(test_sampler)
endif()

add_executable(test_ops test_ops.cpp)
target_link_libraries(test_ops gtest_main faster_rwkv)
if (NOT CMAKE_SYSTEM_NAME STREQUAL "Android" AND NOT CMAKE_CROSSCOMPILING)
    gtest_discover_tests(test_ops)
endif()

add_executable(run_abc_benchmark run_abc_benchmark.cpp)
target_link_libraries(run_abc_benchmark faster_rwkv msgpack-cxx)


set(benchmark_srcs
        benchmark/random.cpp
        benchmark/kernels/transpose.cpp
    )

add_executable(fr_benchmark ${benchmark_srcs})
target_link_libraries(fr_benchmark faster_rwkv benchmark::benchmark)
