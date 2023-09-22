#include <string>

#include <gtest/gtest.h>

#include <utils.h>

inline std::string get_model_dir() {
  if (std::getenv("FR_MODEL_DIR") == nullptr) {
    // skip test in the caller, because GTEST_SKIP() returns
    // current function with a void
    return "";
  }
  return std::string(std::getenv("FR_MODEL_DIR")) + "/";
}

#define TEST_FILE(path)                                                        \
  get_model_dir() + path;                                                      \
  if (!file_exists(get_model_dir() + path)) {                                  \
    GTEST_SKIP() << "\"" << path << "\" not found in \"" << get_model_dir()    \
                 << "\"";                                                      \
  }
