#include <fstream>
#include <string>

inline bool file_exists(const std::string &path) {
  std::ifstream file(path);
  return file.good();
}


