#pragma once
#include <filesystem>
#include <string>

namespace ptz {
class EngineBuilder{
public:
  static void build_all(const std::filesystem::path& models,const std::filesystem::path& cache);
  static std::filesystem::path engine_path(const std::filesystem::path& onnx,const std::filesystem::path& cache);
};
}
