#pragma once
#include "ptz/config.hpp"
#include <string>

namespace ptz {
struct BenchmarkReport{
  double duration_s{},capture_hz{},scrfd_hz{},pose_hz{};
  double scrfd_p95_ms{},pose_p95_ms{},capture_to_state_p95_ms{};
  bool passed{};
  [[nodiscard]] std::string json()const;
};
BenchmarkReport run_benchmark(const AppConfig& config,double duration_s);
}
