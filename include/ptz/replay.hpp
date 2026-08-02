#pragma once
#include "ptz/config.hpp"
#include <cstdint>
#include <filesystem>
#include <string>

namespace ptz {
struct ReplayReport{
  std::uint64_t frames{},annotated_frames{},visible_frames{},valid_hits{},false_positives{},id_switches{};
  double recall{},false_positive_rate{},aim_rmse_px{},processing_p95_ms{};
  bool passed{};
  [[nodiscard]] std::string json()const;
};
ReplayReport run_replay(const AppConfig& config,const std::filesystem::path& video,
                        const std::filesystem::path& ground_truth);
}
