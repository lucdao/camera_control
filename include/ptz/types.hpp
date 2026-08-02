#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ptz {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

struct Point2f { float x{}; float y{}; };
struct Box {
  float x1{}, y1{}, x2{}, y2{};
  [[nodiscard]] float width() const { return x2 - x1; }
  [[nodiscard]] float height() const { return y2 - y1; }
  [[nodiscard]] float area() const;
  [[nodiscard]] Point2f center() const { return {(x1+x2)*0.5F, (y1+y2)*0.5F}; }
};

struct FramePacket {
  std::uint64_t sequence{};
  int width{}, height{};
  TimePoint captured_at{};
  // BGR host fallback. Jetson adapters may set cuda_device_ptr and leave bgr empty.
  std::vector<std::uint8_t> bgr;
  std::uintptr_t cuda_device_ptr{};
  std::size_t cuda_pitch{};
  enum class PixelFormat { BGR, RGBA } pixel_format{PixelFormat::BGR};
  // Keeps the decoder surface / CUDA-EGL registration alive through inference.
  std::shared_ptr<void> owner;
};

struct FaceObservation {
  Box bbox;
  float detection_score{};
  std::array<Point2f, 5> landmarks{};
  std::vector<float> embedding;
};

struct PersonObservation {
  Box bbox;
  float detection_score{};
  std::array<Point2f, 17> keypoints{};
  std::array<float, 17> keypoint_scores{};
  std::vector<float> body_embedding;
  std::uint64_t track_id{};
};

struct IdentityEvidence {
  std::string identity;
  float best_score{};
  float second_score{};
  bool confirmed{};
};

enum class MeasurementSource { None, Face, Pose, BodyTrack, Coast };

struct TargetState {
  std::uint64_t sequence{};
  bool valid{};
  std::string identity;
  std::uint64_t track_id{};
  Point2f aim_px{};
  Point2f velocity_px_s{};
  float scale{};                 // face-height / frame-height
  float scale_velocity_s{};
  float confidence{};
  std::array<float, 36> covariance{}; // row-major 6x6
  MeasurementSource source{MeasurementSource::None};
  int frame_width{}, frame_height{};
  TimePoint captured_at{};
  TimePoint published_at{};
};

struct PTZPosition {
  float pan{}, tilt{}, zoom{};
  bool valid{};
  bool motion_known{},pan_tilt_moving{},zoom_moving{};
  TimePoint sampled_at{};
};

struct PTZCommand {
  std::uint64_t sequence{};
  float pan_velocity{}, tilt_velocity{}, zoom_velocity{};
  std::string reason;
  TimePoint created_at{};
  TimePoint source_captured_at{};
};

} // namespace ptz
