#pragma once

#include "ptz/config.hpp"
#include "ptz/types.hpp"
#include <array>
#include <deque>
#include <optional>
#include <string>
#include <unordered_map>

namespace ptz {

float clamp(float value, float low, float high);
float iou(const Box& a, const Box& b);
std::vector<float> l2_normalize(const std::vector<float>& value);
float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b);
Point2f torso_aim(const PersonObservation& person);
std::optional<std::size_t> associate_face_to_person(
    const FaceObservation& face, const std::vector<PersonObservation>& people);

struct GalleryIdentity { std::string name; std::vector<std::vector<float>> templates; };

class IdentityMatcher {
public:
  IdentityMatcher(float threshold, float margin, int required, int window);
  void set_gallery(std::vector<GalleryIdentity> gallery);
  IdentityEvidence evaluate(const std::vector<float>& embedding, const std::string& target);
  void note_miss();
private:
  float threshold_, margin_;
  int required_, window_;
  std::vector<GalleryIdentity> gallery_;
  std::deque<bool> confirmations_;
};

class KalmanTarget {
public:
  void reset(Point2f point, float scale, TimePoint time);
  void predict(TimePoint time);
  void update(Point2f point, float scale, float confidence, TimePoint time);
  [[nodiscard]] bool initialized() const { return initialized_; }
  [[nodiscard]] Point2f point() const { return {x_[0], x_[1]}; }
  [[nodiscard]] Point2f velocity() const { return {x_[2], x_[3]}; }
  [[nodiscard]] float scale() const { return x_[4]; }
  [[nodiscard]] float scale_velocity() const { return x_[5]; }
  [[nodiscard]] const std::array<float, 36>& covariance() const { return p_; }
private:
  bool initialized_{};
  TimePoint last_{};
  std::array<float, 6> x_{};
  std::array<float, 36> p_{};
};

class OneEuroFilter {
public:
  OneEuroFilter(float frequency, float min_cutoff, float beta, float d_cutoff=1.F);
  float filter(float value, TimePoint time);
  void reset();
private:
  float alpha(float cutoff, float dt) const;
  float frequency_, min_cutoff_, beta_, d_cutoff_;
  bool ready_{};
  float previous_{}, derivative_{};
  TimePoint last_{};
};

class FovCalibration {
public:
  explicit FovCalibration(std::vector<FovPoint> points);
  [[nodiscard]] FovPoint at(float zoom) const;
  [[nodiscard]] Point2f pixel_to_angle(Point2f point, int width, int height,
                                       float target_x, float target_y, float zoom) const;
private:
  std::vector<FovPoint> points_;
};

} // namespace ptz
