#pragma once

#include "ptz/types.hpp"
#include <array>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace ptz {

enum class TrackLifecycle { New, Tracked, Lost, Removed };

class BoxKalmanFilter {
public:
  void initiate(const Box& box);
  void predict(float dt);
  void update(const Box& box,float confidence);
  [[nodiscard]] Box box()const;
  [[nodiscard]] bool initialized()const{return initialized_;}
private:
  bool initialized_{};
  std::array<float,8> mean_{};
  std::array<float,64> covariance_{};
};

struct ByteTrackState {
  std::uint64_t id{};
  TrackLifecycle lifecycle{TrackLifecycle::New};
  BoxKalmanFilter filter;
  PersonObservation observation;
  int age{},hits{},lost_frames{};
  TimePoint updated_at{};
};

class ByteTracker {
public:
  struct Config {
    float high_threshold{0.50F};
    float low_threshold{0.10F};
    float new_track_threshold{0.60F};
    float high_match_iou{0.20F};
    float low_match_iou{0.10F};
    float unconfirmed_match_iou{0.30F};
    int track_buffer{30};
  };
  ByteTracker();
  explicit ByteTracker(Config config);
  std::vector<PersonObservation> update(std::vector<PersonObservation> detections,TimePoint now);
  [[nodiscard]] std::optional<PersonObservation> by_id(std::uint64_t id)const;
  [[nodiscard]] std::vector<ByteTrackState> states()const;
private:
  using Match=std::pair<std::size_t,std::size_t>;
  struct Assignment {std::vector<Match> matches;std::vector<std::size_t> unmatched_tracks,unmatched_detections;};
  Assignment associate(const std::vector<std::uint64_t>& track_ids,
                       const std::vector<PersonObservation>& detections,float min_iou)const;
  void activate(PersonObservation detection,TimePoint now,bool confirmed);
  Config cfg_;std::unordered_map<std::uint64_t,ByteTrackState> tracks_;std::uint64_t next_id_{1};TimePoint last_update_{};
};

std::vector<int> hungarian_min_cost(const std::vector<std::vector<float>>& cost);

} // namespace ptz
