#pragma once

#include "ptz/algorithms.hpp"
#include "ptz/interfaces.hpp"
#include <mutex>

namespace ptz {

class LatestTargetState final : public ITargetStateStore {
public:
  void publish(TargetState state) override;
  [[nodiscard]] TargetState latest() const override;
private:
  mutable std::mutex mutex_;
  TargetState state_{};
};

class TrajectoryAxis {
public:
  float update(float desired,float dt,float max_accel,float max_jerk);
  void reset();
private:
  float velocity_{},acceleration_{};
};

class VisualPTZController final : public IPTZController {
public:
  VisualPTZController(ITargetStateStore& store,IPTZBackend& backend,ControllerConfig config);
  void run(std::stop_token stop) override;
  PTZCommand step(TimePoint now) override;
private:
  PTZCommand make_stop(TimePoint now,std::string reason);
  bool should_send(const PTZCommand& command,TimePoint now) const;
  ITargetStateStore& store_; IPTZBackend& backend_; ControllerConfig cfg_;
  FovCalibration fov_; OneEuroFilter pan_filter_,tilt_filter_;
  TrajectoryAxis pan_axis_,tilt_axis_,zoom_axis_;
  TimePoint last_step_{},last_send_{},last_zoom_step_{};
  PTZCommand last_command_{}; std::uint64_t sequence_{};
  float integral_pan_{},integral_tilt_{},previous_pan_error_{},previous_tilt_error_{};
  bool deadzone_hold_{};
};

} // namespace ptz
