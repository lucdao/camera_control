#pragma once

#include <string>
#include <vector>

namespace ptz {

struct FovPoint { float zoom{}, hfov_deg{}, vfov_deg{}; };

struct ControllerConfig {
  float rate_hz{20.F};
  float target_x{0.5F}, target_y{0.4F};
  float stale_ms{300.F}, coast_ms{250.F};
  float actuation_delay_ms{40.F};
  float deadzone_enter_deg{0.35F}, deadzone_exit_deg{0.55F};
  float one_euro_min_cutoff{1.0F}, one_euro_beta{0.02F};
  float kp{0.60F}, ki{0.0F}, kd{0.05F}, kv{0.10F};
  float max_pan_speed{0.70F}, max_tilt_speed{0.70F};
  float max_accel{2.5F}, max_jerk{15.F};
  float command_change_threshold{0.01F}, heartbeat_ms{500.F};
  bool auto_zoom{true};
  float target_face_height{0.16F}, zoom_deadband{0.15F};
  float max_zoom_speed{0.15F}, max_zoom_accel{0.30F};
  std::vector<FovPoint> fov{{0.F, 60.F, 35.F}, {1.F, 6.F, 3.5F}};
};

struct PerceptionConfig {
  std::string target_identity;
  float face_hz_search{5.F}, face_hz_locked{2.F}, scrfd_hz{10.F}, pose_hz{15.F};
  float face_match_threshold{0.45F}, ambiguity_margin{0.08F};
  int confirmations_required{2}, confirmation_window{3};
  float face_min_score{0.70F}, face_min_pixels{80.F};
  float coast_ms{250.F};
  float body_reid_window_ms{1500.F};
};

struct AppConfig {
  std::string rtsp_uri;
  std::string gallery_dir{"/data/gallery"};
  std::string models_dir{"/models"};
  std::string engine_cache{"/engines"};
  std::string onvif_endpoint;
  std::string onvif_user_env{"PTZ_ONVIF_USER"};
  std::string onvif_password_env{"PTZ_ONVIF_PASSWORD"};
  std::string onvif_profile_token;
  int health_port{8080};
  ControllerConfig control;
  PerceptionConfig perception;
};

AppConfig load_config(const std::string& path);
void validate_config(const AppConfig& cfg);

} // namespace ptz
