#pragma once
#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <stop_token>

namespace ptz {
struct RuntimeMetrics{
  std::atomic<bool> perception_healthy{false},control_healthy{false};
  std::atomic<std::uint64_t> frames{0},target_updates{0},commands{0},failsafe_stops{0};
  std::atomic<double> last_capture_to_command_ms{0.0};
  void record_capture_to_command(double milliseconds);
  [[nodiscard]] double capture_to_command_p95()const;
private:
  mutable std::mutex latency_mutex_;
  std::deque<double> latency_window_;
};
class HealthServer{
public:HealthServer(int port,RuntimeMetrics& metrics):port_(port),metrics_(metrics){}
  void run(std::stop_token stop);
private:int port_;RuntimeMetrics& metrics_;
};
}
