#pragma once
#include "ptz/interfaces.hpp"
#include <mutex>
#include <string>

namespace ptz {

struct OnvifConfig {std::string device_endpoint,username,password,profile_token;long timeout_ms{500};};

class OnvifBackend final : public IPTZBackend {
public:
  explicit OnvifBackend(OnvifConfig config);~OnvifBackend() override;
  void probe() override;PTZPosition position() override;
  void continuous_move(const PTZCommand& command) override;void stop() noexcept override;
private:
  std::string request(const std::string& endpoint,const std::string& action,const std::string& body);
  std::string envelope(const std::string& body)const;
  OnvifConfig config_;std::string media_endpoint_,ptz_endpoint_,profile_token_;
  void* curl_handle_{};
  mutable std::mutex mutex_;PTZPosition cached_{};TimePoint last_status_{};bool probed_{};
};

} // namespace ptz
