#pragma once
#include "ptz/interfaces.hpp"
#include <memory>

namespace ptz {
class GStreamerRtspSource final:public IFrameSource{
public:
  explicit GStreamerRtspSource(std::string uri);~GStreamerRtspSource()override;
  void open()override;std::optional<FramePacket> read(std::stop_token stop)override;void close()noexcept override;
private:
  struct Impl;std::unique_ptr<Impl> impl_;std::string uri_;std::uint64_t sequence_{};
};
}
