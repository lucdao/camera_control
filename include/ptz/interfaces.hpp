#pragma once

#include "ptz/types.hpp"
#include <optional>
#include <stop_token>

namespace ptz {

class IFrameSource {
public:
  virtual ~IFrameSource() = default;
  virtual void open() = 0;
  virtual std::optional<FramePacket> read(std::stop_token stop) = 0;
  virtual void close() noexcept = 0;
};

class IInferenceEngine {
public:
  virtual ~IInferenceEngine() = default;
  virtual std::vector<FaceObservation> faces(const FramePacket&) = 0;
  virtual std::vector<PersonObservation> people(const FramePacket&) = 0;
  virtual std::vector<float> face_embedding(const FramePacket&, const FaceObservation&) = 0;
  virtual std::vector<float> body_embedding(const FramePacket&, const PersonObservation&) = 0;
};

class ITargetStateStore {
public:
  virtual ~ITargetStateStore() = default;
  virtual void publish(TargetState state) = 0;
  [[nodiscard]] virtual TargetState latest() const = 0;
};

class IPTZBackend {
public:
  virtual ~IPTZBackend() = default;
  virtual void probe() = 0;
  [[nodiscard]] virtual PTZPosition position() = 0;
  virtual void continuous_move(const PTZCommand&) = 0;
  virtual void stop() noexcept = 0;
};

class IPerceptionPipeline {
public:
  virtual ~IPerceptionPipeline() = default;
  virtual void run(std::stop_token stop) = 0;
};

class IPTZController {
public:
  virtual ~IPTZController() = default;
  virtual void run(std::stop_token stop) = 0;
  virtual PTZCommand step(TimePoint now) = 0;
};

} // namespace ptz
