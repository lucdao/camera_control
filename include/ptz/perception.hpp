#pragma once

#include "ptz/algorithms.hpp"
#include "ptz/interfaces.hpp"
#include "ptz/tracker.hpp"
#include <deque>
#include <unordered_map>

namespace ptz {

class PerceptionPipeline final : public IPerceptionPipeline {
public:
  PerceptionPipeline(IFrameSource& source,IInferenceEngine& inference,ITargetStateStore& store,
                     PerceptionConfig config,std::vector<GalleryIdentity> gallery);
  void run(std::stop_token stop) override;
  TargetState process(const FramePacket& frame,std::vector<FaceObservation> faces,
                      std::vector<PersonObservation> people,bool allow_identity=true,bool people_fresh=true);
private:
  std::optional<std::size_t> recover_body(const std::vector<PersonObservation>& people)const;
  IFrameSource& source_;IInferenceEngine& inference_;ITargetStateStore& store_;
  PerceptionConfig cfg_;IdentityMatcher matcher_;ByteTracker tracker_;KalmanTarget kalman_;
  std::uint64_t target_track_{};std::vector<float> target_body_embedding_;std::uint64_t sequence_{};
  TimePoint last_face_run_{},last_arcface_run_{},last_pose_run_{},last_measurement_{};bool identity_locked_{};
};

} // namespace ptz
