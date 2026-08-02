#pragma once
#include "ptz/interfaces.hpp"
#include <filesystem>
#include <memory>

namespace ptz {
class TensorRtInference final:public IInferenceEngine{
public:
  TensorRtInference(const std::filesystem::path& models,const std::filesystem::path& cache);
  ~TensorRtInference()override;
  std::vector<FaceObservation> faces(const FramePacket&)override;
  std::vector<PersonObservation> people(const FramePacket&)override;
  std::vector<float> face_embedding(const FramePacket&,const FaceObservation&)override;
  std::vector<float> body_embedding(const FramePacket&,const PersonObservation&)override;
private:struct Impl;std::unique_ptr<Impl> impl_;
};
}
