#include "ptz/perception.hpp"
#include <algorithm>
#include <iostream>
#include <thread>
#include <unordered_set>

namespace ptz {
namespace {void interruptible_sleep(std::stop_token stop,float seconds){const auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<float>(seconds));while(!stop.stop_requested()&&Clock::now()<end)std::this_thread::sleep_for(std::chrono::milliseconds(50));}}

PerceptionPipeline::PerceptionPipeline(IFrameSource& s,IInferenceEngine& i,ITargetStateStore& o,
 PerceptionConfig c,std::vector<GalleryIdentity> g)
 :source_(s),inference_(i),store_(o),cfg_(std::move(c)),
 matcher_(cfg_.face_match_threshold,cfg_.ambiguity_margin,cfg_.confirmations_required,cfg_.confirmation_window){matcher_.set_gallery(std::move(g));}

std::optional<std::size_t> PerceptionPipeline::recover_body(const std::vector<PersonObservation>& people)const{
  if(target_body_embedding_.empty())return std::nullopt;
  float best=.55F;std::optional<std::size_t> index;
  for(std::size_t i=0;i<people.size();++i){const float score=cosine_similarity(target_body_embedding_,people[i].body_embedding);if(score>best){best=score;index=i;}}
  return index;
}

TargetState PerceptionPipeline::process(const FramePacket& frame,std::vector<FaceObservation> faces,
                                        std::vector<PersonObservation> people,bool allow_identity,bool people_fresh){
  const auto now=Clock::now();
  if(people_fresh)people=tracker_.update(std::move(people),now);
  else{people.clear();for(const auto& state:tracker_.states())if(state.lifecycle==TrackLifecycle::Tracked)people.push_back(state.observation);}
  std::optional<std::size_t> selected,selected_face;MeasurementSource source=MeasurementSource::None;float confidence=0.F;bool saw_target_candidate=false;
  for(std::size_t face_index=0;face_index<faces.size();++face_index){
    auto& face=faces[face_index];
    if(face.detection_score<cfg_.face_min_score||face.bbox.height()<cfg_.face_min_pixels)continue;
    if(!allow_identity)continue;
    if(face.embedding.empty())face.embedding=inference_.face_embedding(frame,face);
    const auto evidence=matcher_.evaluate(face.embedding,cfg_.target_identity);
    const bool target_candidate=evidence.identity==cfg_.target_identity&&evidence.best_score>=cfg_.face_match_threshold&&
                                evidence.best_score-evidence.second_score>=cfg_.ambiguity_margin;
    saw_target_candidate=saw_target_candidate||target_candidate;
    if(!evidence.confirmed)continue;
    const auto linked=associate_face_to_person(face,people);if(!linked)continue;
    selected=linked;selected_face=face_index;target_track_=people[*linked].track_id;identity_locked_=true;
    if(people[*linked].body_embedding.empty())people[*linked].body_embedding=inference_.body_embedding(frame,people[*linked]);
    if(!people[*linked].body_embedding.empty())target_body_embedding_=l2_normalize(people[*linked].body_embedding);
    source=MeasurementSource::Face;confidence=std::min(face.detection_score,evidence.best_score);break;
  }
  if(allow_identity&&!saw_target_candidate)matcher_.note_miss();
  if(!selected&&target_track_!=0){for(std::size_t i=0;i<people.size();++i)if(people[i].track_id==target_track_){selected=i;break;}}
  const float since_measurement_ms=last_measurement_==TimePoint{}?1e9F:std::chrono::duration<float,std::milli>(now-last_measurement_).count();
  if(!selected&&identity_locked_&&since_measurement_ms<=cfg_.body_reid_window_ms){
    const bool target_track_present=std::any_of(people.begin(),people.end(),[&](const auto& p){return p.track_id==target_track_;});
    if(!target_track_present){for(auto& p:people)if(p.body_embedding.empty())p.body_embedding=inference_.body_embedding(frame,p);selected=recover_body(people);}
  }
  if(selected&&source==MeasurementSource::None){
    for(std::size_t i=0;i<faces.size();++i){const auto linked=associate_face_to_person(faces[i],people);if(linked&&*linked==*selected){selected_face=i;source=MeasurementSource::Face;confidence=std::min(faces[i].detection_score,people[*selected].detection_score);break;}}
    if(source==MeasurementSource::None){source=MeasurementSource::Pose;confidence=people[*selected].detection_score;}
    target_track_=people[*selected].track_id;
  }
  if(selected&&!people_fresh&&source!=MeasurementSource::Face)selected.reset();

  TargetState state;state.sequence=++sequence_;state.identity=cfg_.target_identity;state.track_id=target_track_;
  state.frame_width=frame.width;state.frame_height=frame.height;state.captured_at=frame.captured_at;state.published_at=now;
  if(selected){
    const auto& person=people[*selected];Point2f aim=torso_aim(person);const float frame_h=static_cast<float>(std::max(1,frame.height));float scale=person.bbox.height()/frame_h*.25F;
    if(source==MeasurementSource::Face&&selected_face){
      const auto& face=faces[*selected_face];aim=face.landmarks[2];scale=face.bbox.height()/frame_h;
    }
    kalman_.update(aim,scale,confidence,frame.captured_at);last_measurement_=now;
    state.valid=identity_locked_;state.source=source;state.confidence=confidence;
  }else if(kalman_.initialized()&&std::chrono::duration<float,std::milli>(now-last_measurement_).count()<=cfg_.coast_ms){
    kalman_.predict(frame.captured_at);state.valid=identity_locked_;state.source=MeasurementSource::Coast;state.confidence=.25F;
  }else{state.valid=false;state.source=MeasurementSource::None;target_track_=0;if(since_measurement_ms>cfg_.body_reid_window_ms)identity_locked_=false;}
  if(kalman_.initialized()){state.aim_px=kalman_.point();state.velocity_px_s=kalman_.velocity();state.scale=kalman_.scale();state.scale_velocity_s=kalman_.scale_velocity();state.covariance=kalman_.covariance();}
  return state;
}

void PerceptionPipeline::run(std::stop_token stop){
  float backoff_s=1.F;
  while(!stop.stop_requested()){
    try{
      source_.open();backoff_s=1.F;
      while(!stop.stop_requested()){
        auto frame=source_.read(stop);if(!frame){if(stop.stop_requested())break;throw std::runtime_error("RTSP source ended");}const auto now=Clock::now();
        const float arcface_hz=identity_locked_?cfg_.face_hz_locked:cfg_.face_hz_search;
        std::vector<FaceObservation> faces;std::vector<PersonObservation> people;
        bool allow_identity=false,face_fresh=false,people_fresh=false;
        if(last_face_run_==TimePoint{}||std::chrono::duration<float>(now-last_face_run_).count()>=1.F/cfg_.scrfd_hz){faces=inference_.faces(*frame);last_face_run_=now;face_fresh=true;allow_identity=last_arcface_run_==TimePoint{}||std::chrono::duration<float>(now-last_arcface_run_).count()>=1.F/arcface_hz;if(allow_identity)last_arcface_run_=now;}
        if(last_pose_run_==TimePoint{}||std::chrono::duration<float>(now-last_pose_run_).count()>=1.F/cfg_.pose_hz){people=inference_.people(*frame);last_pose_run_=now;people_fresh=true;}
        if(face_fresh||people_fresh)store_.publish(process(*frame,std::move(faces),std::move(people),allow_identity,people_fresh));
      }
    }catch(const std::exception& e){std::cerr<<"{\"level\":\"warning\",\"component\":\"perception\",\"message\":\"pipeline unavailable; camera stopped: "<<e.what()<<"\"}\n";store_.publish({});source_.close();if(!stop.stop_requested()){interruptible_sleep(stop,backoff_s);backoff_s=std::min(30.F,backoff_s*2.F);}}
  }
  source_.close();store_.publish({});
}

} // namespace ptz
