#include "ptz/algorithms.hpp"
#include "ptz/config.hpp"
#include "ptz/controller.hpp"
#include "ptz/gallery.hpp"
#include "ptz/perception.hpp"
#include "ptz/tracker.hpp"
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <unistd.h>

namespace {
int failures=0;
#define CHECK(x) do{if(!(x)){std::cerr<<__FILE__<<":"<<__LINE__<<" CHECK failed: " #x "\n";++failures;}}while(false)
#define NEAR(a,b,e) CHECK(std::abs((a)-(b))<=(e))

struct FakeBackend:ptz::IPTZBackend{
  ptz::PTZPosition pos{0,0,.25F,true,false,false,false,ptz::Clock::now()};ptz::PTZCommand last{};std::atomic<int> moves{},stops{};
  void probe()override{}ptz::PTZPosition position()override{return pos;}
  void continuous_move(const ptz::PTZCommand& c)override{last=c;++moves;}void stop()noexcept override{++stops;}
};
struct FakeInference:ptz::IInferenceEngine{
  std::vector<ptz::FaceObservation> faces(const ptz::FramePacket&)override{return {};}
  std::vector<ptz::PersonObservation> people(const ptz::FramePacket&)override{return {};}
  std::vector<float> face_embedding(const ptz::FramePacket&,const ptz::FaceObservation&)override{return {};}
  std::vector<float> body_embedding(const ptz::FramePacket&,const ptz::PersonObservation&)override{return {};}
};
struct FakeSource:ptz::IFrameSource{void open()override{}std::optional<ptz::FramePacket>read(std::stop_token)override{return std::nullopt;}void close()noexcept override{}};

std::vector<float> embedding(int index){std::vector<float> e(512);e[static_cast<std::size_t>(index)]=1.F;return e;}

void test_identity(){
  ptz::IdentityMatcher m(.45F,.08F,2,3);m.set_gallery({{"target",{embedding(1)}},{"other",{embedding(2)}}});
  auto a=m.evaluate(embedding(1),"target");CHECK(!a.confirmed);auto b=m.evaluate(embedding(1),"target");CHECK(b.confirmed);CHECK(b.identity=="target");
  auto wrong=m.evaluate(embedding(2),"target");CHECK(!wrong.confirmed);
}
void test_geometry(){
  ptz::FovCalibration f({{0,60,30},{1,10,5}});auto mid=f.at(.5F);NEAR(mid.hfov_deg,35.F,.001F);
  auto center=f.pixel_to_angle({960,432},1920,1080,.5F,.4F,0);NEAR(center.x,0.F,.001F);NEAR(center.y,0.F,.001F);
  auto right=f.pixel_to_angle({1500,432},1920,1080,.5F,.4F,0);CHECK(right.x>0);auto above=f.pixel_to_angle({960,100},1920,1080,.5F,.4F,0);CHECK(above.y>0);
}
void test_kalman(){
  ptz::KalmanTarget k;auto t=ptz::Clock::now();k.update({100,100},.1F,1,t);k.update({110,100},.1F,1,t+std::chrono::milliseconds(100));CHECK(k.velocity().x>0);auto before=k.point().x;k.predict(t+std::chrono::milliseconds(200));CHECK(k.point().x>before);
}
void test_trajectory(){
  ptz::TrajectoryAxis a;float v=0;for(int i=0;i<10;++i){float next=a.update(1.F,.05F,2.F,10.F);CHECK(std::abs(next-v)<=.101F);v=next;}CHECK(v<=1.F);
}
void test_hungarian_and_bytetrack(){
  auto assignment=ptz::hungarian_min_cost({{1.F,5.F,9.F},{4.F,1.F,8.F},{7.F,6.F,1.F}});CHECK(assignment==std::vector<int>({0,1,2}));
  ptz::ByteTracker tracker;auto t=ptz::Clock::now();ptz::PersonObservation a,b;a.bbox={100,100,200,300};a.detection_score=.9F;b.bbox={500,100,600,300};b.detection_score=.9F;
  auto first=tracker.update({a,b},t);CHECK(first.size()==2);std::uint64_t left=0,right=0;for(const auto&p:first)if(p.bbox.center().x<300)left=p.track_id;else right=p.track_id;CHECK(left&&right&&left!=right);
  // A low-confidence box must rescue an existing track in ByteTrack stage two.
  a.bbox={115,100,215,300};a.detection_score=.25F;b.bbox={485,100,585,300};b.detection_score=.92F;auto second=tracker.update({a,b},t+std::chrono::milliseconds(67));CHECK(second.size()==2);bool rescued=false;for(const auto&p:second)if(p.track_id==left)rescued=true;CHECK(rescued);
  // Predicted motion plus global assignment should preserve both tracks during a
  // close pass. The vertical offset keeps the IoU-only problem observable; at
  // perfectly identical boxes no motion-only tracker can infer identity.
  for(int i=1;i<=5;++i){const float fi=static_cast<float>(i);a.bbox={115.F+45.F*fi,75,215.F+45.F*fi,275};a.detection_score=.9F;b.bbox={485.F-45.F*fi,125,585.F-45.F*fi,325};b.detection_score=.9F;auto out=tracker.update({a,b},t+std::chrono::milliseconds(67*(i+1)));CHECK(out.size()==2);}
  auto states=tracker.states();CHECK(states.size()==2);for(const auto&s:states)CHECK(s.lifecycle==ptz::TrackLifecycle::Tracked);
}
void test_controller(){
  ptz::LatestTargetState store;FakeBackend backend;ptz::ControllerConfig cfg;cfg.auto_zoom=true;ptz::VisualPTZController c(store,backend,cfg);auto now=ptz::Clock::now();
  auto stale=c.step(now);CHECK(stale.pan_velocity==0&&stale.reason=="target_invalid");
  ptz::TargetState target;target.valid=true;target.frame_width=1920;target.frame_height=1080;target.aim_px={1500,300};target.velocity_px_s={50,0};target.scale=.08F;target.captured_at=now-std::chrono::milliseconds(80);target.published_at=now;store.publish(target);
  auto move=c.step(now+std::chrono::milliseconds(50));CHECK(move.pan_velocity>0);CHECK(std::abs(move.pan_velocity)<=cfg.max_pan_speed);CHECK(std::abs(move.zoom_velocity)<=cfg.max_zoom_speed);
  target.published_at=now-std::chrono::seconds(1);store.publish(target);auto stopped=c.step(now);CHECK(stopped.reason=="target_stale"&&stopped.pan_velocity==0);
}
void test_controller_heartbeat_and_failsafe(){
  ptz::LatestTargetState store;FakeBackend backend;ptz::ControllerConfig cfg;cfg.rate_hz=100.F;cfg.heartbeat_ms=30.F;cfg.command_change_threshold=1.F;cfg.stale_ms=60.F;ptz::VisualPTZController controller(store,backend,cfg);auto now=ptz::Clock::now();ptz::TargetState target;target.valid=true;target.frame_width=1920;target.frame_height=1080;target.aim_px={1500,432};target.captured_at=now;target.published_at=now;store.publish(target);
  std::jthread worker([&](std::stop_token token){controller.run(token);});std::this_thread::sleep_for(std::chrono::milliseconds(140));worker.request_stop();worker.join();CHECK(backend.moves.load()>=2);CHECK(backend.stops.load()>=2);
}
void test_perception_lock_and_fallback(){
  FakeSource source;FakeInference inference;ptz::LatestTargetState store;ptz::PerceptionConfig cfg;cfg.target_identity="target";
  ptz::PerceptionPipeline pipe(source,inference,store,cfg,{{"target",{embedding(1)}},{"other",{embedding(2)}}});
  ptz::FramePacket frame;frame.width=1920;frame.height=1080;frame.captured_at=ptz::Clock::now();
  ptz::PersonObservation person;person.bbox={700,100,1200,1000};person.detection_score=.95F;person.body_embedding=embedding(3);person.keypoint_scores[5]=person.keypoint_scores[6]=1;person.keypoints[5]={850,350};person.keypoints[6]={1050,350};
  ptz::FaceObservation face;face.bbox={850,150,1050,350};face.detection_score=.95F;face.embedding=embedding(1);face.landmarks[2]={950,250};
  auto first=pipe.process(frame,{face},{person});CHECK(!first.valid);
  frame.captured_at+=std::chrono::milliseconds(100);auto locked=pipe.process(frame,{face},{person});CHECK(locked.valid);CHECK(locked.source==ptz::MeasurementSource::Face);
  frame.captured_at+=std::chrono::milliseconds(100);auto fallback=pipe.process(frame,{}, {person});CHECK(fallback.valid);CHECK(fallback.source==ptz::MeasurementSource::Pose);
}
void test_gallery(){
  auto root=std::filesystem::temp_directory_path()/("ptz-gallery-test-"+std::to_string(::getpid()));ptz::save_embedding(root,"target","front",embedding(1));auto g=ptz::load_gallery(root);CHECK(g.size()==1&&g[0].templates.size()==1);std::filesystem::remove_all(root);
}
void test_config(){
  auto path=std::filesystem::temp_directory_path()/("ptz-config-test-"+std::to_string(::getpid())+".yaml");
  {std::ofstream out(path);out<<"rtsp:\n  uri: 'rtsp://user:p#ss@camera/live' # comment\nonvif:\n  endpoint: http://camera/onvif/device_service\nperception:\n  target_identity: target\ncontrol:\n  rate_hz: 20\n  fov_calibration:\n    - [0, 60, 35]\n    - [1, 6, 3.5]\nservice:\n  health_port: 8080\n";}
  auto cfg=ptz::load_config(path.string());CHECK(cfg.rtsp_uri=="rtsp://user:p#ss@camera/live");CHECK(cfg.control.fov.size()==2);std::filesystem::remove(path);
}
}

int main(){test_identity();test_geometry();test_kalman();test_trajectory();test_hungarian_and_bytetrack();test_controller();test_controller_heartbeat_and_failsafe();test_perception_lock_and_fallback();test_gallery();test_config();if(failures){std::cerr<<failures<<" test(s) failed\n";return 1;}std::cout<<"all tests passed\n";return 0;}
