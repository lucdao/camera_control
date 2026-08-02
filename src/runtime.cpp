#include "ptz/runtime.hpp"
#include "ptz/controller.hpp"
#include "ptz/gallery.hpp"
#include "ptz/health.hpp"
#include "ptz/onvif.hpp"
#include "ptz/perception.hpp"
#ifdef PTZ_WITH_GSTREAMER
#include "ptz/gstreamer_source.hpp"
#endif
#ifdef PTZ_WITH_TENSORRT
#include "ptz/tensorrt_inference.hpp"
#endif
#include <atomic>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <thread>

namespace ptz {
#if defined(PTZ_WITH_GSTREAMER) && defined(PTZ_WITH_TENSORRT) && defined(PTZ_WITH_ONVIF)
namespace {
volatile std::sig_atomic_t shutdown_requested=0;void signal_handler(int){shutdown_requested=1;}
std::string env(const std::string& key){const char* v=std::getenv(key.c_str());if(!v||!*v)throw std::runtime_error("required secret environment variable is unset or empty: "+key);return v;}
class ObservedSource final:public IFrameSource{
public:ObservedSource(IFrameSource& inner,RuntimeMetrics& metrics):inner_(inner),metrics_(metrics){}
 void open()override{metrics_.perception_healthy=false;inner_.open();}
 std::optional<FramePacket> read(std::stop_token s)override{auto frame=inner_.read(s);if(frame){++metrics_.frames;metrics_.perception_healthy=true;}return frame;}
 void close()noexcept override{metrics_.perception_healthy=false;inner_.close();}
private:IFrameSource& inner_;RuntimeMetrics& metrics_;
};
class ObservedStore final:public ITargetStateStore{
public:ObservedStore(ITargetStateStore& inner,RuntimeMetrics& metrics):inner_(inner),metrics_(metrics){}
 void publish(TargetState s)override{++metrics_.target_updates;inner_.publish(std::move(s));}
 TargetState latest()const override{return inner_.latest();}
private:ITargetStateStore& inner_;RuntimeMetrics& metrics_;
};
class ObservedBackend final:public IPTZBackend{
public:ObservedBackend(IPTZBackend& inner,RuntimeMetrics& metrics):inner_(inner),metrics_(metrics){}
 void probe()override{inner_.probe();metrics_.control_healthy=true;}
 PTZPosition position()override{return inner_.position();}
 void continuous_move(const PTZCommand& c)override{inner_.continuous_move(c);++metrics_.commands;if(c.source_captured_at!=TimePoint{})metrics_.record_capture_to_command(std::chrono::duration<double,std::milli>(Clock::now()-c.source_captured_at).count());}
 void stop()noexcept override{inner_.stop();++metrics_.failsafe_stops;}
private:IPTZBackend& inner_;RuntimeMetrics& metrics_;
};
}
#endif

int run_service(const AppConfig& cfg){
#if !defined(PTZ_WITH_GSTREAMER) || !defined(PTZ_WITH_TENSORRT) || !defined(PTZ_WITH_ONVIF)
  (void)cfg;throw std::runtime_error("run requires PTZ_WITH_GSTREAMER, PTZ_WITH_TENSORRT and PTZ_WITH_ONVIF");
#else
  std::signal(SIGINT,signal_handler);std::signal(SIGTERM,signal_handler);
  LatestTargetState raw_store;RuntimeMetrics metrics;HealthServer health(cfg.health_port,metrics);ObservedStore store(raw_store,metrics);
  GStreamerRtspSource raw_source(cfg.rtsp_uri);ObservedSource source(raw_source,metrics);TensorRtInference inference(cfg.models_dir,cfg.engine_cache);
  OnvifBackend raw_backend({cfg.onvif_endpoint,env(cfg.onvif_user_env),env(cfg.onvif_password_env),cfg.onvif_profile_token});ObservedBackend backend(raw_backend,metrics);
  PerceptionPipeline perception(source,inference,store,cfg.perception,load_gallery(cfg.gallery_dir));
  VisualPTZController control(store,backend,cfg.control);
  std::atomic_bool worker_failed{false};
  std::jthread health_thread([&](std::stop_token s){health.run(s);});
  std::jthread perception_thread([&](std::stop_token s){try{perception.run(s);}catch(const std::exception&e){std::cerr<<"{\"level\":\"error\",\"component\":\"perception\",\"message\":\"worker terminated: "<<e.what()<<"\"}\n";store.publish({});raw_backend.stop();worker_failed=true;}metrics.perception_healthy=false;});
  std::jthread control_thread([&](std::stop_token s){try{control.run(s);}catch(const std::exception&e){std::cerr<<"{\"level\":\"error\",\"component\":\"control\",\"message\":\"worker terminated: "<<e.what()<<"\"}\n";raw_backend.stop();worker_failed=true;}metrics.control_healthy=false;});
  while(!shutdown_requested&&!worker_failed.load())std::this_thread::sleep_for(std::chrono::milliseconds(100));
  perception_thread.request_stop();control_thread.request_stop();health_thread.request_stop();raw_backend.stop();return 0;
#endif
}
}
