#include "ptz/benchmark.hpp"
#include "ptz/gstreamer_source.hpp"
#include "ptz/tensorrt_inference.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace ptz {
namespace {double percentile(std::vector<double> values,double q){if(values.empty())return 0.;std::sort(values.begin(),values.end());const auto index=static_cast<std::size_t>(std::min<double>(values.size()-1,std::ceil(q*values.size())-1));return values[index];}}
std::string BenchmarkReport::json()const{std::ostringstream o;o<<std::fixed<<std::setprecision(3)<<"{\"duration_s\":"<<duration_s<<",\"capture_hz\":"<<capture_hz<<",\"scrfd_hz\":"<<scrfd_hz<<",\"pose_hz\":"<<pose_hz<<",\"scrfd_p95_ms\":"<<scrfd_p95_ms<<",\"pose_p95_ms\":"<<pose_p95_ms<<",\"capture_to_state_p95_ms\":"<<capture_to_state_p95_ms<<",\"thresholds\":{\"capture_hz\":29,\"scrfd_hz\":8,\"pose_hz\":12,\"latency_ms\":200},\"passed\":"<<(passed?"true":"false")<<"}";return o.str();}
BenchmarkReport run_benchmark(const AppConfig&cfg,double duration_s){
  if(duration_s<=0.)throw std::invalid_argument("benchmark duration must be positive");
  GStreamerRtspSource source(cfg.rtsp_uri);TensorRtInference inference(cfg.models_dir,cfg.engine_cache);source.open();auto started=Clock::now(),last_face=TimePoint{},last_pose=TimePoint{};std::uint64_t frames=0,face_calls=0,pose_calls=0;std::vector<double>face_ms,pose_ms,latency_ms;std::stop_source stop;
  try{while(std::chrono::duration<double>(Clock::now()-started).count()<duration_s){auto frame=source.read(stop.get_token());if(!frame)continue;++frames;auto now=Clock::now();bool ran=false;if(last_face==TimePoint{}||std::chrono::duration<float>(now-last_face).count()>=1.F/cfg.perception.scrfd_hz){auto t=Clock::now();(void)inference.faces(*frame);face_ms.push_back(std::chrono::duration<double,std::milli>(Clock::now()-t).count());last_face=now;++face_calls;ran=true;}if(last_pose==TimePoint{}||std::chrono::duration<float>(now-last_pose).count()>=1.F/cfg.perception.pose_hz){auto t=Clock::now();(void)inference.people(*frame);pose_ms.push_back(std::chrono::duration<double,std::milli>(Clock::now()-t).count());last_pose=now;++pose_calls;ran=true;}if(ran)latency_ms.push_back(std::chrono::duration<double,std::milli>(Clock::now()-frame->captured_at).count());}}
  catch(...){source.close();throw;}source.close();const double elapsed=std::chrono::duration<double>(Clock::now()-started).count();BenchmarkReport report;report.duration_s=elapsed;report.capture_hz=frames/elapsed;report.scrfd_hz=face_calls/elapsed;report.pose_hz=pose_calls/elapsed;report.scrfd_p95_ms=percentile(face_ms,.95);report.pose_p95_ms=percentile(pose_ms,.95);report.capture_to_state_p95_ms=percentile(latency_ms,.95);report.passed=report.capture_hz>=29.&&report.scrfd_hz>=8.&&report.pose_hz>=12.&&report.capture_to_state_p95_ms<=200.;return report;
}
}
