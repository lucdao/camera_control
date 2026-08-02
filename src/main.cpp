#include "ptz/config.hpp"
#include "ptz/engine_builder.hpp"
#include "ptz/gallery.hpp"
#include "ptz/runtime.hpp"
#ifdef PTZ_WITH_TENSORRT
#include "ptz/tensorrt_inference.hpp"
#include "ptz/replay.hpp"
#include <opencv2/imgcodecs.hpp>
#endif
#if defined(PTZ_WITH_TENSORRT) && defined(PTZ_WITH_GSTREAMER)
#include "ptz/benchmark.hpp"
#endif
#ifdef PTZ_WITH_ONVIF
#include "ptz/onvif.hpp"
#endif
#include <filesystem>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>

namespace {
std::unordered_map<std::string,std::string> options(int argc,char** argv,int start){std::unordered_map<std::string,std::string> out;for(int i=start;i<argc;++i){std::string key=argv[i];if(key.rfind("--",0)!=0||i+1>=argc)throw std::invalid_argument("expected --option value");out[key.substr(2)]=argv[++i];}return out;}
const std::string& required(const std::unordered_map<std::string,std::string>& o,const std::string& k){auto i=o.find(k);if(i==o.end())throw std::invalid_argument("missing --"+k);return i->second;}
void usage(){std::cout<<"ptz-control 0.2.0\n\n  ptz-control run --config FILE\n  ptz-control validate --config FILE\n  ptz-control build-engines --models DIR --cache DIR\n  ptz-control enroll --identity NAME --images DIR --gallery DIR --models DIR --cache DIR\n  ptz-control replay --config FILE --video CLIP --ground-truth CSV --json REPORT\n  ptz-control benchmark --config FILE --duration 30 --json REPORT\n  ptz-control hil-stop --config FILE --allow-camera-motion yes --json REPORT\n";}
[[maybe_unused]] std::string secret(const std::string& name){const char* value=std::getenv(name.c_str());if(!value||!*value)throw std::runtime_error("required secret environment variable is unset: "+name);return value;}
[[maybe_unused]] void write_report(const std::unordered_map<std::string,std::string>& args,const std::string& report){if(auto it=args.find("json");it!=args.end()){std::ofstream out(it->second);if(!out)throw std::runtime_error("cannot write report: "+it->second);out<<report<<"\n";}std::cout<<report<<"\n";}
}

int main(int argc,char** argv){
  try{
    if(argc<2){usage();return 2;}const std::string command=argv[1];const auto args=options(argc,argv,2);
    if(command=="validate"){auto cfg=ptz::load_config(required(args,"config"));ptz::validate_config(cfg);std::cout<<"configuration valid\n";return 0;}
    if(command=="run")return ptz::run_service(ptz::load_config(required(args,"config")));
    if(command=="replay"){
#ifdef PTZ_WITH_TENSORRT
      auto cfg=ptz::load_config(required(args,"config"));auto report=ptz::run_replay(cfg,required(args,"video"),required(args,"ground-truth"));write_report(args,report.json());return report.passed?0:3;
#else
      throw std::runtime_error("binary was built without TensorRT replay support");
#endif
    }
    if(command=="benchmark"){
#if defined(PTZ_WITH_TENSORRT) && defined(PTZ_WITH_GSTREAMER)
      auto cfg=ptz::load_config(required(args,"config"));double duration=args.contains("duration")?std::stod(args.at("duration")):30.;auto report=ptz::run_benchmark(cfg,duration);write_report(args,report.json());return report.passed?0:3;
#else
      throw std::runtime_error("binary was built without GStreamer/TensorRT benchmark support");
#endif
    }
    if(command=="hil-stop"){
#ifdef PTZ_WITH_ONVIF
      if(!args.contains("allow-camera-motion")||args.at("allow-camera-motion")!="yes")throw std::runtime_error("hil-stop physically moves the camera; pass --allow-camera-motion yes");auto cfg=ptz::load_config(required(args,"config"));ptz::OnvifBackend backend({cfg.onvif_endpoint,secret(cfg.onvif_user_env),secret(cfg.onvif_password_env),cfg.onvif_profile_token,500});backend.probe();ptz::PTZCommand move;move.pan_velocity=.05F;move.reason="hil_stop_measurement";backend.continuous_move(move);std::this_thread::sleep_for(std::chrono::milliseconds(250));const auto stopped_at=ptz::Clock::now();backend.stop();bool idle=false;double stop_ms=2000.;while(std::chrono::duration<double,std::milli>(ptz::Clock::now()-stopped_at).count()<2000.){auto pos=backend.position();if(pos.motion_known&&!pos.pan_tilt_moving&&!pos.zoom_moving){stop_ms=std::chrono::duration<double,std::milli>(ptz::Clock::now()-stopped_at).count();idle=true;break;}std::this_thread::sleep_for(std::chrono::milliseconds(20));}backend.stop();std::ostringstream report;report<<"{\"stop_latency_ms\":"<<stop_ms<<",\"motion_status_observed\":"<<(idle?"true":"false")<<",\"threshold_ms\":350,\"passed\":"<<(idle&&stop_ms<=350.?"true":"false")<<"}";write_report(args,report.str());return idle&&stop_ms<=350.?0:3;
#else
      throw std::runtime_error("binary was built without ONVIF support");
#endif
    }
    if(command=="build-engines"){
#ifdef PTZ_WITH_TENSORRT
      ptz::EngineBuilder::build_all(required(args,"models"),required(args,"cache"));ptz::TensorRtInference validation(required(args,"models"),required(args,"cache"));ptz::FramePacket smoke;smoke.width=640;smoke.height=640;smoke.captured_at=ptz::Clock::now();smoke.bgr.resize(640U*640U*3U);(void)validation.faces(smoke);(void)validation.people(smoke);ptz::FaceObservation face;face.bbox={220,180,420,420};face.landmarks={ptz::Point2f{270,260},ptz::Point2f{370,260},ptz::Point2f{320,310},ptz::Point2f{280,360},ptz::Point2f{360,360}};if(validation.face_embedding(smoke,face).size()!=512)throw std::runtime_error("ArcFace smoke inference failed");ptz::PersonObservation person;person.bbox={180,80,460,620};if(validation.body_embedding(smoke,person).empty())throw std::runtime_error("OSNet smoke inference failed");std::cout<<"TensorRT engines built; tensor contracts and post-processing smoke test passed\n";return 0;
#else
      throw std::runtime_error("binary was built without PTZ_WITH_TENSORRT");
#endif
    }
    if(command=="enroll"){
#ifdef PTZ_WITH_TENSORRT
      const auto& identity=required(args,"identity");const std::filesystem::path images=required(args,"images"),gallery=required(args,"gallery");
      ptz::TensorRtInference inference(required(args,"models"),required(args,"cache"));int accepted=0,rejected=0;
      for(const auto& entry:std::filesystem::directory_iterator(images)){if(!entry.is_regular_file())continue;cv::Mat mat=cv::imread(entry.path().string());if(mat.empty()){++rejected;continue;}ptz::FramePacket frame;frame.width=mat.cols;frame.height=mat.rows;frame.captured_at=ptz::Clock::now();frame.bgr.assign(mat.datastart,mat.dataend);auto faces=inference.faces(frame);
        if(faces.size()!=1||faces[0].detection_score<.70F||faces[0].bbox.height()<80.F){++rejected;continue;}
        const auto& k=faces[0].landmarks;const float eye=std::hypot(k[1].x-k[0].x,k[1].y-k[0].y);const float yaw=eye>1.F?std::abs(k[2].x-(k[0].x+k[1].x)*.5F)/eye:99.F;if(yaw>.45F){++rejected;continue;}
        auto embedding=inference.face_embedding(frame,faces[0]);if(embedding.size()!=512){++rejected;continue;}ptz::save_embedding(gallery,identity,entry.path().stem().string(),embedding);++accepted;
      }
      if(accepted<2)throw std::runtime_error("enrollment needs at least two accepted reference images");
      std::filesystem::create_directories(gallery/identity);std::ofstream meta(gallery/identity/"metadata.json");meta<<"{\"identity\":\""<<identity<<"\",\"model\":\"buffalo_l/w600k_r50\",\"dimension\":512,\"accepted\":"<<accepted<<",\"rejected\":"<<rejected<<"}\n";
      std::cout<<"enrolled "<<identity<<": "<<accepted<<" accepted, "<<rejected<<" rejected\n";return 0;
#else
      throw std::runtime_error("binary was built without PTZ_WITH_TENSORRT");
#endif
    }
    if(command=="--help"||command=="help"){usage();return 0;}throw std::invalid_argument("unknown command: "+command);
  }catch(const std::exception& e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}
}
