#include "ptz/replay.hpp"
#include "ptz/controller.hpp"
#include "ptz/gallery.hpp"
#include "ptz/perception.hpp"
#include "ptz/tensorrt_inference.hpp"
#include <opencv2/videoio.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace ptz {
namespace {
struct Truth{bool visible{};float cx{},cy{};};
class NullSource final:public IFrameSource{void open()override{}std::optional<FramePacket>read(std::stop_token)override{return std::nullopt;}void close()noexcept override{}};
class ReplayStore final:public ITargetStateStore{public:void publish(TargetState s)override{state=std::move(s);}TargetState latest()const override{return state;}TargetState state;};
std::map<std::uint64_t,Truth>load_truth(const std::filesystem::path&p){std::ifstream in(p);if(!in)throw std::runtime_error("cannot open ground truth: "+p.string());std::map<std::uint64_t,Truth>out;std::string line;while(std::getline(in,line)){if(line.empty()||line[0]=='#'||line.rfind("frame",0)==0)continue;std::replace(line.begin(),line.end(),',',' ');std::istringstream s(line);std::uint64_t frame;int visible;Truth t;if(s>>frame>>visible>>t.cx>>t.cy){t.visible=visible!=0;out[frame]=t;}}return out;}
double p95(std::vector<double>v){if(v.empty())return 0.;std::sort(v.begin(),v.end());return v[static_cast<std::size_t>(std::ceil(.95*v.size())-1)];}
}
std::string ReplayReport::json()const{std::ostringstream o;o<<std::fixed<<std::setprecision(4)<<"{\"frames\":"<<frames<<",\"annotated_frames\":"<<annotated_frames<<",\"visible_frames\":"<<visible_frames<<",\"valid_hits\":"<<valid_hits<<",\"false_positives\":"<<false_positives<<",\"id_switches\":"<<id_switches<<",\"recall\":"<<recall<<",\"false_positive_rate\":"<<false_positive_rate<<",\"aim_rmse_px\":"<<aim_rmse_px<<",\"processing_p95_ms\":"<<processing_p95_ms<<",\"passed\":"<<(passed?"true":"false")<<"}";return o.str();}
ReplayReport run_replay(const AppConfig&cfg,const std::filesystem::path&video,const std::filesystem::path& gt){
  auto truth=load_truth(gt);cv::VideoCapture capture(video.string());if(!capture.isOpened())throw std::runtime_error("cannot open replay video: "+video.string());TensorRtInference inference(cfg.models_dir,cfg.engine_cache);NullSource source;ReplayStore store;PerceptionPipeline pipeline(source,inference,store,cfg.perception,load_gallery(cfg.gallery_dir));ReplayReport report;std::vector<double>times;double squared_error=0.;std::uint64_t error_count=0,last_track=0;cv::Mat image;
  while(capture.read(image)){++report.frames;FramePacket frame;frame.sequence=report.frames;frame.width=image.cols;frame.height=image.rows;frame.captured_at=Clock::now();if(!image.isContinuous())image=image.clone();frame.bgr.assign(image.datastart,image.dataend);auto started=Clock::now();auto faces=inference.faces(frame);auto people=inference.people(frame);auto state=pipeline.process(frame,std::move(faces),std::move(people),true,true);times.push_back(std::chrono::duration<double,std::milli>(Clock::now()-started).count());auto it=truth.find(report.frames);if(it==truth.end())continue;++report.annotated_frames;if(it->second.visible){++report.visible_frames;if(state.valid){++report.valid_hits;const double dx=state.aim_px.x-it->second.cx,dy=state.aim_px.y-it->second.cy;squared_error+=dx*dx+dy*dy;++error_count;if(last_track&&state.track_id!=last_track)++report.id_switches;last_track=state.track_id;}}else if(state.valid)++report.false_positives;
  }
  const auto invisible=report.annotated_frames-report.visible_frames;report.recall=report.visible_frames?static_cast<double>(report.valid_hits)/report.visible_frames:0.;report.false_positive_rate=invisible?static_cast<double>(report.false_positives)/invisible:0.;report.aim_rmse_px=error_count?std::sqrt(squared_error/error_count):0.;report.processing_p95_ms=p95(times);report.passed=report.recall>=.90&&report.false_positive_rate<=.02&&report.id_switches==0;return report;
}
}
