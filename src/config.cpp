#include "ptz/config.hpp"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace ptz {
namespace {
std::string trim(std::string s){
  const auto first=s.find_first_not_of(" \t\r\n");if(first==std::string::npos)return {};
  const auto last=s.find_last_not_of(" \t\r\n");s=s.substr(first,last-first+1);
  if(s.size()>=2&&((s.front()=='\"'&&s.back()=='\"')||(s.front()=='\''&&s.back()=='\'')))s=s.substr(1,s.size()-2);
  return s;
}
bool boolean(const std::string& s){return s=="true"||s=="yes"||s=="1";}
float number(const std::unordered_map<std::string,std::string>& m,const std::string& k,float d){auto i=m.find(k);return i==m.end()?d:std::stof(i->second);}
std::string text(const std::unordered_map<std::string,std::string>& m,const std::string& k,std::string d={}){auto i=m.find(k);return i==m.end()?d:i->second;}
}

AppConfig load_config(const std::string& path){
  std::ifstream in(path);if(!in)throw std::runtime_error("cannot open config: "+path);
  std::unordered_map<std::string,std::string> values;std::vector<std::pair<int,std::string>> sections;
  std::vector<FovPoint> fov;std::string raw;
  while(std::getline(in,raw)){
    bool single=false,dbl=false;std::size_t hash=std::string::npos;
    for(std::size_t i=0;i<raw.size();++i){if(raw[i]=='\''&&!dbl)single=!single;else if(raw[i]=='\"'&&!single)dbl=!dbl;else if(raw[i]=='#'&&!single&&!dbl){hash=i;break;}}
    if(hash!=std::string::npos)raw.resize(hash);
    const auto first=raw.find_first_not_of(' ');if(first==std::string::npos)continue;
    const int indent=static_cast<int>(first);std::string line=trim(raw);
    if(line.rfind("- [",0)==0){
      auto close=line.find(']');if(close==std::string::npos)throw std::runtime_error("invalid FOV calibration row");
      std::replace(line.begin(),line.end(),',',' ');std::istringstream ss(line.substr(3,close-3));FovPoint p;
      if(!(ss>>p.zoom>>p.hfov_deg>>p.vfov_deg))throw std::runtime_error("FOV row must be [zoom,hfov,vfov]");
      fov.push_back(p);continue;
    }
    const auto colon=line.find(':');if(colon==std::string::npos)continue;
    const std::string key=trim(line.substr(0,colon)),value=trim(line.substr(colon+1));
    while(!sections.empty()&&sections.back().first>=indent)sections.pop_back();
    std::string prefix;for(const auto& [_,s]:sections){if(!prefix.empty())prefix+='.';prefix+=s;}
    if(value.empty()){sections.emplace_back(indent,key);continue;}
    if(!prefix.empty())prefix+='.';
    values[prefix+key]=value;
  }
  AppConfig c;c.rtsp_uri=text(values,"rtsp.uri");c.gallery_dir=text(values,"models.gallery_dir",c.gallery_dir);
  c.models_dir=text(values,"models.models_dir",c.models_dir);c.engine_cache=text(values,"models.engine_cache",c.engine_cache);
  c.onvif_endpoint=text(values,"onvif.endpoint");c.onvif_user_env=text(values,"onvif.user_env",c.onvif_user_env);
  c.onvif_password_env=text(values,"onvif.password_env",c.onvif_password_env);c.onvif_profile_token=text(values,"onvif.profile_token");
  c.health_port=static_cast<int>(number(values,"service.health_port",static_cast<float>(c.health_port)));
  c.perception.target_identity=text(values,"perception.target_identity");
  c.perception.face_match_threshold=number(values,"perception.face_match_threshold",c.perception.face_match_threshold);
  c.perception.ambiguity_margin=number(values,"perception.ambiguity_margin",c.perception.ambiguity_margin);
  c.perception.scrfd_hz=number(values,"perception.scrfd_hz",c.perception.scrfd_hz);
  c.perception.face_hz_search=number(values,"perception.arcface_hz_search",c.perception.face_hz_search);
  c.perception.face_hz_locked=number(values,"perception.arcface_hz_locked",c.perception.face_hz_locked);
  c.perception.pose_hz=number(values,"perception.pose_hz",c.perception.pose_hz);
  c.perception.confirmations_required=static_cast<int>(number(values,"perception.confirmations_required",static_cast<float>(c.perception.confirmations_required)));
  c.perception.confirmation_window=static_cast<int>(number(values,"perception.confirmation_window",static_cast<float>(c.perception.confirmation_window)));
  c.perception.face_min_score=number(values,"perception.face_min_score",c.perception.face_min_score);
  c.perception.face_min_pixels=number(values,"perception.face_min_pixels",c.perception.face_min_pixels);
  c.perception.coast_ms=number(values,"perception.coast_ms",c.perception.coast_ms);
  c.perception.body_reid_window_ms=number(values,"perception.body_reid_window_ms",c.perception.body_reid_window_ms);
  c.control.rate_hz=number(values,"control.rate_hz",c.control.rate_hz);c.control.stale_ms=number(values,"control.stale_ms",c.control.stale_ms);
  c.control.target_x=number(values,"control.target_x",c.control.target_x);c.control.target_y=number(values,"control.target_y",c.control.target_y);
  c.control.actuation_delay_ms=number(values,"control.actuation_delay_ms",c.control.actuation_delay_ms);
  c.control.deadzone_enter_deg=number(values,"control.deadzone_enter_deg",c.control.deadzone_enter_deg);
  c.control.deadzone_exit_deg=number(values,"control.deadzone_exit_deg",c.control.deadzone_exit_deg);
  c.control.one_euro_min_cutoff=number(values,"control.one_euro_min_cutoff",c.control.one_euro_min_cutoff);
  c.control.one_euro_beta=number(values,"control.one_euro_beta",c.control.one_euro_beta);
  c.control.kp=number(values,"control.kp",c.control.kp);c.control.ki=number(values,"control.ki",c.control.ki);
  c.control.kd=number(values,"control.kd",c.control.kd);c.control.kv=number(values,"control.kv",c.control.kv);
  c.control.max_pan_speed=number(values,"control.max_pan_speed",c.control.max_pan_speed);
  c.control.max_tilt_speed=number(values,"control.max_tilt_speed",c.control.max_tilt_speed);
  c.control.max_accel=number(values,"control.max_accel",c.control.max_accel);c.control.max_jerk=number(values,"control.max_jerk",c.control.max_jerk);
  c.control.command_change_threshold=number(values,"control.command_change_threshold",c.control.command_change_threshold);
  c.control.heartbeat_ms=number(values,"control.heartbeat_ms",c.control.heartbeat_ms);
  c.control.target_face_height=number(values,"control.target_face_height",c.control.target_face_height);
  c.control.zoom_deadband=number(values,"control.zoom_deadband",c.control.zoom_deadband);
  c.control.max_zoom_speed=number(values,"control.max_zoom_speed",c.control.max_zoom_speed);
  c.control.max_zoom_accel=number(values,"control.max_zoom_accel",c.control.max_zoom_accel);
  if(auto i=values.find("control.auto_zoom");i!=values.end())c.control.auto_zoom=boolean(i->second);
  if(!fov.empty())c.control.fov=std::move(fov);
  validate_config(c);return c;
}

void validate_config(const AppConfig& c){
  if(c.rtsp_uri.rfind("rtsp://",0)!=0&&c.rtsp_uri.rfind("rtsps://",0)!=0)throw std::invalid_argument("rtsp.uri must use rtsp:// or rtsps://");
  if(c.onvif_endpoint.rfind("http://",0)!=0&&c.onvif_endpoint.rfind("https://",0)!=0)throw std::invalid_argument("onvif.endpoint must use http(s)://");
  if(c.perception.target_identity.empty())throw std::invalid_argument("perception.target_identity is required");
  if(!std::all_of(c.perception.target_identity.begin(),c.perception.target_identity.end(),[](unsigned char ch){return std::isalnum(ch)||ch=='_'||ch=='-';}))throw std::invalid_argument("target_identity may contain only letters, digits, '_' and '-'");
  if(c.control.rate_hz<1.F||c.control.rate_hz>100.F)throw std::invalid_argument("control.rate_hz outside 1..100");
  if(c.control.stale_ms<=0.F||c.control.stale_ms>2000.F)throw std::invalid_argument("control.stale_ms outside 0..2000");
  if(c.control.fov.empty())throw std::invalid_argument("control.fov_calibration is required");
  if(c.health_port<1||c.health_port>65535)throw std::invalid_argument("service.health_port outside 1..65535");
  if(c.perception.confirmations_required<1||c.perception.confirmation_window<c.perception.confirmations_required)throw std::invalid_argument("invalid identity confirmation window");
  if(c.perception.face_match_threshold<0.F||c.perception.face_match_threshold>1.F)throw std::invalid_argument("face match threshold outside 0..1");
  if(c.control.deadzone_enter_deg<0.F||c.control.deadzone_exit_deg<c.control.deadzone_enter_deg)throw std::invalid_argument("deadzone exit must be >= enter");
  for(const auto& p:c.control.fov)if(p.zoom<0.F||p.zoom>1.F||p.hfov_deg<=0.F||p.vfov_deg<=0.F)throw std::invalid_argument("invalid FOV calibration point");
}

} // namespace ptz
