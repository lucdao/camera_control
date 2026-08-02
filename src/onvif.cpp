#include "ptz/onvif.hpp"
#include <curl/curl.h>
#include <pugixml.hpp>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace ptz {
namespace {
size_t append(char* p,size_t s,size_t n,void* u){static_cast<std::string*>(u)->append(p,s*n);return s*n;}
std::string node_text(const pugi::xml_document& d,const char* xpath){auto n=d.select_node(xpath);return n?n.node().text().as_string():"";}
std::string f(float v){std::ostringstream o;o<<std::fixed<<std::setprecision(5)<<v;return o.str();}
}

OnvifBackend::OnvifBackend(OnvifConfig c):config_(std::move(c)){curl_global_init(CURL_GLOBAL_DEFAULT);curl_handle_=curl_easy_init();if(!curl_handle_)throw std::runtime_error("curl init failed");}
OnvifBackend::~OnvifBackend(){stop();curl_easy_cleanup(static_cast<CURL*>(curl_handle_));curl_global_cleanup();}
std::string OnvifBackend::envelope(const std::string& b)const{return "<?xml version=\"1.0\" encoding=\"UTF-8\"?><s:Envelope xmlns:s=\"http://www.w3.org/2003/05/soap-envelope\" xmlns:tds=\"http://www.onvif.org/ver10/device/wsdl\" xmlns:trt=\"http://www.onvif.org/ver10/media/wsdl\" xmlns:tptz=\"http://www.onvif.org/ver20/ptz/wsdl\" xmlns:tt=\"http://www.onvif.org/ver10/schema\"><s:Body>"+b+"</s:Body></s:Envelope>";}
std::string OnvifBackend::request(const std::string& endpoint,const std::string& action,const std::string& body){
  CURL* c=static_cast<CURL*>(curl_handle_);curl_easy_reset(c);std::string response;curl_slist* headers=nullptr;
  headers=curl_slist_append(headers,("Content-Type: application/soap+xml; charset=utf-8; action=\""+action+"\"").c_str());
  const std::string xml=envelope(body);curl_easy_setopt(c,CURLOPT_URL,endpoint.c_str());curl_easy_setopt(c,CURLOPT_HTTPHEADER,headers);
  curl_easy_setopt(c,CURLOPT_POSTFIELDS,xml.c_str());curl_easy_setopt(c,CURLOPT_POSTFIELDSIZE,static_cast<long>(xml.size()));
  curl_easy_setopt(c,CURLOPT_HTTPAUTH,CURLAUTH_DIGEST|CURLAUTH_BASIC);curl_easy_setopt(c,CURLOPT_USERNAME,config_.username.c_str());curl_easy_setopt(c,CURLOPT_PASSWORD,config_.password.c_str());
  curl_easy_setopt(c,CURLOPT_CONNECTTIMEOUT_MS,std::min(200L,config_.timeout_ms));curl_easy_setopt(c,CURLOPT_TIMEOUT_MS,config_.timeout_ms);curl_easy_setopt(c,CURLOPT_NOSIGNAL,1L);curl_easy_setopt(c,CURLOPT_WRITEFUNCTION,append);curl_easy_setopt(c,CURLOPT_WRITEDATA,&response);
  const CURLcode rc=curl_easy_perform(c);long status=0;curl_easy_getinfo(c,CURLINFO_RESPONSE_CODE,&status);curl_slist_free_all(headers);
  if(rc!=CURLE_OK||status<200||status>=300)throw std::runtime_error("ONVIF request failed: "+std::to_string(status)+" "+curl_easy_strerror(rc));return response;
}
void OnvifBackend::probe(){
  std::lock_guard lock(mutex_);pugi::xml_document doc;
  const auto caps=request(config_.device_endpoint,"http://www.onvif.org/ver10/device/wsdl/GetCapabilities","<tds:GetCapabilities><tds:Category>All</tds:Category></tds:GetCapabilities>");
  if(!doc.load_string(caps.c_str()))throw std::runtime_error("invalid ONVIF capabilities XML");
  media_endpoint_=node_text(doc,"//*[local-name()='Media']/*[local-name()='XAddr']");ptz_endpoint_=node_text(doc,"//*[local-name()='PTZ']/*[local-name()='XAddr']");
  if(media_endpoint_.empty()||ptz_endpoint_.empty())throw std::runtime_error("camera lacks ONVIF media/PTZ capability");
  const auto profiles=request(media_endpoint_,"http://www.onvif.org/ver10/media/wsdl/GetProfiles","<trt:GetProfiles/>");doc.reset();doc.load_string(profiles.c_str());
  const auto profile_node=doc.select_node("//*[local-name()='Profiles']").node();
  profile_token_=config_.profile_token.empty()?profile_node.attribute("token").as_string():config_.profile_token;
  if(profile_token_.empty())throw std::runtime_error("no ONVIF media profile token");probed_=true;
}
PTZPosition OnvifBackend::position(){
  std::lock_guard lock(mutex_);if(!probed_)return {};
  const auto now=Clock::now();if(last_status_!=TimePoint{}&&std::chrono::duration<float>(now-last_status_).count()<.2F)return cached_;
  try{const auto xml=request(ptz_endpoint_,"http://www.onvif.org/ver20/ptz/wsdl/GetStatus","<tptz:GetStatus><tptz:ProfileToken>"+profile_token_+"</tptz:ProfileToken></tptz:GetStatus>");pugi::xml_document d;d.load_string(xml.c_str());
    auto pan=d.select_node("//*[local-name()='PanTilt']").node(),zoom=d.select_node("//*[local-name()='Zoom']").node();
    const std::string pan_status=node_text(d,"//*[local-name()='MoveStatus']/*[local-name()='PanTilt']"),zoom_status=node_text(d,"//*[local-name()='MoveStatus']/*[local-name()='Zoom']");cached_={pan.attribute("x").as_float(),pan.attribute("y").as_float(),zoom.attribute("x").as_float(),static_cast<bool>(pan)&&static_cast<bool>(zoom),!pan_status.empty()&&!zoom_status.empty(),pan_status!="IDLE",zoom_status!="IDLE",now};last_status_=now;
  }catch(...){cached_.valid=false;last_status_=now;}return cached_;
}
void OnvifBackend::continuous_move(const PTZCommand& c){
  std::lock_guard lock(mutex_);if(!probed_)throw std::runtime_error("ONVIF backend not probed");
  const std::string body="<tptz:ContinuousMove><tptz:ProfileToken>"+profile_token_+"</tptz:ProfileToken><tptz:Velocity><tt:PanTilt x=\""+f(c.pan_velocity)+"\" y=\""+f(c.tilt_velocity)+"\"/><tt:Zoom x=\""+f(c.zoom_velocity)+"\"/></tptz:Velocity><tptz:Timeout>PT0.6S</tptz:Timeout></tptz:ContinuousMove>";
  (void)request(ptz_endpoint_,"http://www.onvif.org/ver20/ptz/wsdl/ContinuousMove",body);
}
void OnvifBackend::stop()noexcept{try{std::lock_guard lock(mutex_);if(!probed_)return;(void)request(ptz_endpoint_,"http://www.onvif.org/ver20/ptz/wsdl/Stop","<tptz:Stop><tptz:ProfileToken>"+profile_token_+"</tptz:ProfileToken><tptz:PanTilt>true</tptz:PanTilt><tptz:Zoom>true</tptz:Zoom></tptz:Stop>");}catch(...) {}}

} // namespace ptz
