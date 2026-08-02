#include "ptz/health.hpp"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cerrno>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace ptz {
void RuntimeMetrics::record_capture_to_command(double milliseconds){last_capture_to_command_ms=milliseconds;std::lock_guard lock(latency_mutex_);latency_window_.push_back(milliseconds);if(latency_window_.size()>4096)latency_window_.pop_front();}
double RuntimeMetrics::capture_to_command_p95()const{std::lock_guard lock(latency_mutex_);if(latency_window_.empty())return 0.;std::vector<double> values(latency_window_.begin(),latency_window_.end());std::sort(values.begin(),values.end());return values[static_cast<std::size_t>(std::ceil(.95*static_cast<double>(values.size()))-1.)];}
void HealthServer::run(std::stop_token stop){
  int fd=socket(AF_INET,SOCK_STREAM|SOCK_CLOEXEC,0);if(fd<0)return;int yes=1;setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&yes,sizeof(yes));
  sockaddr_in addr{};addr.sin_family=AF_INET;addr.sin_addr.s_addr=htonl(INADDR_ANY);addr.sin_port=htons(static_cast<std::uint16_t>(port_));
  if(bind(fd,reinterpret_cast<sockaddr*>(&addr),sizeof(addr))<0||listen(fd,8)<0){close(fd);return;}
  while(!stop.stop_requested()){
    fd_set set;FD_ZERO(&set);FD_SET(fd,&set);timeval timeout{0,200000};int ready=select(fd+1,&set,nullptr,nullptr,&timeout);if(ready<=0)continue;
    int client=accept4(fd,nullptr,nullptr,SOCK_CLOEXEC);if(client<0)continue;char request[1024]{};const auto n=read(client,request,sizeof(request)-1);std::string path="/";if(n>0){std::istringstream in(std::string(request,static_cast<std::size_t>(n)));std::string method;in>>method>>path;}
    std::string body;int status=200;
    if(path=="/healthz"){const bool ok=metrics_.perception_healthy&&metrics_.control_healthy;status=ok?200:503;body=ok?"{\"status\":\"ok\"}\n":"{\"status\":\"degraded\"}\n";}
    else if(path=="/metrics"){std::ostringstream o;o<<"ptz_perception_healthy "<<(metrics_.perception_healthy?1:0)<<"\nptz_control_healthy "<<(metrics_.control_healthy?1:0)<<"\nptz_frames_total "<<metrics_.frames<<"\nptz_target_updates_total "<<metrics_.target_updates<<"\nptz_commands_total "<<metrics_.commands<<"\nptz_failsafe_stops_total "<<metrics_.failsafe_stops<<"\nptz_capture_to_command_ms "<<metrics_.last_capture_to_command_ms<<"\nptz_capture_to_command_p95_ms "<<metrics_.capture_to_command_p95()<<"\n";body=o.str();}
    else{status=404;body="not found\n";}
    std::ostringstream response;response<<"HTTP/1.1 "<<status<<(status==200?" OK":" Error")<<"\r\nContent-Type: text/plain\r\nContent-Length: "<<body.size()<<"\r\nConnection: close\r\n\r\n"<<body;const auto data=response.str();(void)write(client,data.data(),data.size());close(client);
  }close(fd);
}
}
