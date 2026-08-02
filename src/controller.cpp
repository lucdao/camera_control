#include "ptz/controller.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <thread>

namespace ptz {
namespace {void interruptible_sleep(std::stop_token stop,float seconds){const auto end=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<float>(seconds));while(!stop.stop_requested()&&Clock::now()<end)std::this_thread::sleep_for(std::chrono::milliseconds(50));}}

void LatestTargetState::publish(TargetState s){std::lock_guard lock(mutex_);state_=std::move(s);}
TargetState LatestTargetState::latest()const{std::lock_guard lock(mutex_);return state_;}

float TrajectoryAxis::update(float desired,float dt,float max_accel,float max_jerk){
  if(dt<=0.F)return velocity_;
  const float requested_accel=clamp((desired-velocity_)/dt,-max_accel,max_accel);
  const float accel_step=max_jerk*dt;
  acceleration_+=clamp(requested_accel-acceleration_,-accel_step,accel_step);
  acceleration_=clamp(acceleration_,-max_accel,max_accel);
  const float next=velocity_+acceleration_*dt;
  if((desired-velocity_)*(desired-next)<=0.F){velocity_=desired;acceleration_=0.F;}else velocity_=next;
  return velocity_;
}
void TrajectoryAxis::reset(){velocity_=acceleration_=0.F;}

VisualPTZController::VisualPTZController(ITargetStateStore& s,IPTZBackend& b,ControllerConfig c)
 :store_(s),backend_(b),cfg_(std::move(c)),fov_(cfg_.fov),
  pan_filter_(cfg_.rate_hz,cfg_.one_euro_min_cutoff,cfg_.one_euro_beta),
  tilt_filter_(cfg_.rate_hz,cfg_.one_euro_min_cutoff,cfg_.one_euro_beta){}

PTZCommand VisualPTZController::make_stop(TimePoint now,std::string reason){
  pan_axis_.reset();tilt_axis_.reset();zoom_axis_.reset();integral_pan_=integral_tilt_=0.F;
  previous_pan_error_=previous_tilt_error_=0.F;deadzone_hold_=false;
  return {++sequence_,0.F,0.F,0.F,std::move(reason),now,{}};
}

bool VisualPTZController::should_send(const PTZCommand& c,TimePoint now)const{
  const float delta=std::max({std::abs(c.pan_velocity-last_command_.pan_velocity),
                              std::abs(c.tilt_velocity-last_command_.tilt_velocity),
                              std::abs(c.zoom_velocity-last_command_.zoom_velocity)});
  const float elapsed=std::chrono::duration<float,std::milli>(now-last_send_).count();
  return delta>=cfg_.command_change_threshold||elapsed>=cfg_.heartbeat_ms||c.reason!=last_command_.reason;
}

PTZCommand VisualPTZController::step(TimePoint now){
  const TargetState target=store_.latest();
  const float age_ms=target.published_at==TimePoint{}?1e9F:
      std::chrono::duration<float,std::milli>(now-target.published_at).count();
  if(!target.valid||age_ms>cfg_.stale_ms) return make_stop(now,!target.valid?"target_invalid":"target_stale");

  PTZPosition position=backend_.position();
  now=Clock::now();
  if(std::chrono::duration<float,std::milli>(now-target.published_at).count()>cfg_.stale_ms)
    return make_stop(now,"target_stale_after_status");
  const float zoom=position.valid?clamp(position.zoom,0.F,1.F):0.F;
  const float send_age=std::chrono::duration<float>(now-target.captured_at).count();
  const float lead=clamp(send_age+cfg_.actuation_delay_ms/1000.F,0.F,.5F);
  Point2f predicted{target.aim_px.x+target.velocity_px_s.x*lead,
                    target.aim_px.y+target.velocity_px_s.y*lead};
  Point2f angle=fov_.pixel_to_angle(predicted,target.frame_width,target.frame_height,
                                    cfg_.target_x,cfg_.target_y,zoom);
  const float magnitude=std::hypot(angle.x,angle.y);
  if(deadzone_hold_){if(magnitude>cfg_.deadzone_exit_deg)deadzone_hold_=false;}
  else if(magnitude<cfg_.deadzone_enter_deg)deadzone_hold_=true;
  if(deadzone_hold_) angle={};

  const float dt=last_step_==TimePoint{}?1.F/cfg_.rate_hz:
      clamp(std::chrono::duration<float>(now-last_step_).count(),.001F,.2F);
  last_step_=now;
  float ep=pan_filter_.filter(angle.x,now),et=tilt_filter_.filter(angle.y,now);
  integral_pan_=clamp(integral_pan_+ep*dt,-.5F/std::max(cfg_.ki,.001F),.5F/std::max(cfg_.ki,.001F));
  integral_tilt_=clamp(integral_tilt_+et*dt,-.5F/std::max(cfg_.ki,.001F),.5F/std::max(cfg_.ki,.001F));
  const float dp=(ep-previous_pan_error_)/dt,dtl=(et-previous_tilt_error_)/dt;
  previous_pan_error_=ep;previous_tilt_error_=et;
  const auto fov=fov_.at(zoom);
  const float angular_vx=target.velocity_px_s.x/static_cast<float>(std::max(1,target.frame_width))*fov.hfov_deg;
  const float angular_vy=-target.velocity_px_s.y/static_cast<float>(std::max(1,target.frame_height))*fov.vfov_deg;
  const float zoom_gain=1.F+.75F*zoom;
  float desired_pan=clamp(zoom_gain*(cfg_.kp*ep+cfg_.ki*integral_pan_+cfg_.kd*dp+cfg_.kv*angular_vx),
                          -cfg_.max_pan_speed,cfg_.max_pan_speed);
  float desired_tilt=clamp(zoom_gain*(cfg_.kp*et+cfg_.ki*integral_tilt_+cfg_.kd*dtl+cfg_.kv*angular_vy),
                           -cfg_.max_tilt_speed,cfg_.max_tilt_speed);
  float zoom_cmd=0.F;
  const float zoom_dt=last_zoom_step_==TimePoint{}?.2F:std::chrono::duration<float>(now-last_zoom_step_).count();
  if(cfg_.auto_zoom&&position.valid&&zoom_dt>=.2F){
    last_zoom_step_=now;
    const float relative=(cfg_.target_face_height-target.scale)/std::max(.01F,cfg_.target_face_height);
    const float desired=std::abs(relative)<=cfg_.zoom_deadband?0.F:
        clamp(relative*.12F,-cfg_.max_zoom_speed,cfg_.max_zoom_speed);
    zoom_cmd=zoom_axis_.update(desired,zoom_dt,cfg_.max_zoom_accel,cfg_.max_zoom_accel*5.F);
  } else if(!position.valid) zoom_axis_.reset();

  PTZCommand out{++sequence_,
    pan_axis_.update(desired_pan,dt,cfg_.max_accel,cfg_.max_jerk),
    tilt_axis_.update(desired_tilt,dt,cfg_.max_accel,cfg_.max_jerk),zoom_cmd,
    deadzone_hold_?"deadzone_hold":"tracking",now,target.captured_at};
  return out;
}

void VisualPTZController::run(std::stop_token stop){
  using namespace std::chrono;
  const auto period=duration_cast<Clock::duration>(duration<float>(1.F/cfg_.rate_hz));
  auto deadline=Clock::now();
  float backoff_s=1.F;bool connected=false;
  while(!stop.stop_requested()){
    if(!connected){
      backend_.stop();
      try{backend_.probe();connected=true;backoff_s=1.F;deadline=Clock::now();}
      catch(const std::exception& e){std::cerr<<"{\"level\":\"warning\",\"component\":\"onvif\",\"message\":\"probe failed; camera stopped: "<<e.what()<<"\"}\n";interruptible_sleep(stop,backoff_s);backoff_s=std::min(30.F,backoff_s*2.F);continue;}
    }
    deadline+=period;auto cmd=step(Clock::now());
    try{
      if(should_send(cmd,Clock::now())){
        if(cmd.reason.rfind("target_",0)==0) backend_.stop();
        else backend_.continuous_move(cmd);
        last_command_=cmd;last_send_=Clock::now();
      }
    }catch(const std::exception& e){std::cerr<<"{\"level\":\"warning\",\"component\":\"onvif\",\"message\":\"command failed; reconnecting: "<<e.what()<<"\"}\n";backend_.stop();connected=false;continue;}
    std::this_thread::sleep_until(deadline);
    if(Clock::now()>deadline+period)deadline=Clock::now();
  }
  backend_.stop();
}

} // namespace ptz
