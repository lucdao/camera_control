#include "ptz/onvif.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>

int main(int argc,char**argv){
  if(argc!=2){std::cerr<<"endpoint required\n";return 2;}
  try{
    ptz::OnvifBackend backend({argv[1],"operator","secret","",500});backend.probe();
    auto p=backend.position();if(!p.valid||std::abs(p.zoom-.25F)>.001F)return 3;
    std::this_thread::sleep_for(std::chrono::milliseconds(220));if(backend.position().valid)return 4; // HTTP 503
    std::this_thread::sleep_for(std::chrono::milliseconds(220));if(backend.position().valid)return 5; // timeout
    std::this_thread::sleep_for(std::chrono::milliseconds(220));if(!backend.position().valid)return 6; // recovery
    ptz::PTZCommand move;move.pan_velocity=.1F;move.tilt_velocity=-.1F;move.zoom_velocity=.05F;
    bool failed=false;try{backend.continuous_move(move);}catch(const std::exception&){failed=true;}if(!failed)return 7;
    backend.probe();backend.continuous_move(move);backend.stop();
    std::cout<<"onvif integration passed\n";return 0;
  }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}
