#include "ptz/algorithms.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace ptz {

float Box::area() const { return std::max(0.F,width()) * std::max(0.F,height()); }
float clamp(float v,float lo,float hi){ return std::max(lo,std::min(v,hi)); }

float iou(const Box& a,const Box& b){
  const float x1=std::max(a.x1,b.x1), y1=std::max(a.y1,b.y1);
  const float x2=std::min(a.x2,b.x2), y2=std::min(a.y2,b.y2);
  const float inter=std::max(0.F,x2-x1)*std::max(0.F,y2-y1);
  const float uni=a.area()+b.area()-inter;
  return uni>0.F?inter/uni:0.F;
}

std::vector<float> l2_normalize(const std::vector<float>& v){
  double sum=0.; for(float x:v) sum+=static_cast<double>(x)*x;
  const float n=static_cast<float>(std::sqrt(sum));
  if(n<=1e-8F) return v;
  std::vector<float> out(v.size());
  std::transform(v.begin(),v.end(),out.begin(),[n](float x){return x/n;});
  return out;
}

float cosine_similarity(const std::vector<float>& a,const std::vector<float>& b){
  if(a.empty()||a.size()!=b.size()) return -1.F;
  const auto na=l2_normalize(a), nb=l2_normalize(b);
  return std::inner_product(na.begin(),na.end(),nb.begin(),0.F);
}

Point2f torso_aim(const PersonObservation& p){
  // COCO: shoulders 5/6, hips 11/12. Prefer the shoulder/hip torso axis.
  auto valid=[&](int i){ return p.keypoint_scores[static_cast<std::size_t>(i)]>=0.35F; };
  if(valid(5)&&valid(6)){
    Point2f s{(p.keypoints[5].x+p.keypoints[6].x)*.5F,
              (p.keypoints[5].y+p.keypoints[6].y)*.5F};
    if(valid(11)&&valid(12)){
      Point2f h{(p.keypoints[11].x+p.keypoints[12].x)*.5F,
                (p.keypoints[11].y+p.keypoints[12].y)*.5F};
      return {s.x*.7F+h.x*.3F,s.y*.7F+h.y*.3F};
    }
    return s;
  }
  return {p.bbox.center().x,p.bbox.y1+p.bbox.height()*.30F};
}

std::optional<std::size_t> associate_face_to_person(
    const FaceObservation& face,const std::vector<PersonObservation>& people){
  const Point2f c=face.bbox.center();
  std::optional<std::size_t> best; float best_score=-1.F;
  for(std::size_t i=0;i<people.size();++i){
    const auto& b=people[i].bbox;
    const bool within=c.x>=b.x1&&c.x<=b.x2&&c.y>=b.y1&&c.y<=b.y1+b.height()*.48F;
    if(!within) continue;
    const float nx=std::abs(c.x-b.center().x)/std::max(1.F,b.width());
    const float ny=std::abs(c.y-(b.y1+b.height()*.18F))/std::max(1.F,b.height());
    const float score=people[i].detection_score-nx-ny;
    if(score>best_score){best_score=score;best=i;}
  }
  return best;
}

IdentityMatcher::IdentityMatcher(float t,float m,int r,int w)
 :threshold_(t),margin_(m),required_(r),window_(w){
  if(r<1||w<r) throw std::invalid_argument("invalid confirmation policy");
}
void IdentityMatcher::set_gallery(std::vector<GalleryIdentity> g){gallery_=std::move(g);confirmations_.clear();}
IdentityEvidence IdentityMatcher::evaluate(const std::vector<float>& e,const std::string& target){
  float best=-1.F,second=-1.F; std::string name;
  for(const auto& id:gallery_){
    float score=-1.F;
    for(const auto& t:id.templates) score=std::max(score,cosine_similarity(e,t));
    if(score>best){second=best;best=score;name=id.name;} else second=std::max(second,score);
  }
  const bool hit=name==target&&best>=threshold_&&(best-second)>=margin_;
  // Multiple faces are scored in one frame. Only the target candidate advances
  // temporal confirmation; the caller records one miss if the frame had none.
  if(hit) confirmations_.push_back(true);
  while(static_cast<int>(confirmations_.size())>window_) confirmations_.pop_front();
  const int count=static_cast<int>(std::count(confirmations_.begin(),confirmations_.end(),true));
  return {name,best,second,hit&&count>=required_};
}
void IdentityMatcher::note_miss(){
  confirmations_.push_back(false);
  while(static_cast<int>(confirmations_.size())>window_) confirmations_.pop_front();
}

void KalmanTarget::reset(Point2f point,float scale,TimePoint time){
  initialized_=true;last_=time;x_={point.x,point.y,0.F,0.F,scale,0.F};p_.fill(0.F);
  for(int i=0;i<6;++i)p_[static_cast<std::size_t>(i*6+i)]=(i==0||i==1)?25.F:10.F;
}
void KalmanTarget::predict(TimePoint time){
  if(!initialized_) return;
  const float dt=clamp(std::chrono::duration<float>(time-last_).count(),0.F,.5F); last_=time;
  x_[0]+=x_[2]*dt;x_[1]+=x_[3]*dt;x_[4]+=x_[5]*dt;
  const float q=4.F*std::max(dt,.001F);
  for(int i=0;i<6;++i)p_[static_cast<std::size_t>(i*6+i)]+=q*(i<2?1.F:.25F);
}
void KalmanTarget::update(Point2f z,float s,float confidence,TimePoint time){
  if(!initialized_){reset(z,s,time);return;}
  const float dt=clamp(std::chrono::duration<float>(time-last_).count(),.001F,.5F);
  predict(time);
  const float gain=clamp(.15F+.70F*confidence,.15F,.85F);
  const float rx=z.x-x_[0],ry=z.y-x_[1],rs=s-x_[4];
  x_[0]+=gain*rx;x_[1]+=gain*ry;x_[4]+=gain*rs;
  x_[2]+=.45F*gain*rx/dt;x_[3]+=.45F*gain*ry/dt;x_[5]+=.35F*gain*rs/dt;
  for(int i=0;i<6;++i)p_[static_cast<std::size_t>(i*6+i)]*=1.F-gain*.7F;
}

OneEuroFilter::OneEuroFilter(float f,float c,float b,float dc):frequency_(f),min_cutoff_(c),beta_(b),d_cutoff_(dc){}
float OneEuroFilter::alpha(float cutoff,float dt)const{
  constexpr float pi=3.14159265358979323846F; const float tau=1.F/(2.F*pi*cutoff);return 1.F/(1.F+tau/dt);
}
float OneEuroFilter::filter(float v,TimePoint t){
  if(!ready_){ready_=true;previous_=v;last_=t;derivative_=0.F;return v;}
  const float dt=clamp(std::chrono::duration<float>(t-last_).count(),1.F/(frequency_*4.F),1.F);last_=t;
  const float raw_d=(v-previous_)/dt; derivative_+=alpha(d_cutoff_,dt)*(raw_d-derivative_);
  const float cutoff=min_cutoff_+beta_*std::abs(derivative_);
  previous_+=alpha(cutoff,dt)*(v-previous_);return previous_;
}
void OneEuroFilter::reset(){ready_=false;previous_=derivative_=0.F;}

FovCalibration::FovCalibration(std::vector<FovPoint> p):points_(std::move(p)){
  if(points_.empty()) throw std::invalid_argument("FOV calibration is empty");
  std::sort(points_.begin(),points_.end(),[](auto a,auto b){return a.zoom<b.zoom;});
}
FovPoint FovCalibration::at(float z)const{
  z=clamp(z,points_.front().zoom,points_.back().zoom);
  auto hi=std::lower_bound(points_.begin(),points_.end(),z,[](auto p,float v){return p.zoom<v;});
  if(hi==points_.begin())return *hi;
  if(hi==points_.end())return points_.back();
  auto lo=hi-1;
  const float a=(z-lo->zoom)/std::max(1e-6F,hi->zoom-lo->zoom);
  return {z,lo->hfov_deg+a*(hi->hfov_deg-lo->hfov_deg),lo->vfov_deg+a*(hi->vfov_deg-lo->vfov_deg)};
}
Point2f FovCalibration::pixel_to_angle(Point2f p,int w,int h,float tx,float ty,float z)const{
  if(w<=0||h<=0) return {};
  constexpr float pi=3.14159265358979323846F;const auto f=at(z);
  const float fw=static_cast<float>(w),fh=static_cast<float>(h);
  const float fx=(fw*.5F)/std::tan(f.hfov_deg*pi/360.F),fy=(fh*.5F)/std::tan(f.vfov_deg*pi/360.F);
  const float target_px=tx*fw,target_py=ty*fh;
  return {std::atan((p.x-target_px)/fx)*180.F/pi,std::atan((target_py-p.y)/fy)*180.F/pi};
}

} // namespace ptz
