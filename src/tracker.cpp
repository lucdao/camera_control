#include "ptz/tracker.hpp"
#include "ptz/algorithms.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace ptz {
namespace {
std::array<float,4> measurement(const Box& b){const float h=std::max(1.F,b.height());return {b.center().x,b.center().y,std::max(1.F,b.width())/h,h};}
std::array<float,16> invert4(std::array<float,16> a){
  std::array<float,16> inv{};for(int i=0;i<4;++i)inv[static_cast<std::size_t>(i*4+i)]=1.F;
  for(int col=0;col<4;++col){int pivot=col;for(int row=col+1;row<4;++row)if(std::abs(a[static_cast<std::size_t>(row*4+col)])>std::abs(a[static_cast<std::size_t>(pivot*4+col)]))pivot=row;
    if(std::abs(a[static_cast<std::size_t>(pivot*4+col)])<1e-8F)return {};
    if(pivot!=col)for(int j=0;j<4;++j){std::swap(a[static_cast<std::size_t>(col*4+j)],a[static_cast<std::size_t>(pivot*4+j)]);std::swap(inv[static_cast<std::size_t>(col*4+j)],inv[static_cast<std::size_t>(pivot*4+j)]);}
    const float scale=a[static_cast<std::size_t>(col*4+col)];for(int j=0;j<4;++j){a[static_cast<std::size_t>(col*4+j)]/=scale;inv[static_cast<std::size_t>(col*4+j)]/=scale;}
    for(int row=0;row<4;++row)if(row!=col){const float f=a[static_cast<std::size_t>(row*4+col)];for(int j=0;j<4;++j){a[static_cast<std::size_t>(row*4+j)]-=f*a[static_cast<std::size_t>(col*4+j)];inv[static_cast<std::size_t>(row*4+j)]-=f*inv[static_cast<std::size_t>(col*4+j)];}}
  }return inv;
}
}

void BoxKalmanFilter::initiate(const Box& b){
  const auto z=measurement(b);mean_={z[0],z[1],z[2],z[3],0,0,0,0};covariance_.fill(0.F);const float h=z[3];
  // The original ByteTrack Kalman model advances one unit per video frame.
  // This implementation advances in seconds, so velocity covariance is stored
  // in pixels/second (roughly the upstream value divided by 1/30 s).
  const float stds[8]{2.F*h/20.F,2.F*h/20.F,1e-2F,2.F*h/20.F,2.F*h,2.F*h,3e-4F,2.F*h};
  for(int i=0;i<8;++i)covariance_[static_cast<std::size_t>(i*8+i)]=stds[i]*stds[i];
  initialized_=true;
}
void BoxKalmanFilter::predict(float dt){
  if(!initialized_)return;
  dt=clamp(dt,1e-3F,.5F);std::array<float,64> f{};for(int i=0;i<8;++i)f[static_cast<std::size_t>(i*8+i)]=1.F;for(int i=0;i<4;++i)f[static_cast<std::size_t>(i*8+i+4)]=dt;
  std::array<float,8> next{};for(int i=0;i<8;++i)for(int j=0;j<8;++j)next[static_cast<std::size_t>(i)]+=f[static_cast<std::size_t>(i*8+j)]*mean_[static_cast<std::size_t>(j)];mean_=next;
  std::array<float,64> fp{},result{};for(int i=0;i<8;++i)for(int j=0;j<8;++j)for(int k=0;k<8;++k)fp[static_cast<std::size_t>(i*8+j)]+=f[static_cast<std::size_t>(i*8+k)]*covariance_[static_cast<std::size_t>(k*8+j)];for(int i=0;i<8;++i)for(int j=0;j<8;++j)for(int k=0;k<8;++k)result[static_cast<std::size_t>(i*8+j)]+=fp[static_cast<std::size_t>(i*8+k)]*f[static_cast<std::size_t>(j*8+k)];
  const float h=std::max(1.F,mean_[3]);const float q[8]{h/20.F,h/20.F,1e-2F,h/20.F,h/5.F,h/5.F,3e-4F,h/5.F};for(int i=0;i<8;++i)result[static_cast<std::size_t>(i*8+i)]+=q[i]*q[i];covariance_=result;
}
void BoxKalmanFilter::update(const Box& b,float confidence){
  if(!initialized_){initiate(b);return;}const auto z=measurement(b);const float h=std::max(1.F,mean_[3]);const float scale=1.F/clamp(confidence,.1F,1.F);const float rstd[4]{h/20.F,h/20.F,.1F,h/20.F};std::array<float,16>s{};for(int i=0;i<4;++i)for(int j=0;j<4;++j)s[static_cast<std::size_t>(i*4+j)]=covariance_[static_cast<std::size_t>(i*8+j)];for(int i=0;i<4;++i)s[static_cast<std::size_t>(i*4+i)]+=rstd[i]*rstd[i]*scale;const auto sinv=invert4(s);
  std::array<float,32> gain{};for(int i=0;i<8;++i)for(int j=0;j<4;++j)for(int k=0;k<4;++k)gain[static_cast<std::size_t>(i*4+j)]+=covariance_[static_cast<std::size_t>(i*8+k)]*sinv[static_cast<std::size_t>(k*4+j)];std::array<float,4> innovation{};for(int i=0;i<4;++i)innovation[static_cast<std::size_t>(i)]=z[static_cast<std::size_t>(i)]-mean_[static_cast<std::size_t>(i)];for(int i=0;i<8;++i)for(int j=0;j<4;++j)mean_[static_cast<std::size_t>(i)]+=gain[static_cast<std::size_t>(i*4+j)]*innovation[static_cast<std::size_t>(j)];
  std::array<float,64> p{};for(int i=0;i<8;++i)for(int j=0;j<8;++j){p[static_cast<std::size_t>(i*8+j)]=covariance_[static_cast<std::size_t>(i*8+j)];for(int k=0;k<4;++k)p[static_cast<std::size_t>(i*8+j)]-=gain[static_cast<std::size_t>(i*4+k)]*covariance_[static_cast<std::size_t>(k*8+j)];}covariance_=p;
}
Box BoxKalmanFilter::box()const{const float h=std::max(1.F,mean_[3]),w=std::max(1.F,mean_[2]*h);return {mean_[0]-w*.5F,mean_[1]-h*.5F,mean_[0]+w*.5F,mean_[1]+h*.5F};}

std::vector<int> hungarian_min_cost(const std::vector<std::vector<float>>& input){
  if(input.empty())return {};
  const int rows=static_cast<int>(input.size()),cols=input[0].empty()?0:static_cast<int>(input[0].size());
  if(cols==0)return std::vector<int>(static_cast<std::size_t>(rows),-1);
  const int n=std::max(rows,cols);std::vector<std::vector<float>> a(static_cast<std::size_t>(n+1),std::vector<float>(static_cast<std::size_t>(n+1),1e3F));for(int i=1;i<=rows;++i)for(int j=1;j<=cols;++j)a[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]=input[static_cast<std::size_t>(i-1)][static_cast<std::size_t>(j-1)];
  std::vector<float>u(static_cast<std::size_t>(n+1)),v(static_cast<std::size_t>(n+1));std::vector<int>p(static_cast<std::size_t>(n+1)),way(static_cast<std::size_t>(n+1));for(int i=1;i<=n;++i){p[0]=i;int j0=0;std::vector<float>minv(static_cast<std::size_t>(n+1),std::numeric_limits<float>::infinity());std::vector<bool>used(static_cast<std::size_t>(n+1));do{used[static_cast<std::size_t>(j0)]=true;int i0=p[static_cast<std::size_t>(j0)],j1=0;float delta=std::numeric_limits<float>::infinity();for(int j=1;j<=n;++j)if(!used[static_cast<std::size_t>(j)]){float cur=a[static_cast<std::size_t>(i0)][static_cast<std::size_t>(j)]-u[static_cast<std::size_t>(i0)]-v[static_cast<std::size_t>(j)];if(cur<minv[static_cast<std::size_t>(j)]){minv[static_cast<std::size_t>(j)]=cur;way[static_cast<std::size_t>(j)]=j0;}if(minv[static_cast<std::size_t>(j)]<delta){delta=minv[static_cast<std::size_t>(j)];j1=j;}}for(int j=0;j<=n;++j)if(used[static_cast<std::size_t>(j)]){u[static_cast<std::size_t>(p[static_cast<std::size_t>(j)])]+=delta;v[static_cast<std::size_t>(j)]-=delta;}else minv[static_cast<std::size_t>(j)]-=delta;j0=j1;}while(p[static_cast<std::size_t>(j0)]!=0);do{int j1=way[static_cast<std::size_t>(j0)];p[static_cast<std::size_t>(j0)]=p[static_cast<std::size_t>(j1)];j0=j1;}while(j0);}
  std::vector<int>assignment(static_cast<std::size_t>(rows),-1);for(int j=1;j<=n;++j)if(p[static_cast<std::size_t>(j)]>=1&&p[static_cast<std::size_t>(j)]<=rows&&j<=cols)assignment[static_cast<std::size_t>(p[static_cast<std::size_t>(j)]-1)]=j-1;return assignment;
}

ByteTracker::ByteTracker():ByteTracker(Config{}){}
ByteTracker::ByteTracker(Config c):cfg_(c){}
ByteTracker::Assignment ByteTracker::associate(const std::vector<std::uint64_t>& ids,const std::vector<PersonObservation>& ds,float min_iou)const{
  Assignment out;if(ids.empty()){for(std::size_t i=0;i<ds.size();++i)out.unmatched_detections.push_back(i);return out;}if(ds.empty()){for(std::size_t i=0;i<ids.size();++i)out.unmatched_tracks.push_back(i);return out;}std::vector<std::vector<float>> cost(ids.size(),std::vector<float>(ds.size()));for(std::size_t i=0;i<ids.size();++i)for(std::size_t j=0;j<ds.size();++j)cost[i][j]=1.F-iou(tracks_.at(ids[i]).filter.box(),ds[j].bbox);auto assignment=hungarian_min_cost(cost);std::set<std::size_t>used;for(std::size_t i=0;i<assignment.size();++i){int j=assignment[i];if(j>=0&&1.F-cost[i][static_cast<std::size_t>(j)]>=min_iou){out.matches.emplace_back(i,static_cast<std::size_t>(j));used.insert(static_cast<std::size_t>(j));}else out.unmatched_tracks.push_back(i);}for(std::size_t j=0;j<ds.size();++j)if(!used.contains(j))out.unmatched_detections.push_back(j);return out;
}
void ByteTracker::activate(PersonObservation d,TimePoint now,bool confirmed){ByteTrackState t;t.id=next_id_++;t.lifecycle=confirmed?TrackLifecycle::Tracked:TrackLifecycle::New;t.observation=std::move(d);t.observation.track_id=t.id;t.filter.initiate(t.observation.bbox);t.age=t.hits=1;t.updated_at=now;tracks_[t.id]=std::move(t);}
std::vector<PersonObservation> ByteTracker::update(std::vector<PersonObservation> detections,TimePoint now){
  const bool first_frame=last_update_==TimePoint{};const float dt=first_frame?1.F/30.F:clamp(std::chrono::duration<float>(now-last_update_).count(),1e-3F,.5F);last_update_=now;for(auto&[_,t]:tracks_)if(t.lifecycle!=TrackLifecycle::Removed){t.filter.predict(dt);++t.age;t.observation.bbox=t.filter.box();}
  std::vector<PersonObservation>high,low;for(auto&d:detections)if(d.detection_score>=cfg_.high_threshold)high.push_back(std::move(d));else if(d.detection_score>=cfg_.low_threshold)low.push_back(std::move(d));
  std::vector<std::uint64_t>pool;for(const auto&[id,t]:tracks_)if(t.lifecycle==TrackLifecycle::Tracked||t.lifecycle==TrackLifecycle::Lost)pool.push_back(id);auto first=associate(pool,high,cfg_.high_match_iou);std::set<std::uint64_t>matched;
  for(auto [ti,di]:first.matches){auto&t=tracks_.at(pool[ti]);t.filter.update(high[di].bbox,high[di].detection_score);t.observation=high[di];t.observation.bbox=t.filter.box();t.observation.track_id=t.id;t.lifecycle=TrackLifecycle::Tracked;t.lost_frames=0;++t.hits;t.updated_at=now;matched.insert(t.id);}
  std::vector<std::uint64_t>remaining;for(auto ti:first.unmatched_tracks){auto id=pool[ti];if(tracks_.at(id).lifecycle==TrackLifecycle::Tracked)remaining.push_back(id);}auto second=associate(remaining,low,cfg_.low_match_iou);for(auto[ti,di]:second.matches){auto&t=tracks_.at(remaining[ti]);t.filter.update(low[di].bbox,low[di].detection_score);t.observation=low[di];t.observation.bbox=t.filter.box();t.observation.track_id=t.id;t.lifecycle=TrackLifecycle::Tracked;t.lost_frames=0;++t.hits;t.updated_at=now;matched.insert(t.id);}
  for(auto id:pool)if(!matched.contains(id)){auto&t=tracks_.at(id);if(t.lifecycle==TrackLifecycle::Tracked)t.lifecycle=TrackLifecycle::Lost;++t.lost_frames;if(t.lost_frames>cfg_.track_buffer)t.lifecycle=TrackLifecycle::Removed;}
  // ByteTrack handles tentative tracks separately: they must match a remaining
  // high-score detection on the next detector frame or are removed.
  std::vector<std::uint64_t>unconfirmed;for(const auto&[id,t]:tracks_)if(t.lifecycle==TrackLifecycle::New)unconfirmed.push_back(id);std::vector<PersonObservation>remaining_high;for(auto di:first.unmatched_detections)remaining_high.push_back(high[di]);auto tentative=associate(unconfirmed,remaining_high,cfg_.unconfirmed_match_iou);std::set<std::size_t>confirmed_detection;
  for(auto[ti,di]:tentative.matches){auto&t=tracks_.at(unconfirmed[ti]);t.filter.update(remaining_high[di].bbox,remaining_high[di].detection_score);t.observation=remaining_high[di];t.observation.bbox=t.filter.box();t.observation.track_id=t.id;t.lifecycle=TrackLifecycle::Tracked;t.lost_frames=0;++t.hits;t.updated_at=now;confirmed_detection.insert(di);}
  for(auto ti:tentative.unmatched_tracks)tracks_.at(unconfirmed[ti]).lifecycle=TrackLifecycle::Removed;
  for(std::size_t i=0;i<remaining_high.size();++i)if(!confirmed_detection.contains(i)&&remaining_high[i].detection_score>=cfg_.new_track_threshold)activate(remaining_high[i],now,first_frame);
  // Suppress duplicate active/lost tracks created around an occlusion boundary,
  // retaining the longer-lived trajectory as in the reference ByteTrack flow.
  std::vector<std::uint64_t>active,lost;for(const auto&[id,t]:tracks_){if(t.lifecycle==TrackLifecycle::Tracked)active.push_back(id);else if(t.lifecycle==TrackLifecycle::Lost)lost.push_back(id);}for(auto a:active)for(auto l:lost)if(tracks_.at(a).lifecycle!=TrackLifecycle::Removed&&tracks_.at(l).lifecycle!=TrackLifecycle::Removed&&iou(tracks_.at(a).filter.box(),tracks_.at(l).filter.box())>.85F){if(tracks_.at(a).age>=tracks_.at(l).age)tracks_.at(l).lifecycle=TrackLifecycle::Removed;else tracks_.at(a).lifecycle=TrackLifecycle::Removed;}
  std::vector<PersonObservation>out;for(auto&[_,t]:tracks_)if(t.lifecycle==TrackLifecycle::Tracked)out.push_back(t.observation);for(auto it=tracks_.begin();it!=tracks_.end();)if(it->second.lifecycle==TrackLifecycle::Removed)it=tracks_.erase(it);else++it;return out;
}
std::optional<PersonObservation>ByteTracker::by_id(std::uint64_t id)const{auto it=tracks_.find(id);if(it==tracks_.end()||it->second.lifecycle==TrackLifecycle::Removed)return std::nullopt;return it->second.observation;}
std::vector<ByteTrackState>ByteTracker::states()const{std::vector<ByteTrackState>out;for(const auto&[_,t]:tracks_)out.push_back(t);return out;}
} // namespace ptz
