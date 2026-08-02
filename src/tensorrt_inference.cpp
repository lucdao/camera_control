#include "ptz/tensorrt_inference.hpp"
#include "ptz/algorithms.hpp"
#include "ptz/engine_builder.hpp"
#include "ptz/cuda_preprocess.hpp"
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <stdexcept>

namespace ptz {
namespace {
class Logger final:public nvinfer1::ILogger{void log(Severity s,const char* m)noexcept override{if(s<=Severity::kWARNING)fprintf(stderr,"TensorRT: %s\n",m);}} logger;
std::vector<char> bytes(const std::filesystem::path& p){std::ifstream in(p,std::ios::binary|std::ios::ate);if(!in)throw std::runtime_error("cannot open engine "+p.string());auto n=in.tellg();in.seekg(0);std::vector<char>b(static_cast<std::size_t>(n));in.read(b.data(),n);return b;}
std::size_t volume(const nvinfer1::Dims& d){std::size_t n=1;for(int i=0;i<d.nbDims;++i)n*=static_cast<std::size_t>(std::max(1,d.d[i]));return n;}
struct TrtModel{
  nvinfer1::IRuntime* runtime{};nvinfer1::ICudaEngine* engine{};nvinfer1::IExecutionContext* context{};cudaStream_t stream{};
  std::string input;std::vector<std::string> outputs;std::map<std::string,void*> device;std::map<std::string,std::vector<float>> host;std::map<std::string,nvinfer1::Dims> shapes;
  explicit TrtModel(const std::filesystem::path& p){auto data=bytes(p);runtime=nvinfer1::createInferRuntime(logger);engine=runtime->deserializeCudaEngine(data.data(),data.size());if(!engine)throw std::runtime_error("deserialize engine failed: "+p.string());context=engine->createExecutionContext();if(!context)throw std::runtime_error("create execution context failed: "+p.string());if(cudaStreamCreate(&stream)!=cudaSuccess)throw std::runtime_error("create CUDA stream failed");
    for(int i=0;i<engine->getNbIOTensors();++i){std::string name=engine->getIOTensorName(i);auto dims=engine->getTensorShape(name.c_str());if(engine->getTensorDataType(name.c_str())!=nvinfer1::DataType::kFLOAT)throw std::runtime_error("TensorRT contract requires FP32 I/O tensor: "+name);for(int d=0;d<dims.nbDims;++d)if(dims.d[d]<=0)throw std::runtime_error("dynamic/unresolved TensorRT shape is unsupported: "+name);shapes[name]=dims;if(engine->getTensorIOMode(name.c_str())==nvinfer1::TensorIOMode::kINPUT){if(!input.empty())throw std::runtime_error("model has more than one input tensor");input=name;}else{outputs.push_back(name);host[name].resize(volume(dims));}void* ptr=nullptr;if(cudaMalloc(&ptr,volume(dims)*sizeof(float))!=cudaSuccess)throw std::runtime_error("CUDA allocation failed for tensor: "+name);device[name]=ptr;}
    if(input.empty()||outputs.empty())throw std::runtime_error("TensorRT engine has an invalid I/O contract: "+p.string());
  }
  ~TrtModel(){for(auto [_,p]:device)cudaFree(p);if(stream)cudaStreamDestroy(stream);delete context;delete engine;delete runtime;}
  std::map<std::string,std::vector<float>> infer_bound(){context->setInputTensorAddress(input.c_str(),device[input]);for(auto& n:outputs)context->setOutputTensorAddress(n.c_str(),device[n]);if(!context->enqueueV3(stream))throw std::runtime_error("TensorRT enqueue failed");for(auto& [n,v]:host)if(cudaMemcpyAsync(v.data(),device[n],v.size()*sizeof(float),cudaMemcpyDeviceToHost,stream)!=cudaSuccess)throw std::runtime_error("TensorRT output copy failed");if(cudaStreamSynchronize(stream)!=cudaSuccess)throw std::runtime_error("TensorRT stream failed");return host;}
  std::map<std::string,std::vector<float>> infer(const std::vector<float>& input_data){auto dims=engine->getTensorShape(input.c_str());if(input_data.size()!=volume(dims))throw std::runtime_error("TensorRT input size mismatch");if(cudaMemcpyAsync(device[input],input_data.data(),input_data.size()*sizeof(float),cudaMemcpyHostToDevice,stream)!=cudaSuccess)throw std::runtime_error("TensorRT input copy failed");return infer_bound();}
  std::map<std::string,std::vector<float>> infer_device(){return infer_bound();}
  float* input_device(){return static_cast<float*>(device.at(input));}
  void* cuda_stream(){return static_cast<void*>(stream);}
  [[nodiscard]] std::array<int,4> input_shape()const{auto d=shapes.at(input);if(d.nbDims!=4)throw std::runtime_error("expected NCHW input");return {d.d[0],d.d[1],d.d[2],d.d[3]};}
};
cv::Mat image(const FramePacket& f){if(f.bgr.size()<static_cast<std::size_t>(f.width*f.height*3))throw std::runtime_error("host BGR frame unavailable");return cv::Mat(f.height,f.width,CV_8UC3,const_cast<std::uint8_t*>(f.bgr.data()));}
struct Prep{std::vector<float> blob;float scale{};int pad_x{},pad_y{};};
Prep letterbox(const cv::Mat& src,int size,float factor,float input_mean,float canvas_value){float scale=std::min(static_cast<float>(size)/src.cols,static_cast<float>(size)/src.rows);int nw=static_cast<int>(std::round(src.cols*scale)),nh=static_cast<int>(std::round(src.rows*scale));cv::Mat resized,canvas(size,size,CV_8UC3,cv::Scalar(canvas_value,canvas_value,canvas_value));cv::resize(src,resized,{nw,nh});int px=(size-nw)/2,py=(size-nh)/2;resized.copyTo(canvas(cv::Rect(px,py,nw,nh)));cv::Mat blob=cv::dnn::blobFromImage(canvas,factor,{size,size},cv::Scalar(input_mean,input_mean,input_mean),true,false,CV_32F);return {{reinterpret_cast<float*>(blob.datastart),reinterpret_cast<float*>(blob.dataend)},scale,px,py};}
float probability(float x){return (x>=0.F&&x<=1.F)?x:1.F/(1.F+std::exp(-x));}
bool device_frame(const FramePacket& f){return f.cuda_device_ptr!=0&&f.pixel_format==FramePacket::PixelFormat::RGBA&&f.cuda_pitch>0;}
void require_input(const TrtModel&m,int height,int width,const char*name){auto s=m.input_shape();if(s!std::array<int,4>{1,3,height,width})throw std::runtime_error(std::string(name)+" input must be 1x3x"+std::to_string(height)+"x"+std::to_string(width));}
int output_fields(const nvinfer1::Dims&dims){if(dims.nbDims<=0)return 0;int last=dims.d[dims.nbDims-1];return last==1||last==2||last==4||last==10||last==56?last:1;}
}

struct TensorRtInference::Impl{
  TrtModel face,arc,pose,body;
  Impl(const std::filesystem::path& m,const std::filesystem::path& c):
    face(EngineBuilder::engine_path(m/"det_10g.onnx",c)),arc(EngineBuilder::engine_path(m/"w600k_r50.onnx",c)),
    pose(EngineBuilder::engine_path(m/"yolo11n-pose.onnx",c)),body(EngineBuilder::engine_path(m/"osnet_x0_25.onnx",c)){
      require_input(face,640,640,"SCRFD");require_input(arc,112,112,"ArcFace");require_input(pose,640,640,"YOLO pose");require_input(body,256,128,"OSNet");
      std::map<int,std::array<int,3>> heads;for(const auto&[name,dims]:face.shapes)if(name!=face.input){int fields=output_fields(dims);int count=static_cast<int>(volume(dims))/fields;for(int stride:{8,16,32})if(count==2*(640/stride)*(640/stride)){if(fields<=2)heads[stride][0]++;else if(fields==4)heads[stride][1]++;else if(fields==10)heads[stride][2]++;}}
      for(int stride:{8,16,32})if(heads[stride]!=std::array<int,3>{1,1,1})throw std::runtime_error("SCRFD outputs do not match score/bbox/kps contract at stride "+std::to_string(stride));
      if(arc.outputs.size()!=1||volume(arc.shapes.at(arc.outputs[0]))!=512)throw std::runtime_error("ArcFace output must contain 512 floats");
      if(pose.outputs.size()!=1||volume(pose.shapes.at(pose.outputs[0]))%56!=0)throw std::runtime_error("YOLO pose output must be [1,56,N] or [1,N,56]");
      if(body.outputs.size()!=1||volume(body.shapes.at(body.outputs[0]))<128)throw std::runtime_error("OSNet output embedding is invalid");
    }
};
TensorRtInference::TensorRtInference(const std::filesystem::path&m,const std::filesystem::path&c):impl_(std::make_unique<Impl>(m,c)){}
TensorRtInference::~TensorRtInference()=default;

std::vector<FaceObservation> TensorRtInference::faces(const FramePacket& frame){
  const float scale=std::min(640.F/static_cast<float>(frame.width),640.F/static_cast<float>(frame.height));const int px=(640-static_cast<int>(std::round(frame.width*scale)))/2,py=(640-static_cast<int>(std::round(frame.height*scale)))/2;Prep prep{{},scale,px,py};std::map<std::string,std::vector<float>> outs;
  if(device_frame(frame)){cuda_letterbox_rgba(frame,impl_->face.input_device(),640,1.F/128.F,127.5F,127.5F,scale,px,py,impl_->face.cuda_stream());outs=impl_->face.infer_device();}else{prep=letterbox(image(frame),640,1.F/128.F,127.5F,127.5F);outs=impl_->face.infer(prep.blob);}
  struct Head{const std::vector<float>* scores{};const std::vector<float>* boxes{};const std::vector<float>* kps{};int count{},stride{};};std::map<int,Head> heads;
  for(auto& [name,v]:outs){int fields=1;const auto dims=impl_->face.shapes.at(name);if(dims.nbDims>0){const int last=dims.d[dims.nbDims-1];if(last==1||last==2||last==4||last==10)fields=last;}int count=static_cast<int>(v.size())/fields,stride=0;for(int s:{8,16,32})if(count==2*(640/s)*(640/s)){stride=s;break;}if(!stride)continue;auto& h=heads[stride];h.count=count;if(fields<=2)h.scores=&v;else if(fields==4)h.boxes=&v;else if(fields==10)h.kps=&v;
  }
  struct Candidate{FaceObservation f;};std::vector<Candidate> candidates;
  for(auto& [stride,h]:heads){if(!h.scores||!h.boxes||!h.kps)continue;int grid=640/stride;
    for(int i=0;i<h.count;++i){float score=probability((*h.scores)[static_cast<std::size_t>(i)*(h.scores->size()/h.count)+(h.scores->size()/h.count-1)]);if(score<.5F)continue;int cell=i/2,gx=cell%grid,gy=cell/grid;float ax=(gx+.5F)*stride,ay=(gy+.5F)*stride;auto base=static_cast<std::size_t>(i)*4;Box b{ax-(*h.boxes)[base]*stride,ay-(*h.boxes)[base+1]*stride,ax+(*h.boxes)[base+2]*stride,ay+(*h.boxes)[base+3]*stride};
      FaceObservation f;f.detection_score=score;f.bbox={(b.x1-prep.pad_x)/prep.scale,(b.y1-prep.pad_y)/prep.scale,(b.x2-prep.pad_x)/prep.scale,(b.y2-prep.pad_y)/prep.scale};auto kb=static_cast<std::size_t>(i)*10;for(int k=0;k<5;++k)f.landmarks[static_cast<std::size_t>(k)]={(((*h.kps)[kb+2*k]*stride+ax)-prep.pad_x)/prep.scale,(((*h.kps)[kb+2*k+1]*stride+ay)-prep.pad_y)/prep.scale};candidates.push_back({f});
    }}
  std::sort(candidates.begin(),candidates.end(),[](auto&a,auto&b){return a.f.detection_score>b.f.detection_score;});std::vector<FaceObservation> result;
  for(auto& c:candidates){bool keep=true;for(auto& r:result)if(iou(c.f.bbox,r.bbox)>.4F){keep=false;break;}if(keep){c.f.bbox.x1=clamp(c.f.bbox.x1,0.F,static_cast<float>(frame.width));c.f.bbox.y1=clamp(c.f.bbox.y1,0.F,static_cast<float>(frame.height));c.f.bbox.x2=clamp(c.f.bbox.x2,0.F,static_cast<float>(frame.width));c.f.bbox.y2=clamp(c.f.bbox.y2,0.F,static_cast<float>(frame.height));result.push_back(std::move(c.f));}}
  return result;
}

std::vector<float> TensorRtInference::face_embedding(const FramePacket& frame,const FaceObservation& face){
  static const std::array<Point2f,5> ref{{{38.2946F,51.6963F},{73.5318F,51.5014F},{56.0252F,71.7366F},{41.5493F,92.3655F},{70.7299F,92.2041F}}};
  std::vector<cv::Point2f> src,dst;for(int i=0;i<5;++i){src.emplace_back(face.landmarks[static_cast<std::size_t>(i)].x,face.landmarks[static_cast<std::size_t>(i)].y);dst.emplace_back(ref[static_cast<std::size_t>(i)].x,ref[static_cast<std::size_t>(i)].y);}
  cv::Mat transform=cv::estimateAffinePartial2D(src,dst);if(transform.empty())return {};std::map<std::string,std::vector<float>> out;
  if(device_frame(frame)){cv::Mat inverse;cv::invertAffineTransform(transform,inverse);float m[6];for(int r=0;r<2;++r)for(int c=0;c<3;++c)m[r*3+c]=static_cast<float>(inverse.at<double>(r,c));cuda_affine_face_rgba(frame,impl_->arc.input_device(),m,impl_->arc.cuda_stream());out=impl_->arc.infer_device();}
  else{cv::Mat aligned;cv::warpAffine(image(frame),aligned,transform,{112,112});cv::Mat blob=cv::dnn::blobFromImage(aligned,1.F/127.5F,{112,112},cv::Scalar(127.5,127.5,127.5),true,false,CV_32F);std::vector<float> input(reinterpret_cast<float*>(blob.datastart),reinterpret_cast<float*>(blob.dataend));out=impl_->arc.infer(input);}return out.empty()?std::vector<float>{}:l2_normalize(out.begin()->second);
}

std::vector<PersonObservation> TensorRtInference::people(const FramePacket& frame){
  const float scale=std::min(640.F/static_cast<float>(frame.width),640.F/static_cast<float>(frame.height));const int px=(640-static_cast<int>(std::round(frame.width*scale)))/2,py=(640-static_cast<int>(std::round(frame.height*scale)))/2;Prep prep{{},scale,px,py};std::map<std::string,std::vector<float>> outs;if(device_frame(frame)){cuda_letterbox_rgba(frame,impl_->pose.input_device(),640,1.F/255.F,0.F,114.F,scale,px,py,impl_->pose.cuda_stream());outs=impl_->pose.infer_device();}else{prep=letterbox(image(frame),640,1.F/255.F,0.F,114.F);outs=impl_->pose.infer(prep.blob);}if(outs.empty())return {};const auto& v=outs.begin()->second;
  constexpr int fields=56;const int count=static_cast<int>(v.size()/fields);const auto dims=impl_->pose.shapes.at(outs.begin()->first);const bool transposed=dims.nbDims>=2&&dims.d[dims.nbDims-2]==fields;std::vector<PersonObservation> candidates;
  auto value=[&](int i,int j){return transposed?v[static_cast<std::size_t>(j)*count+i]:v[static_cast<std::size_t>(i)*fields+j];};
  for(int i=0;i<count;++i){float conf=probability(value(i,4));if(conf<.25F)continue;float cx=value(i,0),cy=value(i,1),w=value(i,2),h=value(i,3);PersonObservation p;p.detection_score=conf;p.bbox={(cx-w/2-prep.pad_x)/prep.scale,(cy-h/2-prep.pad_y)/prep.scale,(cx+w/2-prep.pad_x)/prep.scale,(cy+h/2-prep.pad_y)/prep.scale};for(int k=0;k<17;++k){p.keypoints[static_cast<std::size_t>(k)]={(value(i,5+k*3)-prep.pad_x)/prep.scale,(value(i,6+k*3)-prep.pad_y)/prep.scale};p.keypoint_scores[static_cast<std::size_t>(k)]=probability(value(i,7+k*3));}candidates.push_back(std::move(p));}
  std::sort(candidates.begin(),candidates.end(),[](auto&a,auto&b){return a.detection_score>b.detection_score;});std::vector<PersonObservation> result;for(auto& c:candidates){bool keep=true;for(auto&r:result)if(iou(c.bbox,r.bbox)>.5F){keep=false;break;}if(keep)result.push_back(std::move(c));}return result;
}

std::vector<float> TensorRtInference::body_embedding(const FramePacket& frame,const PersonObservation& p){
  if(p.bbox.width()<=1||p.bbox.height()<=1)return {};std::map<std::string,std::vector<float>>out;if(device_frame(frame)){cuda_body_crop_rgba(frame,impl_->body.input_device(),p.bbox,impl_->body.cuda_stream());out=impl_->body.infer_device();}else{cv::Mat src=image(frame);int x=std::max(0,static_cast<int>(p.bbox.x1)),y=std::max(0,static_cast<int>(p.bbox.y1));int w=std::min(src.cols-x,static_cast<int>(p.bbox.width())),h=std::min(src.rows-y,static_cast<int>(p.bbox.height()));if(w<=1||h<=1)return {};cv::Mat crop=src(cv::Rect(x,y,w,h));cv::Mat blob=cv::dnn::blobFromImage(crop,1.F/255.F,{128,256},cv::Scalar(),true,false,CV_32F);std::vector<float> input(reinterpret_cast<float*>(blob.datastart),reinterpret_cast<float*>(blob.dataend));constexpr float mean[3]{.485F,.456F,.406F},stddev[3]{.229F,.224F,.225F};const std::size_t plane=256U*128U;for(std::size_t c=0;c<3;++c)for(std::size_t i=0;i<plane;++i)input[c*plane+i]=(input[c*plane+i]-mean[c])/stddev[c];out=impl_->body.infer(input);}return out.empty()?std::vector<float>{}:l2_normalize(out.begin()->second);
}
} // namespace ptz
