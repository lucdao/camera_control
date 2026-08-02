#include "ptz/gstreamer_source.hpp"
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/video/video.h>
#include <cstring>
#include <stdexcept>
#ifdef PTZ_WITH_JETSON_NVMM
#include <nvbufsurface.h>
#include <cudaEGL.h>
#include <cuda_runtime_api.h>
#endif

namespace ptz {
struct GStreamerRtspSource::Impl{GstElement* pipeline{};GstAppSink* sink{};};
#ifdef PTZ_WITH_JETSON_NVMM
namespace {
struct NvmmFrameOwner{
  GstSample* sample{};GstBuffer* buffer{};GstMapInfo map{};NvBufSurface* surface{};cudaGraphicsResource_t resource{};bool egl_mapped{};
  ~NvmmFrameOwner(){if(resource)cudaGraphicsUnregisterResource(resource);if(egl_mapped&&surface)NvBufSurfaceUnMapEglImage(surface,0);if(buffer&&map.data)gst_buffer_unmap(buffer,&map);if(sample)gst_sample_unref(sample);}
};
std::shared_ptr<NvmmFrameOwner> map_nvmm(GstSample* sample){
  auto owner=std::make_shared<NvmmFrameOwner>();owner->sample=sample;owner->buffer=gst_sample_get_buffer(sample);
  if(!gst_buffer_map(owner->buffer,&owner->map,GST_MAP_READ))throw std::runtime_error("cannot map NVMM GstBuffer");owner->surface=reinterpret_cast<NvBufSurface*>(owner->map.data);
  if(!owner->surface||owner->surface->numFilled<1)throw std::runtime_error("empty NvBufSurface");
  if(NvBufSurfaceMapEglImage(owner->surface,0)!=0)throw std::runtime_error("NvBufSurfaceMapEglImage failed");owner->egl_mapped=true;auto image=static_cast<EGLImageKHR>(owner->surface->surfaceList[0].mappedAddr.eglImage);
  if(cudaGraphicsEGLRegisterImage(&owner->resource,image,cudaGraphicsRegisterFlagsReadOnly)!=cudaSuccess)throw std::runtime_error("CUDA EGL registration failed");return owner;
}
}
#endif
GStreamerRtspSource::GStreamerRtspSource(std::string u):impl_(std::make_unique<Impl>()),uri_(std::move(u)){}
GStreamerRtspSource::~GStreamerRtspSource(){close();}
void GStreamerRtspSource::open(){
  gst_init(nullptr,nullptr);std::string safe=uri_;for(auto& c:safe)if(c=='\"')c=' ';
#ifdef PTZ_WITH_JETSON_NVMM
  const std::string desc="rtspsrc location=\""+safe+"\" latency=0 protocols=tcp drop-on-latency=true ! rtph264depay ! h264parse ! nvv4l2decoder enable-max-performance=true ! nvvidconv ! video/x-raw(memory:NVMM),format=RGBA ! appsink name=frames max-buffers=1 drop=true sync=false";
#else
  const std::string desc="rtspsrc location=\""+safe+"\" latency=0 protocols=tcp drop-on-latency=true ! rtph264depay ! h264parse ! nvv4l2decoder enable-max-performance=true ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink name=frames max-buffers=1 drop=true sync=false";
#endif
  GError* error=nullptr;impl_->pipeline=gst_parse_launch(desc.c_str(),&error);
  if(!impl_->pipeline){std::string reason=error?error->message:"unknown";if(error)g_error_free(error);throw std::runtime_error("GStreamer pipeline failed: "+reason);}
  auto* sink=gst_bin_get_by_name(GST_BIN(impl_->pipeline),"frames");impl_->sink=GST_APP_SINK(sink);
  if(gst_element_set_state(impl_->pipeline,GST_STATE_PLAYING)==GST_STATE_CHANGE_FAILURE){close();throw std::runtime_error("RTSP pipeline did not enter PLAYING");}
}
std::optional<FramePacket> GStreamerRtspSource::read(std::stop_token stop){
  while(!stop.stop_requested()){
    GstSample* sample=gst_app_sink_try_pull_sample(impl_->sink,100*GST_MSECOND);if(!sample){if(gst_app_sink_is_eos(impl_->sink))throw std::runtime_error("RTSP stream reached EOS");continue;}
    GstCaps* caps=gst_sample_get_caps(sample);GstVideoInfo info{};gst_video_info_from_caps(&info,caps);int w=GST_VIDEO_INFO_WIDTH(&info),h=GST_VIDEO_INFO_HEIGHT(&info);
    GstBuffer* buffer=gst_sample_get_buffer(sample);GstMapInfo map{};FramePacket out;
#ifdef PTZ_WITH_JETSON_NVMM
    auto owner=map_nvmm(sample);cudaEglFrame egl{};if(cudaGraphicsResourceGetMappedEglFrame(&egl,owner->resource,0,0)!=cudaSuccess||egl.frameType!=cudaEglFrameTypePitch)throw std::runtime_error("NVMM surface is not CUDA pitch memory");const cudaPitchedPtr plane=egl.frame.pPitch[0];if(!plane.ptr||plane.pitch==0)throw std::runtime_error("NVMM CUDA plane is empty");out.sequence=++sequence_;out.width=w;out.height=h;out.captured_at=Clock::now();out.cuda_device_ptr=reinterpret_cast<std::uintptr_t>(plane.ptr);out.cuda_pitch=plane.pitch;out.pixel_format=FramePacket::PixelFormat::RGBA;out.owner=std::move(owner);
#else
    if(gst_buffer_map(buffer,&map,GST_MAP_READ)){
      out.sequence=++sequence_;out.width=w;out.height=h;out.captured_at=Clock::now();
      const std::size_t row=static_cast<std::size_t>(w)*3U,stride=static_cast<std::size_t>(GST_VIDEO_INFO_PLANE_STRIDE(&info,0));out.bgr.resize(row*static_cast<std::size_t>(h));
      for(int y=0;y<h;++y)std::memcpy(out.bgr.data()+static_cast<std::size_t>(y)*row,map.data+static_cast<std::size_t>(y)*stride,row);
      gst_buffer_unmap(buffer,&map);
    }
    gst_sample_unref(sample);
#endif
    const GstClockTime pts=GST_BUFFER_PTS(buffer);GstClock* clock=gst_element_get_clock(impl_->pipeline);
    if(clock&&GST_CLOCK_TIME_IS_VALID(pts)){const GstClockTime running=gst_clock_get_time(clock)-gst_element_get_base_time(impl_->pipeline);if(running>pts)out.captured_at-=std::chrono::nanoseconds(running-pts);}if(clock)gst_object_unref(clock);
    if(out.cuda_device_ptr||!out.bgr.empty())return out;
  }return std::nullopt;
}
void GStreamerRtspSource::close()noexcept{if(!impl_)return;if(impl_->pipeline)gst_element_set_state(impl_->pipeline,GST_STATE_NULL);if(impl_->sink){gst_object_unref(impl_->sink);impl_->sink=nullptr;}if(impl_->pipeline){gst_object_unref(impl_->pipeline);impl_->pipeline=nullptr;}}
}
