#pragma once
#include "ptz/types.hpp"
#include <cstddef>

namespace ptz {
void cuda_letterbox_rgba(const FramePacket& frame,float* output,int size,float factor,float input_mean,float canvas_value,
                         float scale,int pad_x,int pad_y,void* stream);
void cuda_affine_face_rgba(const FramePacket& frame,float* output,const float inverse[6],void* stream);
void cuda_body_crop_rgba(const FramePacket& frame,float* output,const Box& box,void* stream);
}
