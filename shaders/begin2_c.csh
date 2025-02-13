#version 460 compatibility

#include "/_Base.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

#define CLEAR_IMAGE1 uimg_gbufferViewZ
#define CLEAR_IMAGE2 uimg_translucentColor
layout(r32f) uniform writeonly image2D CLEAR_IMAGE1;
layout(rgba16f) uniform writeonly image2D CLEAR_IMAGE2;

#define CLEAR_IMAGE_BOUND ivec4(0, 0, global_mainImageSizeI)
#define CLEAR_COLOR1 vec4(-65536.0)
#define CLEAR_COLOR2 vec4(0.0)

#include "/general/Clear.comp"