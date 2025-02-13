#version 460 compatibility

#include "/_Base.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

#define CLEAR_IMAGE1 uimg_temp1
#define CLEAR_IMAGE2 uimg_temp2
#define CLEAR_IMAGE3 uimg_temp3
#define CLEAR_IMAGE4 uimg_temp4
#define CLEAR_IMAGE5 uimg_temp5
#define CLEAR_IMAGE6 uimg_temp6
#define CLEAR_IMAGE7 uimg_temp7
layout(rgba16f) uniform writeonly image2D CLEAR_IMAGE1;
layout(rgba16f) uniform writeonly image2D CLEAR_IMAGE2;
layout(rgba16f) uniform writeonly image2D CLEAR_IMAGE3;
layout(rgba16f) uniform writeonly image2D CLEAR_IMAGE4;
layout(rgba8) uniform writeonly image2D CLEAR_IMAGE5;
layout(rgba8) uniform writeonly image2D CLEAR_IMAGE6;
layout(rgba8) uniform writeonly image2D CLEAR_IMAGE7;

#define CLEAR_IMAGE_BOUND ivec4(0, 0, global_mainImageSizeI)
#define CLEAR_COLOR1 vec4(0.0)
#define CLEAR_COLOR2 vec4(0.0)
#define CLEAR_COLOR3 vec4(0.0)
#define CLEAR_COLOR4 vec4(0.0)
#define CLEAR_COLOR5 vec4(0.0)
#define CLEAR_COLOR6 vec4(0.0)
#define CLEAR_COLOR7 vec4(0.0)

#include "/general/Clear.comp"