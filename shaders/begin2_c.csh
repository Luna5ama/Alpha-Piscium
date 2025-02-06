#version 460 compatibility
#include "rtwsm/RTWSM.glsl"

const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform writeonly image2D uimg_temp1;
#define CLEAR_IMAGE1 uimg_temp1
const ivec4 CLEAR_IMAGE_BOUND = ivec4(0, 0, global_mainImageSizeI);
const vec4 CLEAR_COLOR1 = vec4(0.0);

#include "general/Clear1.comp"