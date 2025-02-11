#version 460 compatibility
#include "rtwsm/RTWSM.glsl"

const vec2 workGroupsRender = vec2(1.0, 1.0);

#define CLEAR_IMAGE1 uimg_gbufferViewZ
#define CLEAR_IMAGE2 uimg_translucentColor
layout(r32f) uniform writeonly image2D CLEAR_IMAGE1;
layout(rgba16f) uniform writeonly image2D CLEAR_IMAGE2;

const ivec4 CLEAR_IMAGE_BOUND = ivec4(0, 0, global_mainImageSizeI);
const vec4 CLEAR_COLOR1 = vec4(-65536.0);
const vec4 CLEAR_COLOR2 = vec4(0.0);

#include "general/Clear.comp"