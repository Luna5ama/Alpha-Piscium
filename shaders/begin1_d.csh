#version 460 compatibility
#include "rtwsm/RTWSM.glsl"

const vec2 workGroupsRender = vec2(1.0, 1.0);

#define CLEAR_IMAGE1 uimg_temp1
#define CLEAR_IMAGE2 uimg_gbufferData
#define CLEAR_IMAGE3 uimg_gbufferViewZ
layout(rgba16f) uniform writeonly image2D CLEAR_IMAGE1;
layout(rgba32ui) uniform writeonly uimage2D CLEAR_IMAGE2;
layout(r32f) uniform writeonly image2D CLEAR_IMAGE3;

const ivec4 CLEAR_IMAGE_BOUND = ivec4(0, 0, global_mainImageSizeI);
const vec4 CLEAR_COLOR1 = vec4(0.0);
const uvec4 CLEAR_COLOR2 = uvec4(0u);
const vec4 CLEAR_COLOR3 = vec4(-65536.0);

#include "general/Clear3.comp"