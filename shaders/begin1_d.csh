#version 460 compatibility
#include "rtwsm/RTWSM.glsl"

const vec2 workGroupsRender = vec2(1.0, 1.0);

#define CLEAR_IMAGE1 uimg_gbufferData
#define CLEAR_IMAGE2 uimg_gbufferViewZ
#define CLEAR_IMAGE3 uimg_temp1
#define CLEAR_IMAGE4 uimg_temp2
#define CLEAR_IMAGE5 uimg_temp3
#define CLEAR_IMAGE6 uimg_temp4
layout(rgba32ui) uniform writeonly uimage2D CLEAR_IMAGE1;
layout(r32f) uniform writeonly image2D CLEAR_IMAGE2;
layout(rgba16f) uniform writeonly image2D CLEAR_IMAGE3;
layout(rgba16f) uniform writeonly image2D CLEAR_IMAGE4;
layout(rgba16f) uniform writeonly image2D CLEAR_IMAGE5;
layout(rgba16f) uniform writeonly image2D CLEAR_IMAGE6;

const ivec4 CLEAR_IMAGE_BOUND = ivec4(0, 0, global_mainImageSizeI);
const uvec4 CLEAR_COLOR1 = uvec4(0u);
const vec4 CLEAR_COLOR2 = vec4(-65536.0);
const vec4 CLEAR_COLOR3 = vec4(0.0);
const vec4 CLEAR_COLOR4 = vec4(0.0);
const vec4 CLEAR_COLOR5 = vec4(0.0);
const vec4 CLEAR_COLOR6 = vec4(0.0);

#include "general/Clear.comp"