#version 460 compatibility

#include "general/EnvProbe.glsl"
#include "/_Base.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const ivec3 workGroups = ivec3(32, 32, 1);

#define CLEAR_IMAGE1 uimg_envProbe
layout(rgba32ui) uniform writeonly uimage2D CLEAR_IMAGE1;

const ivec4 CLEAR_IMAGE_BOUND = ivec4(0, 0, ENV_PROBE_SIZEI);
uvec4 CLEAR_COLOR1 = uvec4(0u, 0u, 0u, floatBitsToUint(32768.0));

#include "general/Clear.comp"