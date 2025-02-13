#version 460 compatibility

#include "/general/EnvProbe.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const ivec3 workGroups = ivec3(32, 32, 1);

#define CLEAR_IMAGE1 uimg_envProbe
layout(rgba32ui) uniform writeonly uimage2D CLEAR_IMAGE1;

#define CLEAR_IMAGE_BOUND ivec4(0, 0, ENV_PROBE_SIZEI)
#define CLEAR_COLOR1 uvec4(0u, 0u, 0u, floatBitsToUint(32768.0))

#include "/general/Clear.comp.glsl"