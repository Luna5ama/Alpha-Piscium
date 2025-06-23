#include "/general/EnvProbe.glsl"

layout(local_size_x = 32, local_size_y = 16) in;
const ivec3 workGroups = ivec3(32, 32, 1);

#define CLEAR_IMAGE1 uimg_envProbe
layout(rgba32ui) uniform writeonly uimage2D CLEAR_IMAGE1;

#define CLEAR_IMAGE_BOUND ivec4(0, 0, ENV_PROBE_SIZEI.x * 2, ENV_PROBE_SIZEI.y)
#define CLEAR_COLOR1 uvec4(0u)

#include "/general/Clear.comp.glsl"