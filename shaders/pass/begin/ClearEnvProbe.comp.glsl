#include "/techniques/EnvProbe.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const ivec3 workGroups = ivec3(32, 32, 1);

#define CLEAR_IMAGE1 uimg_envProbe
layout(rgba32ui) uniform writeonly uimage2D CLEAR_IMAGE1;

#define CLEAR_IMAGE_SIZE ivec2(ENV_PROBE_SIZEI.x, ENV_PROBE_SIZEI.y)
#define CLEAR_OFFSET1 ivec2(ENV_PROBE_SIZEI.x, 0)
#define CLEAR_COLOR1 uvec4(0u)

#include "/techniques/Clear.comp.glsl"