#include "/techniques/EnvProbe.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(3.0, 3.0);

#define CLEAR_IMAGE1 uimg_rgba32ui
layout(rgba32ui) uniform restrict writeonly uimage2D CLEAR_IMAGE1;

#define CLEAR_IMAGE_SIZE (ivec2(uval_mainImageSizeI) * ivec2(3, 3))
#define CLEAR_COLOR1 uvec4(0u)

#include "/techniques/Clear.comp.glsl"