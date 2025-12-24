#include "/techniques/EnvProbe.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(3.0, 3.0);

#define CLEAR_IMAGE1 uimg_rgba8
layout(rgba8) uniform restrict writeonly image2D CLEAR_IMAGE1;

#define CLEAR_IMAGE_SIZE (ivec2(uval_mainImageSizeI) * ivec2(2, 2))
#define CLEAR_COLOR1 vec4(0.0)

#include "/techniques/Clear.comp.glsl"