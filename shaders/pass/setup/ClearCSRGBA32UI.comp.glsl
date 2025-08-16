#include "/techniques/textile/CSRGBA32UI.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 2.0);

#define CLEAR_IMAGE1 uimg_csrgba32ui
layout(rgba32ui) uniform writeonly uimage2D CLEAR_IMAGE1;

#define CLEAR_IMAGE_BOUND ivec4(0, 0, _CSRGBA32UI_TEXTURE_SIZE)
#define CLEAR_COLOR1 uvec4(0u)

#include "/techniques/Clear.comp.glsl"