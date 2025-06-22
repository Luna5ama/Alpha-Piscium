#include "/Base.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

#define CLEAR_IMAGE1 uimg_gbufferData32UI
#define CLEAR_IMAGE2 uimg_gbufferData8UN
#define CLEAR_IMAGE3 uimg_gbufferViewZ
#define CLEAR_IMAGE4 uimg_translucentColor
layout(rgba32ui) uniform writeonly uimage2D CLEAR_IMAGE1;
layout(rgba8) uniform writeonly image2D CLEAR_IMAGE2;
layout(r32f) uniform writeonly image2D CLEAR_IMAGE3;
layout(rgba16f) uniform writeonly image2D CLEAR_IMAGE4;

#define CLEAR_IMAGE_BOUND ivec4(0, 0, global_mainImageSizeI)
#define CLEAR_COLOR1 uvec4(0.0)
#define CLEAR_COLOR2 vec4(0.0)
#define CLEAR_COLOR3 vec4(-65536.0)
#define CLEAR_COLOR4 vec4(0.0)

#include "/general/Clear.comp.glsl"