#include "/Base.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

#define CLEAR_IMAGE1 uimg_gbufferData1
#define CLEAR_IMAGE2 uimg_gbufferData2
#define CLEAR_IMAGE3 uimg_gbufferViewZ
#define CLEAR_IMAGE4 uimg_translucentColor
#define CLEAR_IMAGE5 uimg_translucentDepthLayers
#define CLEAR_IMAGE6 CLEAR_IMAGE5
#define CLEAR_IMAGE7 CLEAR_IMAGE5
#define CLEAR_IMAGE8 CLEAR_IMAGE5

layout(rgba32ui) uniform writeonly uimage2D CLEAR_IMAGE1;
layout(rgba8) uniform writeonly image2D CLEAR_IMAGE2;
layout(r32f) uniform writeonly image2D CLEAR_IMAGE3;
layout(rgba16f) uniform writeonly image2D CLEAR_IMAGE4;
layout(r32f) uniform writeonly image2D CLEAR_IMAGE5;

#define CLEAR_IMAGE_SIZE ivec2(global_mainImageSizeI)
#define CLEAR_COLOR1 uvec4(0u)
#define CLEAR_COLOR2 vec4(0.0)
#define CLEAR_COLOR3 vec4(-65536.0)
#define CLEAR_COLOR4 vec4(0.0)

#define CLEAR_OFFSET5 ivec2(0)
#define CLEAR_COLOR5 vec4(0.0)
#define CLEAR_OFFSET6 ivec2(global_mainImageSizeI.x, 0)
#define CLEAR_COLOR6 vec4(65536.0)
#define CLEAR_OFFSET7 ivec2(0, global_mainImageSizeI.y)
#define CLEAR_COLOR7 vec4(0.0)
#define CLEAR_OFFSET8 ivec2(global_mainImageSizeI.x, global_mainImageSizeI.y)
#define CLEAR_COLOR8 vec4(65536.0)

#include "/techniques/Clear.comp.glsl"