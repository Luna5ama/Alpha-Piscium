#include "/Base.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

/*const*/
#define CLEAR_IMAGE_SIZE ivec2(uval_mainImageSizeI)

#define CLEAR_IMAGE1 uimg_gbufferVoxySolidData
#define CLEAR_IMAGE2 uimg_gbufferVoxyTranslucentData

layout(rgba32ui) uniform writeonly uimage2D CLEAR_IMAGE1;
layout(rgba32ui) uniform writeonly uimage2D CLEAR_IMAGE2;

#define CLEAR_COLOR1 uvec4(0u)
#define CLEAR_COLOR2 uvec4(0u)
/*const*/

#include "/techniques/Clear.comp.glsl"