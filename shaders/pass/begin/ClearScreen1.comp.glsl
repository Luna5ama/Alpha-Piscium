#include "/Base.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

/*const*/
#define CLEAR_IMAGE1 uimg_gbufferSolidData1
#define CLEAR_IMAGE2 uimg_gbufferSolidData2
#define CLEAR_IMAGE3 uimg_gbufferSolidViewZ
#define CLEAR_IMAGE4 uimg_gbufferTranslucentData1
#define CLEAR_IMAGE5 uimg_gbufferTranslucentData2
#define CLEAR_IMAGE6 uimg_translucentColor

layout(rgba32ui) uniform writeonly uimage2D CLEAR_IMAGE1;
layout(rgba8) uniform writeonly image2D CLEAR_IMAGE2;
layout(r32f) uniform writeonly image2D CLEAR_IMAGE3;
layout(rgba32ui) uniform writeonly uimage2D CLEAR_IMAGE4;
layout(rgba8) uniform writeonly image2D CLEAR_IMAGE5;
layout(rgba16f) uniform writeonly image2D CLEAR_IMAGE6;

#define CLEAR_IMAGE_SIZE ivec2(uval_mainImageSizeI)
#define CLEAR_COLOR1 uvec4(0u)
#define CLEAR_COLOR2 vec4(0.0)
#define CLEAR_COLOR3 vec4(-65536.0)
#define CLEAR_COLOR4 uvec4(0u)
#define CLEAR_COLOR5 vec4(0.0)
#define CLEAR_COLOR6 vec4(1.0, 1.0, 1.0, 0.0)
/*const*/

#include "/techniques/Clear.comp.glsl"