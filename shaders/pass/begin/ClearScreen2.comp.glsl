#include "/Base.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

/*const*/
#define CLEAR_IMAGE1 uimg_csr32f
#define CLEAR_IMAGE2 CLEAR_IMAGE1
#define CLEAR_IMAGE3 CLEAR_IMAGE1
#define CLEAR_IMAGE4 CLEAR_IMAGE1
#define CLEAR_IMAGE5 CLEAR_IMAGE1
#define CLEAR_IMAGE6 CLEAR_IMAGE1

layout(r32f) uniform writeonly image2D CLEAR_IMAGE1;

#define CLEAR_IMAGE_SIZE ivec2(uval_mainImageSizeI)

#define CLEAR_OFFSET1 ivec2(0)
#define CLEAR_COLOR1 vec4(65536.0)
#define CLEAR_OFFSET2 ivec2(uval_mainImageSizeI.x, 0)
#define CLEAR_COLOR2 vec4(0.0)
#define CLEAR_OFFSET3 ivec2(0, uval_mainImageSizeI.y)
#define CLEAR_COLOR3 vec4(65536.0)
#define CLEAR_OFFSET4 ivec2(uval_mainImageSizeI.x, uval_mainImageSizeI.y)
#define CLEAR_COLOR4 vec4(0.0)
#define CLEAR_OFFSET5 ivec2(0, uval_mainImageSizeI.y * 2)
#define CLEAR_COLOR5 vec4(0.0)
#define CLEAR_OFFSET6 ivec2(uval_mainImageSizeI.x, uval_mainImageSizeI.y * 2)
#define CLEAR_COLOR6 vec4(0.0)
/*const*/

#include "/techniques/Clear.comp.glsl"