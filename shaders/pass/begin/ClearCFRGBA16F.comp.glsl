#include "/techniques/EnvProbe.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const ivec3 workGroups = ivec3(64, 48, 1);

/*const*/
#define CLEAR_IMAGE1 uimg_cfrgba16f
layout(rgba16f) uniform restrict writeonly image2D CLEAR_IMAGE1;

#define CLEAR_IMAGE_SIZE ivec2(1024)
#define CLEAR_COLOR1 uvec4(0u)
/*const*/

#include "/techniques/Clear.comp.glsl"