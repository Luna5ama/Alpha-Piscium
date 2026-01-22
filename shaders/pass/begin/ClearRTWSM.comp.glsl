#include "/techniques/rtwsm/RTWSM.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const ivec3 workGroups = ivec3(IMAP_SIZE_D16, IMAP_SIZE_D16, 1);

/*const*/
#define CLEAR_IMAGE_SIZE ivec2(RTWSM_IMAP_SIZE)
layout(r32ui) uniform writeonly uimage2D uimg_fr32f;
#define CLEAR_1(texelPos) persistent_rtwsm_importance2D_store(texelPos, uvec4(0u))
/*const*/

#include "/techniques/Clear2.comp.glsl"