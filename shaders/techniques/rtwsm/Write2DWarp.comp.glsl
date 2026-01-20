#extension GL_KHR_shader_subgroup_arithmetic : enable

#include "RTWSM.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const ivec3 workGroups = ivec3(IMAP_SIZE_D16, IMAP_SIZE_D16, 1);

layout(rgba16) uniform writeonly restrict image2D uimg_frgba16;

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    float warpX = persistent_rtwsm_warp_fetch(ivec2(texelPos.x, 0)).x;
    float warpY = persistent_rtwsm_warp_fetch(ivec2(texelPos.y, 1)).x;
    float texelSizeX = persistent_rtwsm_texelSize_fetch(ivec2(texelPos.x, 0)).x;
    float texelSizeY = persistent_rtwsm_texelSize_fetch(ivec2(texelPos.y, 1)).x;
    vec2 warpXY = vec2(warpX, warpY) * 0.5 + 0.5;
    vec2 texelSizeXY = vec2(texelSizeX, texelSizeY);
    persistent_rtwsm_warpTexelSize_store(texelPos, vec4(warpXY, texelSizeXY));
}
