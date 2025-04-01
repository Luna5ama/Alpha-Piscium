#pragma optionNV (unroll all)

#extension GL_KHR_shader_subgroup_arithmetic : enable

#include "RTWSM.glsl"

layout(local_size_x = SETTING_RTWSM_IMAP_SIZE, local_size_y = 1, local_size_z = 1) in;
const ivec3 workGroups = ivec3(1, 2, 1);

layout(r32f) uniform restrict image2D uimg_rtwsm_imap;

shared float shared_prefixBuffer[IMAP_SIZE_D16];

void main() {
    shared_prefixBuffer[gl_LocalInvocationID.x] = 0.0;
    barrier();

    ivec2 ti = ivec2(gl_GlobalInvocationID.xy);
    ivec2 inputPos = ti;
    inputPos.y += IMAP1D_X_Y;
    float tValue = imageLoad(uimg_rtwsm_imap, inputPos).r;
    float prefix = subgroupInclusiveAdd(tValue);
    if (gl_SubgroupInvocationID == gl_SubgroupSize - 1) {
        shared_prefixBuffer[gl_SubgroupID] = prefix;
    }
    barrier();

    float tValue2 = shared_prefixBuffer[gl_LocalInvocationID.x];
    barrier();
    if (gl_SubgroupID == 0) {
        float prefix2 = subgroupInclusiveAdd(tValue2);
        shared_prefixBuffer[gl_LocalInvocationID.x] = prefix2;
    }
    barrier();

    prefix += gl_SubgroupID == 0 ? 0.0 : shared_prefixBuffer[gl_SubgroupID - 1];
    barrier();

    float k = float(gl_LocalInvocationID.x + 1);
    float n = float(gl_WorkGroupSize.x);

    float prefixExcl = prefix - tValue;
    float total = shared_prefixBuffer[gl_NumSubgroups - 1];
    float warp = (prefixExcl / total) - (k / n);
    float texelSize = tValue / max(total, 1.0);

    ivec2 warpOutputPos = ti;
    warpOutputPos.y += WARP_X_Y;

    ivec2 texelSizeOutputPos = ti;
    texelSizeOutputPos.y += TEXELSIZE_X_Y;

    float prevWarp = imageLoad(uimg_rtwsm_imap, warpOutputPos).r;
    float prevTexelSize = imageLoad(uimg_rtwsm_imap, texelSizeOutputPos).r;
    imageStore(uimg_rtwsm_imap, warpOutputPos, vec4(warp));
    imageStore(uimg_rtwsm_imap, texelSizeOutputPos, vec4(texelSize));
}
