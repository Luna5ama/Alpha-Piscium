#extension GL_KHR_shader_subgroup_arithmetic : enable

#include "RTWSM.glsl"

layout(local_size_x = RTWSM_IMAP_SIZE, local_size_y = 1, local_size_z = 1) in;
const ivec3 workGroups = ivec3(1, 2, 1);

layout(r32f) uniform restrict image2D uimg_fr32f;

shared float shared_prefixBuffer[IMAP_SIZE_D16];

void main() {
    shared_prefixBuffer[gl_LocalInvocationID.x] = 0.0;
    barrier();

    ivec2 ti = ivec2(gl_GlobalInvocationID.xy);
    ivec2 inputPos = ti;
    float tValue = persistent_rtwsm_importance1D_load(inputPos).r;
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
    persistent_rtwsm_warp_store(warpOutputPos, vec4(warp));

    ivec2 texelSizeOutputPos = ti;
    persistent_rtwsm_texelSize_store(texelSizeOutputPos, vec4(texelSize));
}
