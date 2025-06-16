#version 460 compatibility

#extension GL_KHR_shader_subgroup_arithmetic : enable

#include "/util/Coords.glsl"

layout(local_size_x = 32, local_size_y = 32) in;
const ivec3 workGroups = ivec3(1, 1, 1);

uniform sampler2D usam_gbufferViewZ;

shared vec2 shared_subgroupTemp[32];
shared vec2 shared_groupTemp;

void main() {
    vec2 centerTexelPos = global_mainImageSize / 2.0;
    vec2 texelPos = centerTexelPos + vec2(gl_LocalInvocationID.xy);
    float viewZ = abs(texelFetch(usam_gbufferViewZ, ivec2(texelPos), 0).r);

    {
        float subgroupMinV = viewZ;
        subgroupMinV = subgroupMin(subgroupMinV);
        if (subgroupElect()) {
            shared_subgroupTemp[gl_SubgroupID].x = subgroupMinV;
        }
    }
    barrier();

    if (gl_SubgroupID == 0 && gl_SubgroupInvocationID < gl_NumSubgroups) {
        float subgroupMinV = shared_subgroupTemp[gl_SubgroupInvocationID].x;
        subgroupMinV = subgroupMin(subgroupMinV);
        if (subgroupElect()) {
            shared_groupTemp.x = subgroupMinV;
        }
    }
    barrier();

    {
        const float WEIGHT_K = 0.1;
        float weight = WEIGHT_K / (WEIGHT_K + pow2(viewZ - shared_groupTemp.x));
        weight /= 1024.0;
        vec2 subgroupSum = vec2(viewZ * weight, weight);
        subgroupSum = subgroupAdd(subgroupSum);
        if (subgroupElect()) {
            shared_subgroupTemp[gl_SubgroupID] = subgroupSum;
        }
    }
    barrier();

    if (gl_SubgroupID == 0 && gl_SubgroupInvocationID < gl_NumSubgroups) {
        vec2 subgroupSum = shared_subgroupTemp[gl_SubgroupInvocationID];
        subgroupSum = subgroupAdd(subgroupSum);
        if (subgroupElect()) {
            float weightedAverage = subgroupSum.x / subgroupSum.y;
            const float FRAME_TIME_60FPS_SECS = 1.0 / 60.0;
            float mixWeight = exp2(-SETTING_DOF_FOCUS_TIME + log2(max(frameTime / FRAME_TIME_60FPS_SECS, 1.0)));
            global_focusDistance = mix(global_focusDistance, weightedAverage, mixWeight);
        }
    }
    barrier();
}