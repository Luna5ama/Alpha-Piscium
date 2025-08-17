#extension GL_KHR_shader_subgroup_arithmetic : enable

#include "/util/Coords.glsl"
#include "/techniques/HiZ.glsl"

layout(local_size_x = 32, local_size_y = 32) in;
const ivec3 workGroups = ivec3(1, 1, 1);


shared vec2 shared_subgroupTemp[32];
shared vec2 shared_groupTemp;

void main() {
    vec2 centerTexelPos = global_mainImageSize / 4.0;
    vec2 texelPos = centerTexelPos + vec2(gl_LocalInvocationID.xy) - 16.0;
    float hiz = hiz_closest_load(ivec2(texelPos), 1);
    float viewZ = abs(coords_reversedZToViewZ(hiz, near));

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
            float a = saturate(coords_viewZToReversedZ(-global_focusDistance, near));
            float b = saturate(coords_viewZToReversedZ(-weightedAverage, near));
            float c = mix(a, b, mixWeight);
            global_focusDistance = -coords_reversedZToViewZ(c, near);
        }
    }
    barrier();
}