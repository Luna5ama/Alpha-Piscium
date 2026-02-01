#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_shuffle : enable
#extension GL_KHR_shader_subgroup_ballot : enable

#include "/techniques/gi/RaySort.glsl"

bool checkBinRayCount(uvec2 binIdx, out uint rayCountOut) {
    vec2 gatherTexel = vec2(binIdx) * 2.0 + 1.0;
    vec4 rayCounts = transient_initialSampleRayCount_gatherTexel(gatherTexel, 0);
    rayCountOut = uint(sum4(rayCounts));
    return any(greaterThan(rayCounts, vec4(0.5)));
}

uint incrementCounter(uint binGroupCount) {
    return atomicAdd(global_dispatchSize3.x, binGroupCount);
}