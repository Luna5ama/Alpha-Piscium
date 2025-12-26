#ifndef INCLUDE_techniques_HiZCheck_glsl
#define INCLUDE_techniques_HiZCheck_glsl a
/* Required extensions:
#extension GL_KHR_shader_subgroup_ballot : enable
*/

#include "HiZ.glsl"

bool hiz_groupGroundCheckSubgroup(uvec2 groupOrigin, int level) {
    bool subgroupCheck = false;
    if (subgroupElect()) {
        subgroupCheck = hiz_closest_load(ivec2(groupOrigin), level) > coords_viewZToReversedZ(-65536.0, near);
    }
    return subgroupBroadcastFirst(subgroupCheck);
}

float hiz_groupGroundCheckSubgroupLoadViewZ(uvec2 groupOrigin, int level, ivec2 texelPos) {
    bool subgroupCheck = false;
    if (subgroupElect()) {
        subgroupCheck = hiz_closest_load(ivec2(groupOrigin), level) > coords_viewZToReversedZ(-65536.0, near);
    }
    float viewZ = -65536.0;
    if (subgroupBroadcastFirst(subgroupCheck)) {
        viewZ = texelFetch(usam_gbufferViewZ, texelPos, 0).r;
    }
    return viewZ;
}

bool hiz_groupSkyCheckSubgroup(uvec2 groupOrigin, int level) {
    bool subgroupCheck = false;
    if (subgroupElect()) {
        subgroupCheck = hiz_furthest_load(ivec2(groupOrigin), level) <= coords_viewZToReversedZ(-65536.0, near);
    }
    return subgroupBroadcastFirst(subgroupCheck);
}

#endif