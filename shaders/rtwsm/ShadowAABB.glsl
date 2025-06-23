#include "RTWSM.glsl"
#include "/util/FullScreenComp.glsl"

shared vec3 shared_shadowAABBMin[16];
shared vec3 shared_shadowAABBMax[16];

//uniform sampler2D usam_gbufferViewZ;

void rtwsm_shadowAABB() {
    ivec2 clampedTexelPos = coords_clampTexelPos(texelPos, global_mainImageSizeI);
    vec2 screenPos = coords_texelToUV(clampedTexelPos, global_mainImageSizeRcp);
    float viewZ = texelFetch(usam_gbufferViewZ, clampedTexelPos, 0).x;
    vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);
    vec4 scenePos = viewZ != -65536.0 ? gbufferModelViewInverse * vec4(viewPos, 1.0) : vec4(0.0, 0.0, 0.0, 1.0);
    vec3 shadowViewPos = (shadowModelView * scenePos).xyz;
    vec3 shadowViewPosExtended = shadowViewPos;
    shadowViewPosExtended.z += 512.0;

    const float EXTEND_SIZE = 1.0;
    const vec3 EXTEND_VEC = vec3(EXTEND_SIZE, EXTEND_SIZE, 0.0);

    vec3 min0 = min(shadowViewPos - EXTEND_VEC, shadowViewPosExtended);
    vec3 max0 = max(shadowViewPos + EXTEND_VEC, shadowViewPosExtended);

    vec3 min1 = subgroupMin(min0);
    vec3 max1 = subgroupMax(max0);

    if (subgroupElect()) {
        shared_shadowAABBMin[gl_SubgroupID] = min1;
        shared_shadowAABBMax[gl_SubgroupID] = max1;
    }

    barrier();

    if (gl_SubgroupID == 0 && gl_SubgroupInvocationID < gl_NumSubgroups) {
        vec3 min2 = shared_shadowAABBMin[gl_SubgroupInvocationID];
        vec3 max2 = shared_shadowAABBMax[gl_SubgroupInvocationID];

        vec3 min3 = subgroupMin(min2);
        vec3 max3 = subgroupMax(max2);

        if (subgroupElect()) {
            ivec3 min4 = ivec3(floor(min3)) - 1;
            ivec3 max4 = ivec3(ceil(max3)) + 1;
            atomicMin(global_shadowAABBMin.x, min4.x);
            atomicMin(global_shadowAABBMin.y, min4.y);
            atomicMin(global_shadowAABBMin.z, min4.z);
            atomicMax(global_shadowAABBMax.x, max4.x);
            atomicMax(global_shadowAABBMax.y, max4.y);
            atomicMax(global_shadowAABBMax.z, max4.z);
        }
    }
}
