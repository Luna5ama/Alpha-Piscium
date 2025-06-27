#if defined(MC_GL_VENDOR_NVIDIA)
#extension GL_NV_shader_subgroup_partitioned : enable
#extension GL_KHR_shader_subgroup_ballot : enable
#else
#extension GL_KHR_shader_subgroup_vote : enable
#extension GL_KHR_shader_subgroup_clustered : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#endif

#include "RTWSM.glsl"
#include "/util/Coords.glsl"
#include "/util/GBufferData.glsl"
#include "/util/Morton.glsl"

// Required resources
// layout(r32i) uniform iimage2D uimg_rtwsm_imap;

shared vec3 shared_shadowAABBMin[16];
shared vec3 shared_shadowAABBMax[16];

void shadowAABB1(vec3 shadowViewPos) {
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
}

void shadowAABB2() {
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

void importance(ivec2 texelPos, float viewZ, GBufferData gData, out uint p, out float v) {
    vec2 screenPos = (vec2(texelPos) + 0.5 - global_taaJitter) * global_mainImageSizeRcp;
    vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);
    vec4 scenePos = gbufferModelViewInverse * vec4(viewPos, 1.0);

    vec3 viewDir = normalize(-viewPos);

    vec4 shadowViewPos = shadowModelView * scenePos;
    shadowAABB1(shadowViewPos.xyz);

    vec4 shadowClipPos = global_shadowRotationMatrix * shadowProjection * shadowViewPos;
    vec3 shadowNDCPos = shadowClipPos.xyz / shadowClipPos.w;
    vec2 shadowScreenPos = shadowNDCPos.xy * 0.5 + 0.5;

    float importance = SETTING_RTWSM_B_BASE;

    // Distance function
    #if SETTING_RTWSM_B_D > 0.0
    importance *= (SETTING_RTWSM_B_D) / (SETTING_RTWSM_B_D + dot(viewPos, viewPos));
    #endif

    // Surface normal function
    #if SETTING_RTWSM_B_SN > 0.0
    importance *= 1.0 + SETTING_RTWSM_B_SN * saturate(dot(gData.geometryNormal, viewDir));
    #endif

    #if SETTING_RTWSM_B_P > 0.0
    float lightDir = dot(gData.geometryNormal, uval_shadowLightDirView);
    importance *= 1.0 + SETTING_RTWSM_B_P * pow(1.0 - lightDir * lightDir, float(SETTING_RTWSM_B_PP));
    #endif

    // Shadow Edge function
    #if SETTING_RTWSM_B_SE > 0.0
    vec2 shadowScreenPosWarped = rtwsm_warpTexCoord(usam_rtwsm_imap, shadowScreenPos);

    float center = rtwsm_linearDepth(texture(shadowtex0, shadowScreenPosWarped).x);
    float maxDiff = 0.0;
    maxDiff = max(maxDiff, abs(center - rtwsm_linearDepth(textureOffset(shadowtex0, shadowScreenPosWarped, ivec2(-1, -1)).x)));
    maxDiff = max(maxDiff, abs(center - rtwsm_linearDepth(textureOffset(shadowtex0, shadowScreenPosWarped, ivec2(-1, 0)).x)));
    maxDiff = max(maxDiff, abs(center - rtwsm_linearDepth(textureOffset(shadowtex0, shadowScreenPosWarped, ivec2(-1, 1)).x)));
    maxDiff = max(maxDiff, abs(center - rtwsm_linearDepth(textureOffset(shadowtex0, shadowScreenPosWarped, ivec2(0, -1)).x)));
    maxDiff = max(maxDiff, abs(center - rtwsm_linearDepth(textureOffset(shadowtex0, shadowScreenPosWarped, ivec2(0, 1)).x)));
    maxDiff = max(maxDiff, abs(center - rtwsm_linearDepth(textureOffset(shadowtex0, shadowScreenPosWarped, ivec2(1, -1)).x)));
    maxDiff = max(maxDiff, abs(center - rtwsm_linearDepth(textureOffset(shadowtex0, shadowScreenPosWarped, ivec2(1, 0)).x)));
    maxDiff = max(maxDiff, abs(center - rtwsm_linearDepth(textureOffset(shadowtex0, shadowScreenPosWarped, ivec2(1, 1)).x)));

    importance *= 1.0 + SETTING_RTWSM_B_SE * linearStep(0.5, 4.0, maxDiff);
    #endif

    if (any(greaterThanEqual(abs(shadowClipPos.xy), vec2(1.0)))) {
        p = 0xFFFFFFFFu;
        v = 0u;
        return;
    }

    uvec2 shadowPos = uvec2(shadowScreenPos.xy * SETTING_RTWSM_IMAP_SIZE);

    importance = max(importance, uval_rtwsmMin.y);

    p = shadowPos.y << 16 | shadowPos.x;
    v = importance;

    return;
}

void writeOutput(uint p, float v) {
    if (p == 0xFFFFFFFFu) return;
    imageAtomicMax(uimg_rtwsm_imap, ivec2(p & 0xFFFFu, p >> 16), floatBitsToInt(v));
}

void backwardOutput(uint p, float v) {
    #ifdef MC_GL_VENDOR_NVIDIA
    uvec4 pballot = subgroupPartitionNV(p);
    float maxV = subgroupPartitionedMaxNV(v, pballot);
    if (subgroupBallotFindLSB(pballot) == gl_SubgroupInvocationID) {
        writeOutput(p, maxV);
    }
    #else
    if (p == 0xFFFFFFFFu) {
        p = subgroupMin(p);
    }
    if (subgroupAllEqual(p)) {
        if (gl_SubgroupInvocationID == 0) {
            writeOutput(p, subgroupMax(v));
        }
        return;
    }

    #define CLUSTERED_DEDUP(n) if (subgroupClusteredXor(p, n) != 0) { writeOutput(p, v); return; } if ((gl_SubgroupInvocationID & (n - 1u)) != 0u) return; v = subgroupClusteredMax(v, n);

    CLUSTERED_DEDUP(2u)
    CLUSTERED_DEDUP(4u)
    CLUSTERED_DEDUP(8u)
    CLUSTERED_DEDUP(16u)
    writeOutput(p, v);
    #endif
}

void rtwsm_backward(ivec2 texelPos, float viewZ, GBufferData gData) {
    if (gl_LocalInvocationIndex < 16) {
        shared_shadowAABBMax[gl_LocalInvocationIndex] = vec3(-FLT_MAX);
        shared_shadowAABBMin[gl_LocalInvocationIndex] = vec3(FLT_MAX);
    }
    
    barrier();
    
    if (!gData.isHand) {
        #ifdef SETTING_RTWSM_B
        uint p;
        float v;
        importance(texelPos, viewZ, gData, p, v);
        backwardOutput(p, v);
        #else
        vec2 screenPos = (vec2(texelPos) + 0.5 - global_taaJitter) * global_mainImageSizeRcp;
        vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);
        vec4 scenePos = gbufferModelViewInverse * vec4(viewPos, 1.0);
        vec3 viewDir = normalize(-viewPos);
        vec4 shadowViewPos = shadowModelView * vec4(viewPos, 1.0);
        shadowAABB1(shadowViewPos.xyz);
        #endif
    }

    barrier();
    shadowAABB2();
}
