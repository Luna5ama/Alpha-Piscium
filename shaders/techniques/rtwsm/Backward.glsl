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
// layout(r32i) uniform iimage2D uimg_fr32f;

shared vec3 shared_shadowAABBMin[16];
shared vec3 shared_shadowAABBMax[16];

void shadowAABB1(vec3 shadowViewPos) {
    vec3 min1 = subgroupMin(shadowViewPos);
    vec3 max1 = subgroupMax(shadowViewPos);

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
            ivec3 min4 = ivec3(floor(min3 / 16.0)) * 16;
            ivec3 max4 = ivec3(ceil(max3 / 16.0)) * 16;
            atomicMin(global_shadowAABBMinNew.x, min4.x);
            atomicMin(global_shadowAABBMinNew.y, min4.y);
            atomicMin(global_shadowAABBMinNew.z, min4.z);
            atomicMax(global_shadowAABBMaxNew.x, max4.x);
            atomicMax(global_shadowAABBMaxNew.y, max4.y);
            atomicMax(global_shadowAABBMaxNew.z, max4.z);
        }
    }
}

void importance(ivec2 texelPos, float viewZ, GBufferData gData, out uint p, out float v) {
    vec2 screenPos = (vec2(texelPos) + 0.5 - global_taaJitter) * uval_mainImageSizeRcp;
    vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);
    float importance = SETTING_RTWSM_B_BASE;
    if (dot(viewPos, viewPos) > shadowDistance * shadowDistance) {
        viewPos = vec3(0.0);
        importance = 0.0;
    }
    vec4 scenePos = gbufferModelViewInverse * vec4(viewPos, 1.0);
    vec4 shadowViewPos = global_shadowRotationMatrix * shadowModelView * scenePos;
    shadowAABB1(shadowViewPos.xyz);

    vec4 shadowClipPos = global_shadowProj * shadowViewPos;
    vec3 shadowNDCPos = shadowClipPos.xyz / shadowClipPos.w;
    vec2 shadowScreenPos = shadowNDCPos.xy * 0.5 + 0.5;

    // Distance function
    importance *= 1.0 / (1.0 + pow(dot(viewPos, viewPos), SETTING_RTWSM_B_D));

    // Surface normal function
    #if SETTING_RTWSM_B_SN > 0.0
    vec3 viewDir = normalize(-viewPos);
    importance *= 1.0 + SETTING_RTWSM_B_SN * saturate(dot(gData.geomNormal, viewDir));
    #endif

    #if SETTING_RTWSM_B_P > 0.0
    float lightDir = dot(gData.geomNormal, uval_shadowLightDirView);
    importance *= 1.0 + SETTING_RTWSM_B_P * pow(1.0 - lightDir * lightDir, float(SETTING_RTWSM_B_PP));
    #endif

    // Shadow Edge function
    #if SETTING_RTWSM_B_SE > 0.0
    vec2 shadowScreenPosWarped = rtwsm_warpTexCoord(shadowScreenPos);

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

    importance *= 1.0 + SETTING_RTWSM_B_SE * linearStep(0.5, 2.0, maxDiff) * 100.0;
    #endif

    if (any(greaterThanEqual(abs(shadowClipPos.xy), vec2(1.0)))) {
        p = 0xFFFFFFFFu;
        v = 0u;
        return;
    }

    uvec2 shadowPos = uvec2(shadowScreenPos.xy * RTWSM_IMAP_SIZE);

    importance = max(importance, uval_rtwsmMin.y);

    p = shadowPos.y << 16 | shadowPos.x;
    v = importance;

    return;
}

void writeOutput(uint p, float v) {
    if (p == 0xFFFFFFFFu) return;
    persistent_rtwsm_importance2D_atomicMax(ivec2(p & 0xFFFFu, p >> 16), floatBitsToInt(v));
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
        shared_shadowAABBMax[gl_LocalInvocationIndex] = vec3(0.0);
        shared_shadowAABBMin[gl_LocalInvocationIndex] = vec3(0.0);
    }
    
    barrier();
    
    if (!gData.isHand) {
        #ifdef SETTING_RTWSM_B
        uint p;
        float v;
        importance(texelPos, viewZ, gData, p, v);
        backwardOutput(p, v);
        #else
        vec2 screenPos = (vec2(texelPos) + 0.5 - global_taaJitter) * uval_mainImageSizeRcp;
        vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);
        vec4 scenePos = gbufferModelViewInverse * vec4(viewPos, 1.0);
        vec4 shadowViewPos = global_shadowRotationMatrix * shadowModelView * scenePos;
        shadowAABB1(shadowViewPos.xyz);
        #endif
    }

    barrier();
    shadowAABB2();
}
