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

void importance(ivec2 texelPos, float viewZ, GBufferData gData, out uint p, out float v) {
    vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
    vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);
    vec4 scenePos = gbufferModelViewInverse * vec4(viewPos, 1.0);
    vec4 shadowViewPos = global_shadowRotationMatrix * shadowModelView * scenePos;

    float importance = SETTING_RTWSM_B_BASE;
    if (lengthSq(scenePos.xz) > pow2(shadowDistance)) {
        viewPos = vec3(0.0);
        importance = 0.0;
    }
    vec4 shadowViewPos = global_shadowRotationMatrix * shadowModelView * scenePos;
    shadowAABB1(shadowViewPos.xyz);

    vec4 shadowClipPos = global_shadowProjNext * shadowViewPos;
    vec3 shadowNDCPos = shadowClipPos.xyz / shadowClipPos.w;
    vec2 shadowScreenPos = shadowNDCPos.xy * 0.5 + 0.5;

    // Distance function
    float camDistanceSq = dot(viewPos, viewPos);
    camDistanceSq = max(4.0, camDistanceSq);
    importance *= 1.0 / (1.0 + pow(camDistanceSq, SETTING_RTWSM_B_D));

    // Surface normal function
    #if SETTING_RTWSM_B_SN > 0.0
    vec3 viewDir = normalize(-viewPos);
    importance *= 1.0 + SETTING_RTWSM_B_SN * saturate(dot(gData.geomNormal, viewDir));
    #endif

    #if SETTING_RTWSM_B_P > 0.0
    float lightDot = abs(dot(gData.geomNormal, uval_shadowLightDirView));
    lightDot = clamp(lightDot, 0.1, 0.9);
    importance *= 1.0 + (sqrt(1.0 - pow2(lightDot)) / lightDot) * SETTING_RTWSM_B_P; // tan(acos(lightDot))
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

    p = 0xFFFFFFFFu;
    v = 0u;

    if (!any(greaterThanEqual(abs(shadowClipPos.xy), vec2(1.0)))) {
        uvec2 shadowPos = uvec2(shadowScreenPos.xy * RTWSM_IMAP_SIZE);

        importance = max(importance, uval_rtwsmMin.y);

        p = shadowPos.y << 16 | shadowPos.x;
        v = importance;
    }
}

void writeOutput(uint p, float v) {
    if (p != 0xFFFFFFFFu) {
        persistent_rtwsm_importance2D_atomicMax(ivec2(p & 0xFFFFu, p >> 16), floatBitsToInt(v));
    }
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
    if (!gData.isHand) {
        uint p;
        float v;
        importance(texelPos, viewZ, gData, p, v);
        backwardOutput(p, v);
    }
}
