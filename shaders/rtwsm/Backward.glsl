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
#include "/util/Morton.glsl"

// Required resources
// layout(r32i) uniform iimage2D uimg_rtwsm_imap;

void importance(ivec2 texelPos, float viewZ, vec3 viewNormal, out uint p, out float v) {
    vec2 screenPos = (vec2(texelPos) + 0.5 - global_taaJitter) * global_mainImageSizeRcp;
    vec3 viewCoord = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);
    vec3 worldCoord = (gbufferModelViewInverse * vec4(viewCoord, 1.0)).xyz;

    vec3 viewDir = normalize(-viewCoord);

    vec4 shadowCS = global_shadowRotationMatrix * shadowProjection * shadowModelView * vec4(worldCoord, 1.0);
    shadowCS /= shadowCS.w;
    vec2 shadowTS = shadowCS.xy * 0.5 + 0.5;

    float importance = SETTING_RTWSM_B_BASE;

    // Distance function
    #if SETTING_RTWSM_B_D > 0.0
    importance *= (SETTING_RTWSM_B_D) / (SETTING_RTWSM_B_D + dot(viewCoord, viewCoord));
    #endif

    // Surface normal function
    #if SETTING_RTWSM_B_SN > 0.0
    importance *= 1.0 + SETTING_RTWSM_B_SN * saturate(dot(viewNormal, viewDir));
    #endif

    #if SETTING_RTWSM_B_P > 0.0
    float lightDir = dot(viewNormal, uval_shadowLightDirView);
    importance *= 1.0 + SETTING_RTWSM_B_P * pow(1.0 - lightDir * lightDir, float(SETTING_RTWSM_B_PP));
    #endif

    // Shadow Edge function
    #if SETTING_RTWSM_B_SE > 0.0
    vec2 shadowTSWarped = rtwsm_warpTexCoord(usam_rtwsm_imap, shadowTS);

    float center = rtwsm_linearDepth(texture(shadowtex0, shadowTSWarped).x);
    float maxDiff = 0.0;
    maxDiff = max(maxDiff, abs(center - rtwsm_linearDepth(textureOffset(shadowtex0, shadowTSWarped, ivec2(-1, -1)).x)));
    maxDiff = max(maxDiff, abs(center - rtwsm_linearDepth(textureOffset(shadowtex0, shadowTSWarped, ivec2(-1, 0)).x)));
    maxDiff = max(maxDiff, abs(center - rtwsm_linearDepth(textureOffset(shadowtex0, shadowTSWarped, ivec2(-1, 1)).x)));
    maxDiff = max(maxDiff, abs(center - rtwsm_linearDepth(textureOffset(shadowtex0, shadowTSWarped, ivec2(0, -1)).x)));
    maxDiff = max(maxDiff, abs(center - rtwsm_linearDepth(textureOffset(shadowtex0, shadowTSWarped, ivec2(0, 1)).x)));
    maxDiff = max(maxDiff, abs(center - rtwsm_linearDepth(textureOffset(shadowtex0, shadowTSWarped, ivec2(1, -1)).x)));
    maxDiff = max(maxDiff, abs(center - rtwsm_linearDepth(textureOffset(shadowtex0, shadowTSWarped, ivec2(1, 0)).x)));
    maxDiff = max(maxDiff, abs(center - rtwsm_linearDepth(textureOffset(shadowtex0, shadowTSWarped, ivec2(1, 1)).x)));

    importance *= 1.0 + SETTING_RTWSM_B_SE * linearStep(0.5, 4.0, maxDiff);
    #endif

    if (any(greaterThanEqual(abs(shadowCS.xy), vec2(1.0)))) {
        p = 0xFFFFFFFFu;
        v = 0u;
        return;
    }

    uvec2 shadowPos = uvec2(shadowTS.xy * SETTING_RTWSM_IMAP_SIZE + 0.5);

    importance = max(importance, uval_rtwsmMin.y);

    p = shadowPos.y << 16 | shadowPos.x;
    v = importance;

    return;
}

void writeOutput(uint p, float v) {
    if (p == 0xFFFFFFFFu) return;
    imageAtomicMax(uimg_rtwsm_imap, ivec2(p & 0xFFFFu, p >> 16), floatBitsToInt(v));
}

void rtwsm_backward(ivec2 texelPos, float viewZ, vec3 viewNormal) {
    uint p;
    float v;
    importance(texelPos, viewZ, viewNormal, p, v);

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
