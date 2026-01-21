#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_vote : enable
#extension GL_KHR_shader_subgroup_clustered : enable
#extension GL_KHR_shader_subgroup_ballot : enable
#define GLOBAL_DATA_MODIFIER buffer

#include "/techniques/atmospherics/water/Constants.glsl"
#include "/techniques/HiZCheck.glsl"
#include "/util/Celestial.glsl"
#include "/util/Material.glsl"
#include "/util/Hash.glsl"
#include "/util/Mat2.glsl"
#include "/util/Rand.glsl"
#include "/util/GBufferData.glsl"
#include "/util/ThreadGroupTiling.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

#ifdef DISTANT_HORIZONS
uniform sampler2D dhDepthTex0;
#endif

layout(rgba16f) uniform restrict image2D uimg_rgba16f;
layout(r32i) uniform iimage2D uimg_fr32f;
layout(rgba16f) uniform restrict image2D uimg_translucentColor;

#include "/techniques/rtwsm/Backward.glsl"

// Shared memory for RTWSM warp map cache
shared float shared_warpTexelX[RTWSM_IMAP_SIZE];
shared float shared_warpTexelY[RTWSM_IMAP_SIZE];

vec2 texel2Screen(ivec2 texelPos) {
    return (vec2(texelPos) + 0.5) * uval_mainImageSizeRcp;
}

void shared_bilinearSampleParam(float coord, out int texelPos1, out int texelPos2, out float bilinearWeight) {
    float texelPos = coord * float(RTWSM_IMAP_SIZE) - 0.5;
    texelPos1 = int(texelPos);
    bilinearWeight = texelPos - float(texelPos1);
    texelPos2 = texelPos1 + 1;
    texelPos1 = max(texelPos1, 0);
    texelPos2 = min(texelPos2, RTWSM_IMAP_SIZE - 1);
}

vec2 rtwsm_warpTexCoord_shared(vec2 uv) {
    vec2 result = uv;
    int texelPos1, texelPos2;
    float bilinearWeight;
    shared_bilinearSampleParam(uv.x, texelPos1, texelPos2, bilinearWeight);
    result.x += mix(shared_warpTexelX[texelPos1], shared_warpTexelX[texelPos2], bilinearWeight);
    shared_bilinearSampleParam(uv.y, texelPos1, texelPos2, bilinearWeight);
    result.y += mix(shared_warpTexelY[texelPos1], shared_warpTexelY[texelPos2], bilinearWeight);
    return result;
}

float searchBlocker(ivec2 texelPos, vec3 shadowTexCoord) {
    #define BLOCKER_SEARCH_N SETTING_PCSS_BLOCKER_SEARCH_COUNT

    vec2 blockerSearchRange = 0.1 * vec2(global_shadowProjPrev[0][0], global_shadowProjPrev[1][1]);

    float blockerDepthSum = 0.0;
    float validCount = 0.0;

    vec2 stbnPos = texelPos + ivec2(6, 9);
    float jitterR = rand_stbnVec1(texelPos, frameCounter);
    vec2 dir = rand_stbnUnitVec211(texelPos, frameCounter);
    float rcpSamples = 1.0 / float(BLOCKER_SEARCH_N);

    for (int i = 0; i < BLOCKER_SEARCH_N; i++) {
        dir *= MAT2_GOLDEN_ANGLE;
        float baseRadius = sqrt((float(i) + jitterR) * rcpSamples);
        vec3 sampleTexCoord = shadowTexCoord;

        sampleTexCoord.xy += dir * baseRadius * blockerSearchRange;
        sampleTexCoord.xy = rtwsm_warpTexCoord_shared(sampleTexCoord.xy);
        vec4 depthGather = textureGather(shadowtex1, sampleTexCoord.xy, 0);
        vec4 isBlocker4 = vec4(greaterThan(vec4(sampleTexCoord.z), depthGather));
        validCount += dot(vec4(1.0), isBlocker4);
        blockerDepthSum += dot(vec4(depthGather), isBlocker4);
    }

    blockerDepthSum /= max(validCount, 1.0);
    blockerDepthSum = mix(shadowTexCoord.z, blockerDepthSum, float(validCount > 0.0));

    return abs(rtwsm_linearDepth(blockerDepthSum) - rtwsm_linearDepth(shadowTexCoord.z));
}

vec3 compShadow(ivec2 texelPos, float viewZ, GBufferData gData) {
    vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
    Material material = material_decode(gData);
    vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);

    float sssFactor = material.sss;
    uint skipFlag = uint(dot(gData.normal, uval_upDirView) < -0.99);
    skipFlag &= uint(sssFactor < 0.001);
    if (bool(skipFlag)) {
        return vec3(0.0);
    }

    float cosLightTheta = abs(dot(uval_shadowLightDirView, gData.geomNormal));

    vec3 offsetViewPos = viewPos;
    offsetViewPos += gData.geomNormal * mix(0.03, 0.01, pow2(cosLightTheta));
    vec4 scenePos = gbufferModelViewInverse * vec4(offsetViewPos, 1.0);
    vec4 shadowViewPos = global_shadowRotationMatrix * global_shadowView * scenePos;
    vec4 shadowClipPos = global_shadowProjPrev * shadowViewPos;
    vec3 shadowNDCPos = shadowClipPos.xyz / shadowClipPos.w;
    vec3 shadowScreenPos = shadowNDCPos * 0.5 + 0.5;
    float blockerDistance = searchBlocker(texelPos, shadowScreenPos);

    float ssRange = 0.0;
    #if SETTING_PCSS_BPF > 0
    ssRange += exp2(SETTING_PCSS_BPF - 10.0);
    ssRange = mix(ssRange, ssRange + 0.05, gData.isHand);
    #endif
    ssRange += SUN_ANGULAR_RADIUS * 2.0 * SETTING_PCSS_VPF * blockerDistance;
    ssRange = saturate(ssRange);
    ssRange += sssFactor * SETTING_SSS_DIFFUSE_RANGE;

    const float ssRangeMul = 0.5;
    ssRange *= ssRangeMul;

    float jitterR = rand_stbnVec1(texelPos, frameCounter);
    vec2 dir = rand_stbnUnitVec211(texelPos, frameCounter);

    const uint SAMPLE_COUNT = SETTING_PCSS_SAMPLE_COUNT;
    float rcpSamples = 1.0 / float(SAMPLE_COUNT);

    vec3 shadowSum = vec3(0.0);
    #ifdef SETTING_WATER_CAUSTICS
    vec2 texelPosCenter = vec2(texelPos) + 0.5;
    float causticsSampleRadius = 32.0 / max(abs(viewPos.z), 0.1);

    if (isEyeInWater == 1) {
        for (uint i = 0; i < SAMPLE_COUNT; i++) {
            dir *= MAT2_GOLDEN_ANGLE;
            float baseRadius = sqrt((float(i) + jitterR) * rcpSamples);
            vec2 baseOffset = dir * baseRadius;
            vec3 sampleTexCoord = shadowScreenPos;

            sampleTexCoord.xy += ssRange * baseOffset * vec2(global_shadowProjPrev[0][0], global_shadowProjPrev[1][1]);
            sampleTexCoord.xy = rtwsm_warpTexCoord_shared(sampleTexCoord.xy);

            vec4 sampleShadowDepthOffset4 = textureGather(shadowcolor0, sampleTexCoord.xy, 0);
            sampleTexCoord.z -= max4(abs(sampleShadowDepthOffset4));

            float shadowSampleSolid = rtwsm_sampleShadowDepth(shadowtex1HW, sampleTexCoord, 0.0);
            vec3 sampleShadow = vec3(shadowSampleSolid);

            if (shadowSampleSolid > 0.0 && any(lessThan(sampleShadowDepthOffset4, vec4(0.0)))) {
                vec4 shadowDepthAll = textureGather(shadowtex0, sampleTexCoord.xy, 0);
                bvec4 shadowSampleCompareAll = greaterThan(vec4(sampleTexCoord.z), shadowDepthAll);
                if (any(shadowSampleCompareAll)) {
                    vec4 waterMask4 = textureGather(usam_shadow_waterMask, sampleTexCoord.xy, 0);
                    float waterMaskSum = sum4(waterMask4);
                    vec3 sampleColor = rtwsm_sampleShadowColor(shadowcolor2, sampleTexCoord.xy, 0.0).rgb;
                    if (waterMaskSum > 0.1) {
                        vec4 translucentDistance = saturate(sampleTexCoord.z - shadowDepthAll);
                        float translucentDistanceMasked = dot(translucentDistance, waterMask4) / waterMaskSum;
                        float waterDepth = max(rtwsm_linearDepthOffset(translucentDistanceMasked), 0.0);

                        sampleColor *= exp(-waterDepth * WATER_EXTINCTION);
                        vec2 causticsTexelPos = texelPosCenter + baseOffset * causticsSampleRadius;
                        float caustics = transient_caustics_final_sample(causticsTexelPos * uval_mainImageSizeRcp).r;
                        sampleColor *= mix(1.0, caustics, pow2(linearStep(0.0, 4.0, waterDepth)));
                    }

                    sampleShadow *= sampleColor;
                }
            }

            shadowSum += sampleShadow;
        }
    } else {
        vec2 causticsTexelPos = texelPosCenter + dir * jitterR * causticsSampleRadius;
        float caustics = transient_caustics_final_sample(causticsTexelPos * uval_mainImageSizeRcp).r;

        for (uint i = 0; i < SAMPLE_COUNT; i++) {
            dir *= MAT2_GOLDEN_ANGLE;
            float baseRadius = sqrt((float(i) + jitterR) * rcpSamples);
            vec2 baseOffset = dir * baseRadius;
            vec3 sampleTexCoord = shadowScreenPos;

            sampleTexCoord.xy += ssRange * baseOffset * vec2(global_shadowProjPrev[0][0], global_shadowProjPrev[1][1]);
            sampleTexCoord.xy = rtwsm_warpTexCoord_shared(sampleTexCoord.xy);

            vec4 sampleShadowDepthOffset4 = textureGather(shadowcolor0, sampleTexCoord.xy, 0);
            sampleTexCoord.z -= sum4(abs(sampleShadowDepthOffset4)) * 0.25;

            float shadowSampleSolid = rtwsm_sampleShadowDepth(shadowtex1HW, sampleTexCoord, 0.0);
            vec3 sampleShadow = vec3(shadowSampleSolid);

            if (shadowSampleSolid > 0.0 && any(lessThan(sampleShadowDepthOffset4, vec4(0.0)))) {
                vec4 shadowDepthAll = textureGather(shadowtex0, sampleTexCoord.xy, 0);
                bvec4 shadowSampleCompareAll = greaterThan(vec4(sampleTexCoord.z), shadowDepthAll);
                if (any(shadowSampleCompareAll)) {
                    vec4 waterMask4 = textureGather(usam_shadow_waterMask, sampleTexCoord.xy, 0);
                    float waterMaskSum = sum4(waterMask4);
                    vec3 sampleColor = rtwsm_sampleShadowColor(shadowcolor2, sampleTexCoord.xy, 0.0).rgb;
                    if (waterMaskSum > 0.1) {
                        vec4 translucentDistance = saturate(sampleTexCoord.z - shadowDepthAll);
                        float translucentDistanceMasked = dot(translucentDistance, waterMask4) / waterMaskSum;
                        float waterDepth = max(rtwsm_linearDepthOffset(translucentDistanceMasked), 0.0);

                        sampleColor *= exp(-waterDepth * WATER_EXTINCTION);
                        sampleColor *= mix(1.0, caustics, pow2(linearStep(0.0, 4.0, waterDepth)));
                    }
                    sampleShadow *= sampleColor;
                }
            }

            shadowSum += sampleShadow;
        }
    }
    #else
    for (uint i = 0; i < SAMPLE_COUNT; i++) {
        dir *= MAT2_GOLDEN_ANGLE;
        float baseRadius = sqrt((float(i) + jitterR) * rcpSamples);
        vec2 baseOffset = dir * baseRadius;
        vec3 sampleTexCoord = shadowScreenPos;

        sampleTexCoord.xy += ssRange * baseOffset * vec2(global_shadowProjPrev[0][0], global_shadowProjPrev[1][1]);
        sampleTexCoord.xy = rtwsm_warpTexCoord_shared(sampleTexCoord.xy);

        vec4 sampleShadowDepthOffset4 = textureGather(shadowcolor0, sampleTexCoord.xy, 0);
        sampleTexCoord.z -= sum4(abs(sampleShadowDepthOffset4)) * 0.25;

        float shadowSampleSolid = rtwsm_sampleShadowDepth(shadowtex1HW, sampleTexCoord, 0.0);
        vec3 sampleShadow = vec3(shadowSampleSolid);

        if (shadowSampleSolid > 0.0 && any(lessThan(sampleShadowDepthOffset4, vec4(0.0)))) {
            vec4 shadowDepthAll = textureGather(shadowtex0, sampleTexCoord.xy, 0);
            bvec4 shadowSampleCompareAll = greaterThan(vec4(sampleTexCoord.z), shadowDepthAll);
            if (any(shadowSampleCompareAll)) {
                vec4 waterMask4 = textureGather(usam_shadow_waterMask, sampleTexCoord.xy, 0);
                float waterMaskSum = sum4(waterMask4);
                vec3 sampleColor = rtwsm_sampleShadowColor(shadowcolor2, sampleTexCoord.xy, 0.0).rgb;
                if (waterMaskSum > 0.1) {
                    vec4 translucentDistance = saturate(sampleTexCoord.z - shadowDepthAll);
                    float translucentDistanceMasked = dot(translucentDistance, waterMask4) / waterMaskSum;
                    float waterDepth = max(rtwsm_linearDepthOffset(translucentDistanceMasked), 0.0);
                    sampleColor *= exp(-waterDepth * WATER_EXTINCTION);
                }
                sampleShadow *= sampleColor;
            }
        }
        shadowSum += sampleShadow;
    }
    #endif

    shadowSum *= rcpSamples;

    float shadowRangeBlend = linearStep(shadowDistance - 8.0, shadowDistance, length(scenePos.xz));
    return mix(vec3(shadowSum), vec3(1.0), shadowRangeBlend);
}

void main() {
    uint localThreadIdx = gl_LocalInvocationIndex;
    shared_warpTexelX[localThreadIdx] = persistent_rtwsm_warp_fetch(ivec2(localThreadIdx, 0)).r;
    shared_warpTexelY[localThreadIdx] = persistent_rtwsm_warp_fetch(ivec2(localThreadIdx, 1)).r;
    barrier();

    uint workGroupIdx = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uvec2 swizzledWGPos = ssbo_threadGroupTiling[workGroupIdx];
    uvec2 workGroupOrigin = swizzledWGPos << 4u;
    uint threadIdx = gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
    uvec2 mortonPos = morton_8bDecode(threadIdx);
    uvec2 mortonGlobalPosU = workGroupOrigin + mortonPos;
    ivec2 texelPos = ivec2(mortonGlobalPosU);

    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        float viewZ = hiz_groupGroundCheckSubgroupLoadViewZ(swizzledWGPos.xy, 4, texelPos);

        if (viewZ > -65536.0) {
            GBufferData gData = gbufferData_init();
            gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
            gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);
            rtwsm_backward(texelPos, viewZ, gData);
            vec3 shadowValue = compShadow(texelPos, viewZ, gData);
            shadowValue = clamp(shadowValue, 0.0, FP16_MAX);
            vec4 outputColor = vec4(shadowValue, 1.0);
            transient_shadow_store(texelPos, outputColor);
        }
    }
}
