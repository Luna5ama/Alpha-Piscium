/*
    References:
        [ROS12] Rosen, Paul. "Rectilinear Texture Warping for Fast Adaptive Shadow Mapping". 2012.
            https://www.cspaul.com/publications/Rosen.2012.I3D.pdf
        [MYE21] Myers, Kevin. "Shadows of Cold War: A scalable approach to shadowing". 2021.
            https://research.activision.com/publications/2021/10/shadows-of-cold-war--a-scalable-approach-to-shadowing
*/
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
layout(rgba32ui) uniform restrict writeonly uimage2D uimg_rgba32ui;

#include "/techniques/rtwsm/Backward.glsl"

// Shared memory for RTWSM warp map cache
shared float shared_warpTexelX[RTWSM_IMAP_SIZE];
shared float shared_warpTexelY[RTWSM_IMAP_SIZE];

vec2 rtwsm_warpTexCoord_shared(vec2 uv) {
    vec2 texelPos = fma(uv, vec2(float(RTWSM_IMAP_SIZE)), vec2(-0.5));
    ivec2 t1 = max(ivec2(texelPos), 0);
    ivec2 t2 = min(t1 + 1, RTWSM_IMAP_SIZE - 1);
    vec2 w = fract(texelPos);

    return uv + vec2(
        mix(shared_warpTexelX[t1.x], shared_warpTexelX[t2.x], w.x),
        mix(shared_warpTexelY[t1.y], shared_warpTexelY[t2.y], w.y)
    );
}

float searchBlocker(ivec2 texelPos, vec3 shadowTexCoord, float sssFactor) {
    uint BLOCKER_SEARCH_N = uint(mix(float(SETTING_PCSS_BLOCKER_SEARCH_COUNT), float(SETTING_SSS_SAMPLE_COUNT), float(sssFactor > 0.0)));

    vec2 blockerSearchRange = fma(vec2(sssFactor), vec2(0.2), vec2(0.05)) * vec2(global_shadowProj[0][0], global_shadowProj[1][1]);

    float blockerDepthSum = 0.0;
    float validCount = 0.0;

    float jitterR = rand_stbnVec1(texelPos, frameCounter);
    vec2 dir = rand_stbnUnitVec211(texelPos, frameCounter);

    float rcpSamples = 1.0 / float(BLOCKER_SEARCH_N);
    float jitterRcpSamples = jitterR * rcpSamples;

    // Loop-invariant weight exponentiation calculation
    float currentWeight = exp2(-2.0 * jitterRcpSamples);
    float weightMult = exp2(-2.0 * rcpSamples);

    for (uint i = 0u; i < BLOCKER_SEARCH_N; i++) {
        dir *= MAT2_GOLDEN_ANGLE;
        float baseRadius = sqrt(fma(float(i), rcpSamples, jitterRcpSamples));

        vec3 sampleTexCoord = shadowTexCoord;
        sampleTexCoord.xy = fma(dir * baseRadius, blockerSearchRange, sampleTexCoord.xy);
        sampleTexCoord.xy = rtwsm_warpTexCoord_shared(sampleTexCoord.xy);

        vec4 depthGather = textureGather(shadowtex1, sampleTexCoord.xy, 0);
        vec4 isBlocker4 = vec4(greaterThan(vec4(sampleTexCoord.z), depthGather));

        validCount += dot(vec4(currentWeight), isBlocker4);
        blockerDepthSum += dot(vec4(depthGather), isBlocker4) * currentWeight;

        currentWeight *= weightMult;
    }

    if (validCount > 0.0) {
        blockerDepthSum /= validCount;
    } else {
        blockerDepthSum = shadowTexCoord.z;
    }

    return max(rtwsm_linearDepthOffset(shadowTexCoord.z - blockerDepthSum), 0.0);
}

// Insprired by [MYE21]
float shadowHarden(float x, float b) {
    float x2 = fma(x, 2.0, -1.0);
    return fma(sign(x2), fma(pow(1.0 - abs(x2), b), -0.5, 0.5), 0.5);
}

vec4 compShadow(ivec2 texelPos, float viewZ, GBufferData gData) {
    Material material = material_decode(gData);

    float sssFactor = material.sss;
    uint skipFlag = uint(dot(gData.normal, uval_upDirView) < -0.99);
    skipFlag &= uint(sssFactor < 0.001);
    vec4 result = vec4(0.0);
    if (!bool(skipFlag)) {
        vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
        vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);

        float cosLightTheta = abs(dot(uval_shadowLightDirView, gData.geomNormal));
        vec3 offsetViewPos = viewPos + gData.geomNormal * mix(0.03, 0.01, pow2(cosLightTheta));

        vec4 scenePos = gbufferModelViewInverse * vec4(offsetViewPos, 1.0);
        vec4 shadowViewPos = global_shadowRotationMatrix * global_shadowView * scenePos;
        vec4 shadowClipPos = global_shadowProj * shadowViewPos;
        vec3 shadowNDCPos = shadowClipPos.xyz / shadowClipPos.w;
        vec3 shadowScreenPos = shadowNDCPos * 0.5 + 0.5;

        float blockerDistance = searchBlocker(texelPos, shadowScreenPos, sssFactor);
        float ssRange = 0.0;

        #if SETTING_PCSS_BPF > 0
        ssRange += exp2(SETTING_PCSS_BPF - 10.0);
        ssRange = mix(ssRange, ssRange + 0.05, gData.isHand);
        #endif

        float clampedBlockerDistance = softMax(blockerDistance, 0.5, 8.0);
        ssRange = saturate(fma(SUN_ANGULAR_RADIUS * 2.0 * SETTING_PCSS_VPF, clampedBlockerDistance, ssRange));
        ssRange += sssFactor * SETTING_SSS_DIFFUSE_RANGE;

        // Pre-calculated projection scaling
        vec2 shadowProjScale = vec2(global_shadowProj[0][0], global_shadowProj[1][1]) * 0.25;
        vec2 ssRange2 = ssRange * shadowProjScale;

        float jitterR = rand_stbnVec1(texelPos, frameCounter);
        vec2 dir = rand_stbnUnitVec211(texelPos, frameCounter);

        const uint SAMPLE_COUNT = SETTING_PCSS_SAMPLE_COUNT;
        float rcpSamples = 1.0 / float(SAMPLE_COUNT);
        float jitterRcpSamples = jitterR * rcpSamples;

        float16_t solidShadowSum = float16_t(0.0);
        f16vec4 translucentShadowSum = f16vec4(0.0);

        #ifdef SETTING_WATER_CAUSTICS
        vec2 texelPosCenter = vec2(texelPos) + 0.5;
        float causticsSampleRadius = 32.0 / max(abs(viewPos.z), 0.1);
        #endif

        shadowScreenPos.z -= rtwsm_linearDepthOffsetInverse(jitterR * pow(sssFactor, 0.25) * SETTING_SSS_DEPTH_RANGE);

        for (uint i = 0; i < SAMPLE_COUNT; i++) {
            dir *= MAT2_GOLDEN_ANGLE;
            float baseRadius = sqrt(fma(float(i), rcpSamples, jitterRcpSamples));
            vec2 baseOffset = dir * baseRadius;

            vec3 sampleTexCoord = shadowScreenPos;
            sampleTexCoord.xy = fma(ssRange2, baseOffset, sampleTexCoord.xy);
            sampleTexCoord.xy = rtwsm_warpTexCoord_shared(sampleTexCoord.xy);

            vec4 sampleShadowDepthOffset4 = textureGather(shadowcolor0, sampleTexCoord.xy, 0);
            sampleTexCoord.z -= max4(abs(sampleShadowDepthOffset4));

            float shadowSampleSolid = rtwsm_sampleShadowDepth(shadowtex1HW, sampleTexCoord, 0.0);
            solidShadowSum += float16_t(shadowSampleSolid);

            // Bitwise optimization kept, simplified structural checking logic
            bool cond = (or4(floatBitsToUint(sampleShadowDepthOffset4)) & 0x80000000u) != 0u && floatBitsToUint(shadowSampleSolid) > 0u;

            if (cond) {
                vec4 shadowDepthAll = textureGather(shadowtex0, sampleTexCoord.xy, 0);
                if (any(greaterThan(vec4(sampleTexCoord.z), shadowDepthAll))) {
                    vec4 waterMask4 = textureGather(usam_shadow_waterMask, sampleTexCoord.xy, 0);
                    float waterMaskSum = sum4(waterMask4);
                    f16vec3 sampleColor = f16vec3(textureLod(shadowcolor2, sampleTexCoord.xy, 0.0).rgb);

                    if (waterMaskSum > 0.1) {
                        vec4 translucentDistance = saturate(sampleTexCoord.z - shadowDepthAll);
                        float translucentDistanceMasked = dot(translucentDistance, waterMask4) / waterMaskSum;
                        float waterDepth = max(rtwsm_linearDepthOffset(translucentDistanceMasked), 0.0);
                        sampleColor *= f16vec3(exp(-waterDepth * WATER_EXTINCTION));

                        #ifdef SETTING_WATER_CAUSTICS
                        vec2 causticsTexelPos = texelPosCenter + baseOffset * causticsSampleRadius;
                        float caustics = transient_caustics_final_sample(causticsTexelPos * uval_mainImageSizeRcp).r;
                        sampleColor *= float16_t(mix(1.0, caustics, pow2(linearStep(0.0, 4.0, waterDepth))));
                        #endif
                    }
                    translucentShadowSum += f16vec4(sampleColor, 1.0);
                }
            }
        }

        float bendShadow = transient_bendShadow_fetch(texelPos).r;
        float solidShadow = float(solidShadowSum) * rcpSamples;
        float w = rcp(fma(blockerDistance, 0.5, 0.0001)) + 1.0;
        float sh = shadowHarden(solidShadow, w);
        solidShadow = sh * (2.0 - sh);

        if (sssFactor < 0.001) {
            solidShadow = min(bendShadow, solidShadow);
        }

        vec3 finalShadow = vec3(solidShadow);
        if (translucentShadowSum.a > float16_t(0.0)) {
            finalShadow *= vec3(translucentShadowSum.rgb) * rcp(float(translucentShadowSum.a));
        }

        float surfaceDepth = sssFactor > 0.0 ? max(blockerDistance, 0.1) : 0.0;
        float shadowRangeBlend = smoothstep(shadowDistance - 16.0, shadowDistance - 8.0, length(scenePos.xz));

        result = mix(vec4(finalShadow, surfaceDepth), vec4(bendShadow.rrr, 1.0), shadowRangeBlend);
    }
    return result;
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
    ivec2 texelPos = ivec2(workGroupOrigin + morton_8bDecode(threadIdx));

    if (all(lessThan(texelPos, uval_mainImageSizeI))) {
        transient_lowCloudRender_store(texelPos, uvec4(0u));
        transient_lowCloudAccumulated_store(texelPos, uvec4(0u));

        float viewZ = hiz_groupGroundCheckSubgroupLoadViewZ(swizzledWGPos.xy, 4, texelPos);

        if (viewZ > -65536.0) {
            GBufferData gData = gbufferData_init();
            gbufferData1_unpack(texelFetch(usam_gbufferData1, texelPos, 0), gData);
            gbufferData2_unpack(texelFetch(usam_gbufferData2, texelPos, 0), gData);
            rtwsm_backward(texelPos, viewZ, gData);

            vec4 shadowValue = compShadow(texelPos, viewZ, gData);
            shadowValue = clamp(shadowValue, 0.0, FP16_MAX);
            transient_shadow_store(texelPos, shadowValue);
        }
    }
}