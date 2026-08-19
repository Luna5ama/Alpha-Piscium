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
layout(r32f) uniform restrict image2D uimg_r32f;

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

float searchBlocker(vec3 shadowTexCoord, float sssFactor, vec2 shadowProjScale, float jitterR, vec2 dir, uint sampleCount, float shadowDepthRange) {
    vec2 blockerSearchRange = shadowProjScale * fma(sssFactor, 0.2, 0.05);

    float blockerDepthSum = 0.0;
    float validCount = 0.0;

    float rcpSamples = 1.0 / float(sampleCount);
    float jitterRcpSamples = jitterR * rcpSamples;

    // Loop-invariant weight exponentiation calculation
    float currentWeight = exp2(-2.0 * jitterRcpSamples);
    float weightMult = exp2(-2.0 * rcpSamples);
    float radius2 = jitterRcpSamples;

    for (uint i = 0u; i < sampleCount; i++) {
        dir *= MAT2_GOLDEN_ANGLE;
        float baseRadius = sqrt(radius2);
        radius2 += rcpSamples;

        vec3 sampleTexCoord = shadowTexCoord;
        sampleTexCoord.xy = fma(dir * baseRadius, blockerSearchRange, sampleTexCoord.xy);
        sampleTexCoord.xy = rtwsm_warpTexCoord_shared(sampleTexCoord.xy);

        vec4 depthGather = textureGather(shadowtex1, sampleTexCoord.xy, 0);
        vec4 isBlocker4 = vec4(greaterThan(vec4(shadowTexCoord.z), depthGather));

        validCount += sum4(isBlocker4) * currentWeight;
        blockerDepthSum += dot(depthGather, isBlocker4) * currentWeight;

        currentWeight *= weightMult;
    }

    if (validCount > 0.0) {
        blockerDepthSum /= validCount;
    } else {
        blockerDepthSum = shadowTexCoord.z;
    }

    return max((shadowTexCoord.z - blockerDepthSum) * shadowDepthRange, 0.0);
}

// Insprired by [MYE21]
float shadowHarden(float x, float b) {
    float x2 = fma(x, 2.0, -1.0);
    return fma(sign(x2), fma(pow(1.0 - abs(x2), b), -0.5, 0.5), 0.5);
}

vec4 compShadow(ivec2 texelPos, float viewZ, GBufferData gData) {
    float sssFactor = material_decodeSSS(gData.materialID, gData.pbrSpecular.b, gData.forceBuiltInPBR);
    bool hasSSS = sssFactor > 0.0;
    bool isSSS = sssFactor > 0.001;
    if (dot(gData.normal, uval_upDirView) < -0.99 && !isSSS) {
        return vec4(0.0);
    }
    vec2 screenPos = coords_texelToUV(texelPos, uval_mainImageSizeRcp);
    vec3 viewPos = coords_toViewCoord(screenPos, viewZ, global_camProjInverse);

    float cosLightTheta = abs(dot(uval_shadowLightDirView, gData.geomNormal));
    vec3 offsetViewPos = viewPos + gData.geomNormal * mix(0.03, 0.01, pow2(cosLightTheta));

    vec4 scenePos = gbufferModelViewInverse * vec4(offsetViewPos, 1.0);
    float shadowRangeStart = min(shadowDistance, far) - 16.0;
    float shadowRangeEnd = shadowRangeStart + 8.0;
    float sceneDistance2 = dot(scenePos.xz, scenePos.xz);
    float bendShadow = transient_bendShadow_fetch(texelPos).r;
    if (sceneDistance2 >= pow2(shadowRangeEnd)) {
        return vec4(bendShadow.rrr, 1.0);
    }
    if (!hasSSS && bendShadow == 0.0 && sceneDistance2 <= pow2(shadowRangeStart)) {
        return vec4(0.0);
    }
    vec4 shadowClipPos = global_sceneToShadowNDC * scenePos;
    vec3 shadowScreenPos = shadowClipPos.xyz * 0.5 + 0.5;

    vec2 shadowProjScale = vec2(global_shadowProj[0][0], global_shadowProj[1][1]);
    vec2 stbn = rand_stbnVec2(texelPos, frameCounter);
    float jitterR = stbn.x;
    float sampleAngle = stbn.y * PI_2;
    vec2 dir = vec2(cos(sampleAngle), sin(sampleAngle));
    float shadowDepthRange = rtwsm_linearDepthOffset(1.0);
    float blockerDistance;
    if (hasSSS) {
        blockerDistance = searchBlocker(shadowScreenPos, sssFactor, shadowProjScale, jitterR, dir, uint(SETTING_SSS_SAMPLE_COUNT), shadowDepthRange);
    } else {
        blockerDistance = searchBlocker(shadowScreenPos, sssFactor, shadowProjScale, jitterR, dir, uint(SETTING_PCSS_BLOCKER_SEARCH_COUNT), shadowDepthRange);
    }
    float ssRange = 0.0;

    #if SETTING_PCSS_BPF > 0
    ssRange += exp2(SETTING_PCSS_BPF - 10.0);
    ssRange = mix(ssRange, ssRange + 0.05, gData.isHand);
    #endif

    float clampedBlockerDistance = softMax(blockerDistance, 0.5, 8.0);
    ssRange = saturate(fma(SUN_ANGULAR_RADIUS * 2.0 * SETTING_PCSS_VPF, clampedBlockerDistance, ssRange));
    ssRange += sssFactor * SETTING_SSS_DIFFUSE_RANGE;

    vec2 ssRange2 = ssRange * shadowProjScale * 0.25;

    const uint SAMPLE_COUNT = SETTING_PCSS_SAMPLE_COUNT;
    float rcpSamples = 1.0 / float(SAMPLE_COUNT);
    float jitterRcpSamples = jitterR * rcpSamples;

    float solidShadowSum = 0.0;
    f16vec4 translucentShadowSum = f16vec4(0.0);

    #ifdef SETTING_WATER_CAUSTICS
    vec2 texelPosCenter = vec2(texelPos) + 0.5;
    float causticsSampleRadius = 32.0 / max(abs(viewPos.z), 0.1);
    #endif

    if (hasSSS) {
        shadowScreenPos.z -= rtwsm_linearDepthOffsetInverse(jitterR * pow(sssFactor, 0.25) * SETTING_SSS_DEPTH_RANGE);
    }

    float radius2 = jitterRcpSamples;
    for (uint i = 0; i < SAMPLE_COUNT; i++) {
        dir *= MAT2_GOLDEN_ANGLE;
        float baseRadius = sqrt(radius2);
        radius2 += rcpSamples;
        vec2 baseOffset = dir * baseRadius;

        vec2 sampleTexCoordXY = fma(ssRange2, baseOffset, shadowScreenPos.xy);
        sampleTexCoordXY = rtwsm_warpTexCoord_shared(sampleTexCoordXY);

        vec4 sampleShadowDepthOffset4 = textureGather(shadowcolor0, sampleTexCoordXY, 0);
        float sampleTexCoordZ = shadowScreenPos.z - max4(abs(sampleShadowDepthOffset4));

        float shadowSampleSolid = rtwsm_sampleShadowDepth(shadowtex1HW, vec3(sampleTexCoordXY, sampleTexCoordZ), 0.0);
        solidShadowSum += shadowSampleSolid;

        // Only translucent samples need the extra shadow gathers.
        bool hasTranslucentSample = (or4(floatBitsToUint(sampleShadowDepthOffset4)) & 0x80000000u) != 0u && shadowSampleSolid > 0.0;
        if (hasTranslucentSample) {
            vec4 waterMask4 = textureGather(usam_shadow_waterMask, sampleTexCoordXY, 0);
            float waterMaskSum = sum4(waterMask4);
            vec4 shadowDepthAll = textureGather(shadowtex0, sampleTexCoordXY, 0);
            if (any(greaterThan(vec4(sampleTexCoordZ), shadowDepthAll))) {
                f16vec3 sampleColor = f16vec3(textureLod(shadowcolor2, sampleTexCoordXY, 0.0).rgb);

                if (waterMaskSum > 0.1) {
                    vec4 translucentDistance = saturate(sampleTexCoordZ - shadowDepthAll);
                    float translucentDistanceMasked = dot(translucentDistance, waterMask4) / waterMaskSum;
                    float waterDepth = max(translucentDistanceMasked * shadowDepthRange, 0.0);
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

    float solidShadow = solidShadowSum * rcpSamples;
    if (solidShadow > 0.0 && solidShadow < 1.0) {
        float w = rcp(fma(blockerDistance, 0.5, 0.0001)) + 1.0;
        float sh = shadowHarden(solidShadow, w);
        solidShadow = sh * (2.0 - sh);
    }

    float sssAdjustedBendShadow = bendShadow;
    if (isSSS) {
        const float DECAY_FACTOR = 1.0;
        float factor = DECAY_FACTOR * rcp(abs(viewPos.z) + DECAY_FACTOR);
        sssAdjustedBendShadow = mix(sssAdjustedBendShadow, solidShadow, factor);
    }

    solidShadow = min(sssAdjustedBendShadow, solidShadow);

    vec3 finalShadow = vec3(solidShadow);
    if (translucentShadowSum.a > float16_t(0.0)) {
        finalShadow *= vec3(translucentShadowSum.rgb) * rcp(float(translucentShadowSum.a));
    }

    float surfaceDepth = hasSSS ? max(blockerDistance, 0.1) : 0.0;

    float shadowRangeBlend = smoothstep(shadowRangeStart, shadowRangeEnd, sqrt(sceneDistance2));

    return mix(vec4(finalShadow, surfaceDepth), vec4(bendShadow.rrr, 1.0), shadowRangeBlend);
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
        float viewZ = hiz_groupGroundCheckSubgroupLoadViewZ(swizzledWGPos.xy, 4, texelPos);

        if (viewZ > -65536.0) {
            GBufferData gData = gbufferData_init();
            gbufferData1_unpack(texelFetch(usam_gbufferSolidData1, texelPos, 0), gData);
            gbufferData2_unpack(texelFetch(usam_gbufferSolidData2, texelPos, 0), gData);
            #ifdef SETTING_RTWSM_B
            rtwsm_backward(texelPos, viewZ, gData);
            #endif
            vec4 shadowValue = compShadow(texelPos, viewZ, gData);
            shadowValue = clamp(shadowValue, 0.0, FP16_MAX);
            transient_shadow_store(texelPos, shadowValue);
        }
    }
}
