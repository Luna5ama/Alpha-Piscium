/*
    References:
        [EPI20] Epic Games, Inc. "Unreal Engine Sky Atmosphere Rendering Technique". 2020.
            MIT License. Copyright (c) 2020 Epic Games, Inc.
            https://github.com/sebh/UnrealEngineSkyAtmosphere
        [INT17] Intel Corporation. "Outdoor Light Scattering Sample". 2017.
            Apache License 2.0. Copyright (c) 2017 Intel Corporation.
            https://github.com/GameTechDev/OutdoorLightScattering

        You can find full license texts in /licenses
*/
#include "Constants.glsl"
#include "../air/RaymarchingBase.glsl"
#include "../air/lut/API.glsl"
#include "../Utils.glsl"
#include "/techniques/rtwsm/RTWSM.glsl"
#include "/util/Coords.glsl"
#include "/util/Lighting.glsl"
#include "/util/Rand.glsl"
#include "/util/Celestial.glsl"

#if SETTING_SLICE_SAMPLES == 128
#define WORK_GROUP_SIZE 128
#define LOOP_COUNT 1
#elif SETTING_SLICE_SAMPLES == 256
#define WORK_GROUP_SIZE 256
#define LOOP_COUNT 1
#elif SETTING_SLICE_SAMPLES == 512
#define WORK_GROUP_SIZE 256
#define LOOP_COUNT 2
#elif SETTING_SLICE_SAMPLES == 1024
#define WORK_GROUP_SIZE 256
#define LOOP_COUNT 4
#endif
layout(local_size_x = 1, local_size_y = WORK_GROUP_SIZE) in;
const ivec3 workGroups = ivec3(SETTING_EPIPOLAR_SLICES, 1, 1);

layout(rgba32ui) uniform restrict uimage2D uimg_epipolarData;

vec2 _processShadowSampleUV(vec2 sampleShadowUV, ivec2 randCoord) {
    float rv = rand_stbnVec1(randCoord, 0);
    vec2 dir = rand_stbnUnitVec211(randCoord, 0);
    float sqrtJitterR = sqrt(rv);
    //    const int SOFTNESS = SETTING_LIGHT_SHAFT_SOFTNESS;
    const int SOFTNESS = 7;
    float r = ldexp(sqrtJitterR, -12 + SOFTNESS);
    vec2 result = sampleShadowUV;
    result += r * dir * vec2(global_shadowProjPrev[0][0], global_shadowProjPrev[1][1]);
    result = rtwsm_warpTexCoord(usam_rtwsm_imap, result);
    return result;
}
shared vec4 shared_sliceShadowScreenStartEnd;
shared vec3 shared_sliceShadowScreenStartLength;
#define SHADOW_SAMPLE_COUNT (WORK_GROUP_SIZE * 8)
shared float shared_sliceShadowSamples[SHADOW_SAMPLE_COUNT];


void loadSharedShadowSample(uint index) {
    float fi = float(index);
    float t = saturate(pow2(fi / float(SHADOW_SAMPLE_COUNT - 1)));

    vec4 endPoints = shared_sliceShadowScreenStartEnd;
    vec2 sampleShadowUV = mix(endPoints.xy, endPoints.zw, saturate(t));
    ivec2 randCoord = ivec2(gl_WorkGroupID.x, index);
    sampleShadowUV.xy = _processShadowSampleUV(sampleShadowUV.xy, randCoord);

    float shadowSampleDepth = texture(shadowtex1, sampleShadowUV).r;
    vec2 ndcCoord = sampleShadowUV * 2.0 - 1.0;
    float edgeCoord = max(abs(ndcCoord.x), abs(ndcCoord.y));
    shadowSampleDepth = mix(shadowSampleDepth, 1.0, linearStep(1.0 - SHADOW_MAP_SIZE.y * 16, 1.0, edgeCoord));
    shared_sliceShadowSamples[index] = shadowSampleDepth;
}

void screenViewRaymarch_init(vec2 screenPos) {
    if (gl_LocalInvocationIndex == (WORK_GROUP_SIZE - 1)) {
        vec3 viewDir = normalize(coords_toViewCoord(screenPos, -1.0, global_camProjInverse));
        vec3 sliceNormal = normalize(cross(uval_shadowLightDirView, viewDir));
        vec3 perpViewDir = normalize(cross(uval_shadowLightDirView, sliceNormal));
        perpViewDir = viewDir;

        vec3 sliceShadowStartView = perpViewDir * near;
        vec3 sliceShadowEndView = perpViewDir * shadowDistance;

        vec4 sliceShadowStartScene = gbufferModelViewInverse * vec4(sliceShadowStartView, 1.0);
        vec4 sliceShadowEndScene = gbufferModelViewInverse * vec4(sliceShadowEndView, 1.0);

        vec4 sliceShadowStartShadowClip = global_shadowProjPrev * global_shadowRotationMatrix * global_shadowView * sliceShadowStartScene;
        vec4 sliceShadowEndShadowClip = global_shadowProjPrev * global_shadowRotationMatrix * global_shadowView * sliceShadowEndScene;

        vec2 sliceShadowStartShadowScreen = sliceShadowStartShadowClip.xy / sliceShadowStartShadowClip.w;
        sliceShadowStartShadowScreen = sliceShadowStartShadowScreen * 0.5 + 0.5;
        vec2 sliceShadowEndShadowScreen = sliceShadowEndShadowClip.xy / sliceShadowEndShadowClip.w;
        sliceShadowEndShadowScreen = sliceShadowEndShadowScreen * 0.5 + 0.5;

        shared_sliceShadowScreenStartEnd = vec4(sliceShadowStartShadowScreen, sliceShadowEndShadowScreen);
        shared_sliceShadowScreenStartLength = vec3(sliceShadowStartShadowScreen, distance(sliceShadowStartShadowScreen, sliceShadowEndShadowScreen));
    }

    barrier();

    loadSharedShadowSample(gl_LocalInvocationIndex);
    loadSharedShadowSample(gl_LocalInvocationIndex + WORK_GROUP_SIZE);
    loadSharedShadowSample(gl_LocalInvocationIndex + WORK_GROUP_SIZE * 2);
    loadSharedShadowSample(gl_LocalInvocationIndex + WORK_GROUP_SIZE * 3);
    loadSharedShadowSample(gl_LocalInvocationIndex + WORK_GROUP_SIZE * 4);
    loadSharedShadowSample(gl_LocalInvocationIndex + WORK_GROUP_SIZE * 5);
    loadSharedShadowSample(gl_LocalInvocationIndex + WORK_GROUP_SIZE * 6);
    loadSharedShadowSample(gl_LocalInvocationIndex + WORK_GROUP_SIZE * 7);
    barrier();
}

float compT(vec3 startLength, vec3 shadowPos) {
    return distance(shadowPos.xy, startLength.xy) / startLength.z;
}

float atmosphere_sample_shadow(vec3 startShadowPos, vec3 endShadowPos, float jitter) {
    vec3 startLength = shared_sliceShadowScreenStartLength;
    float startT = sqrt(compT(startLength, startShadowPos));
    float endT = sqrt(compT(startLength, endShadowPos));
    float shadowSum = 0.0;
    const uint SHADOW_STEPS = 128u;
    vec2 startTAndDepth = vec2(startT, startShadowPos.z);
    vec2 stepT = vec2(endT - startT, endShadowPos.z - startShadowPos.z) / float(SHADOW_STEPS);
    for (uint i = 0u; i < SHADOW_STEPS; ++i) {
        float fi = float(i) + jitter;
        vec2 sampleTAndDepth = startTAndDepth + fi * stepT;
        float indexF = sampleTAndDepth.x * float(SHADOW_SAMPLE_COUNT - 1);
        uint index = uint(indexF);
        float shadowTerm = float(shared_sliceShadowSamples[index] > saturate(sampleTAndDepth.y));
        shadowSum += saturate(shadowTerm + float(sampleTAndDepth.x > 1.0));
    }
    return shadowSum / float(SHADOW_STEPS);
}

const vec3 ORIGIN_VIEW = vec3(0.0);

float waterSurfaceDistance(vec3 shadowUVPos) {
    shadowUVPos.xy = rtwsm_warpTexCoord(usam_rtwsm_imap, shadowUVPos.xy);
    vec2 ndcCoord = shadowUVPos.xy * 2.0 - 1.0;
    float edgeCoord = max(abs(ndcCoord.x), abs(ndcCoord.y));
    if (edgeCoord > 1.0 - SHADOW_MAP_SIZE.y * 16) {
        return -1.0;
    }
    if (texture(usam_shadow_waterMask, shadowUVPos.xy).r < 0.9) {
        return -1.0;
    }
    float sampleDepth = texture(shadowtex0, shadowUVPos.xy).r;
    return abs(rtwsm_linearDepth(shadowUVPos.z) - rtwsm_linearDepth(sampleDepth));
}

// https://www.desmos.com/calculator/tbl4g5bvlc
float waterPhase(float cosTheta) {
    const float wKn = 0.99;
    const float gE = 20000.0;
    const float gCS = -0.6;
    return mix(
        phasefunc_CornetteShanks(cosTheta, gCS),
        phasefunc_KleinNishinaE(cosTheta, gE),
        wKn
    );
}

ScatteringResult raymarchWaterVolume(
    vec3 rayStart,
    vec3 rayEnd,
    vec3 shadowStart,
    vec3 shadowEnd,
    float jitter
) {
    ScatteringResult result = scatteringResult_init();
    AtmosphereParameters atmosphere = getAtmosphereParameters();

    float startLightRayLength = waterSurfaceDistance(shadowStart);
    float endLightRayLength = waterSurfaceDistance(shadowEnd);
    float rcpShadowY = rcp(uval_shadowLightDirWorld.y);
    float startWorldHeight = rayStart.y + cameraPosition.y;
    float endWorldHeight = rayEnd.y + cameraPosition.y;
//    if (startLightRayLength == -1.0) {
        startLightRayLength = max(63.0 - startWorldHeight, 0.0) * rcpShadowY;
//    }
//    if (endLightRayLength == -1.0) {
        endLightRayLength = max(63.0 - endWorldHeight, 0.0) * rcpShadowY;
//    }

    vec3 rayDiff = rayEnd - rayStart;
    float totalRayLength = length(rayDiff);
    vec3 rayDir = rayDiff / totalRayLength;
    float phaseCosTheta = dot(rayDir, uval_shadowLightDirWorld);
    float phaseV = waterPhase(phaseCosTheta);

    vec3 inSctrInt = volumetrics_intergrateScatteringLerpLightOpticalDepth(
        WATER_SCATTERING,
        WATER_EXTINCTION,
        totalRayLength,
        startLightRayLength,
        endLightRayLength
    );

    vec3 totalInSctr = vec3(0.0);

    float shadowSample = atmosphere_sample_shadow(shadowStart, shadowEnd, jitter);
    float shadowIsSun = float(all(equal(sunPosition, shadowLightPosition)));
    float midPointWorldHeight = (startWorldHeight + endWorldHeight) * 0.5;
    float atmHeight = atmosphere_height(atmosphere, midPointWorldHeight);
    const vec3 UP_VECTOR = vec3(0.0, 1.0, 0.0);

    {
        float cosZenith = dot(UP_VECTOR, uval_sunDirWorld);
        vec3 atmT = atmospherics_air_lut_sampleTransmittance(atmosphere, cosZenith, atmHeight);
        float sunT1 = phaseV * mix(1.0, shadowSample, shadowIsSun);
        vec3 sunT3 = atmT * SUN_ILLUMINANCE;
        totalInSctr += inSctrInt * sunT1 * sunT3;
    }

    {
        float cosZenith = dot(UP_VECTOR, uval_moonDirWorld);
        vec3 atmT = atmospherics_air_lut_sampleTransmittance(atmosphere, cosZenith, atmHeight);
        float moonT1 = phaseV * mix(shadowSample, 1.0, shadowIsSun);
        vec3 moonT3 = atmT * MOON_ILLUMINANCE;
        totalInSctr += inSctrInt * moonT1 * moonT3;
    }

    vec3 totalTransmittance = exp(-WATER_EXTINCTION * totalRayLength);

    result.inScattering = totalInSctr;
    result.transmittance = totalTransmittance;

    return result;
}

ScatteringResult raymarchScreenViewWater(ivec2 texelPos, float startZ, float endZ, uint steps, float noiseV) {
    ScatteringResult result = scatteringResult_init();

    vec2 screenPos = (vec2(texelPos) + 0.5) * uval_mainImageSizeRcp;
    vec3 startViewPos = coords_toViewCoord(screenPos, startZ, global_camProjInverse);
    vec3 endViewPos = coords_toViewCoord(screenPos, max(endZ, -max(shadowDistance, far)), global_camProjInverse);

    ivec2 texePos2x2 = texelPos >> 1;
    float lmCoordSky = abs(unpackHalf2x16(texelFetch(usam_packedZN, texePos2x2 + ivec2(0, global_mipmapSizesI[1].y), 0).y).y);
    float multiSctrFactor = max(lmCoordSky, linearStep(0.0, 240.0, float(eyeBrightnessSmooth.y)));

    vec4 originScene = gbufferModelViewInverse * vec4(startViewPos, 1.0);
    vec4 endScene = gbufferModelViewInverse * vec4(endViewPos, 1.0);

    vec4 originShadowCS = global_shadowProjPrev * global_shadowRotationMatrix * global_shadowView * originScene;
    vec4 endShadowCS = global_shadowProjPrev * global_shadowRotationMatrix * global_shadowView * endScene;

    vec3 startShadow = originShadowCS.xyz / originShadowCS.w;
    startShadow = startShadow * 0.5 + 0.5;
    vec3 endShadow = endShadowCS.xyz / endShadowCS.w;
    endShadow = endShadow * 0.5 + 0.5;

    result = raymarchWaterVolume(originScene.xyz, endScene.xyz, startShadow, endShadow, noiseV);

    return result;
}

void main() {
    ivec2 imgSizei = ivec2(SETTING_EPIPOLAR_SLICES, SETTING_SLICE_SAMPLES);
    vec2 imgSize = vec2(SETTING_EPIPOLAR_SLICES, SETTING_SLICE_SAMPLES);
    uint sliceIndex = gl_WorkGroupID.x;
    vec4 sliceEndPoints = uintBitsToFloat(imageLoad(uimg_epipolarData, ivec2(sliceIndex, 0)));

    uint cond = uint(isValidScreenLocation(sliceEndPoints.xy)) | uint(isValidScreenLocation(sliceEndPoints.zw));

    if (bool(cond)) {
        {
            uint sliceSampleIndex = gl_LocalInvocationID.y + (LOOP_COUNT - 1) * WORK_GROUP_SIZE;
            float sliceSampleP = float(sliceSampleIndex);
            vec2 screenPos = mix(sliceEndPoints.xy, sliceEndPoints.zw, sliceSampleP) * 0.5 + 0.5;
            screenViewRaymarch_init(screenPos);
        }
        for (uint i = 0; i < LOOP_COUNT; i++) {
            uint sliceSampleIndex = gl_LocalInvocationID.y + i * WORK_GROUP_SIZE;
            float sliceSampleP = float(sliceSampleIndex);
            sliceSampleP /= float(SETTING_SLICE_SAMPLES - 1);

            vec2 screenPos = mix(sliceEndPoints.xy, sliceEndPoints.zw, sliceSampleP) * 0.5 + 0.5;

            vec2 texelPos = screenPos * uval_mainImageSize;
            texelPos = clamp(texelPos, vec2(0.5), vec2(uval_mainImageSize - 0.5));
            ivec2 texelPosI = ivec2(texelPos);
            float noiseV = rand_stbnVec1(texelPosI, frameCounter);

            ivec2 readScreenTexelPos = texelPosI;
            readScreenTexelPos.y += int(uval_mainImageSizeIY);
            vec2 layerViewZ = -abs(texelFetch(usam_csrg32f, readScreenTexelPos, 0).rg);
            ScatteringResult result = scatteringResult_init();

            if (layerViewZ.x > -FLT_MAX) {
                result = raymarchScreenViewWater(
                    texelPosI,
                    layerViewZ.x,
                    layerViewZ.y,
                    SETTING_LIGHT_SHAFT_SAMPLES,
                    noiseV
                );
            }

            result.inScattering = clamp(result.inScattering, 0.0, FP16_MAX);
            uvec4 outputData;
            packEpipolarData(outputData, result, texelPosI);
            ivec2 writeTexelPos = ivec2(gl_GlobalInvocationID.x, sliceSampleIndex + 1u);
            writeTexelPos.y += int(SETTING_SLICE_SAMPLES);
            imageStore(uimg_epipolarData, writeTexelPos, outputData);
        }
    } else {
        ivec2 texelPos = ivec2(65535);
        ScatteringResult result = scatteringResult_init();
        uvec4 outputData;
        packEpipolarData(outputData, result, texelPos);
        for (uint i = 0; i < LOOP_COUNT; i++) {
            ivec2 writeTexelPos = ivec2(gl_GlobalInvocationID.xy);
            writeTexelPos.y += int(1u + i * WORK_GROUP_SIZE + SETTING_SLICE_SAMPLES);
            imageStore(uimg_epipolarData, writeTexelPos, outputData);
        }
    }
}