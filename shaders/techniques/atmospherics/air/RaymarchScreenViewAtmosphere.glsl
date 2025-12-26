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
#include "Common.glsl"
#include "/techniques/rtwsm/RTWSM.glsl"
#include "/util/Celestial.glsl"
#include "/util/Rand.glsl"

vec2 _processShadowSampleUV(vec2 sampleShadowUV, ivec2 randCoord) {
    float rv = rand_stbnVec1(randCoord, 0);
    vec2 dir = rand_stbnUnitVec211(randCoord, 0);
    float sqrtJitterR = sqrt(rv);
    float r = ldexp(sqrtJitterR, -12 + SETTING_WATER_LIGHT_SHAFT_SOFTNESS);
    vec2 result = sampleShadowUV;
    result += r * dir * vec2(global_shadowProjPrev[0][0], global_shadowProjPrev[1][1]);
    result = rtwsm_warpTexCoord(usam_rtwsm_imap, result);
    return result;
}

#ifdef SHARED_MEMORY_SHADOW_SAMPLE
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
    if (gl_LocalInvocationIndex == 0) {
        vec3 viewDir = normalize(coords_toViewCoord(screenPos, -1.0, global_camProjInverse));
        vec3 sliceNormal = normalize(cross(uval_shadowLightDirView, viewDir));
        vec3 perpViewDir = normalize(cross(uval_shadowLightDirView, sliceNormal));
        perpViewDir = faceforward(-perpViewDir, viewDir, perpViewDir);

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

float atmosphere_sample_shadow(vec3 startShadowPos, vec3 endShadowPos) {
    vec3 startLength = shared_sliceShadowScreenStartLength;
    float startT = sqrt(compT(startLength, startShadowPos));
    float endT = sqrt(compT(startLength, endShadowPos));
    float shadowSum = 0.0;
    const uint SHADOW_STEPS = SETTING_LIGHT_SHAFT_SHADOW_SAMPLES;
    vec2 startTAndDepth = vec2(startT, startShadowPos.z);
    vec2 stepT = vec2(endT - startT, endShadowPos.z - startShadowPos.z) / float(SHADOW_STEPS);
    for (uint i = 0u; i < SHADOW_STEPS; ++i) {
        float fi = float(i) + 0.5;
        vec2 sampleTAndDepth = startTAndDepth + fi * stepT;
        float indexF = sampleTAndDepth.x * float(SHADOW_SAMPLE_COUNT - 1);
        uint index = uint(indexF);
        float shadowTerm = float(shared_sliceShadowSamples[index] > saturate(sampleTAndDepth.y));
        shadowSum += saturate(shadowTerm + float(sampleTAndDepth.x > 1.0));
    }
    return shadowSum / float(SHADOW_STEPS);
}
#else
ivec2 _texelPos = ivec2(-1);
float atmosphere_sample_shadow(vec3 startShadowPos, vec3 endShadowPos) {
    vec3 sampleShadowUV = (startShadowPos + endShadowPos) * 0.5;
    sampleShadowUV.xy = _processShadowSampleUV(sampleShadowUV.xy, _texelPos);
    return rtwsm_sampleShadowDepth(shadowtex0HW, sampleShadowUV, 0.0);
}
#endif

/*const*/
#define ATMOSPHERE_RAYMARCHING_AERIAL_PERSPECTIVE a
/*const*/
#include "Raymarching.glsl"

const vec3 ORIGIN_VIEW = vec3(0.0);

ScatteringResult raymarchScreenViewAtmosphere(ivec2 texelPos, float startZ, float endZ, uint steps, float noiseV) {
    #ifndef SHARED_MEMORY_SHADOW_SAMPLE
    _texelPos = texelPos;
    #endif
    ScatteringResult result = scatteringResult_init();

    vec2 screenPos = (vec2(texelPos) + 0.5) * uval_mainImageSizeRcp;
    vec3 startViewPos = coords_toViewCoord(screenPos, startZ, global_camProjInverse);
    vec3 endViewPos = coords_toViewCoord(screenPos, max(endZ, -shadowDistance), global_camProjInverse);

    ivec2 texePos2x2 = texelPos >> 1;
//    TODO: Fixed lightmap coordinate-based multi-scattering factor
//    float lmCoordSky = abs(unpackHalf2x16(transient_packedZN_fetch(texePos2x2 + ivec2(0, global_mipmapSizesI[1].y)).y).y);
    float lmCoordSky = 1.0;
    float multiSctrFactor = max(lmCoordSky, linearStep(0.0, 240.0, float(eyeBrightnessSmooth.y)));

    mat3 vectorView2World = mat3(gbufferModelViewInverse);
    vec3 viewDirWorld = normalize(vectorView2World * (endViewPos - startViewPos));

    vec3 rayDir = viewDirWorld;

    AtmosphereParameters atmosphere = getAtmosphereParameters();
    RaymarchParameters params = raymarchParameters_init();
    params.rayStart = atmosphere_viewToAtm(atmosphere, startViewPos);
    params.steps = steps;
    LightParameters sunParam = lightParameters_init(atmosphere, SUN_ILLUMINANCE , uval_sunDirWorld, rayDir);
    LightParameters moonParams = lightParameters_init(atmosphere, MOON_ILLUMINANCE, uval_moonDirWorld, rayDir);
    ScatteringParameters scatteringParams = scatteringParameters_init(sunParam, moonParams, multiSctrFactor);

    vec4 originScene = gbufferModelViewInverse * vec4(startViewPos, 1.0);
    vec4 endScene = gbufferModelViewInverse * vec4(endViewPos, 1.0);

    vec4 originShadowCS = global_shadowProjPrev * global_shadowRotationMatrix * global_shadowView * originScene;
    vec4 endShadowCS = global_shadowProjPrev * global_shadowRotationMatrix * global_shadowView * endScene;

    vec3 startShadow = originShadowCS.xyz / originShadowCS.w;
    startShadow = startShadow * 0.5 + 0.5;
    vec3 endShadow = endShadowCS.xyz / endShadowCS.w;
    endShadow = endShadow * 0.5 + 0.5;

    if (endZ == -65536.0) {
        scatteringParams.multiSctrFactor = 1.0;

        if (setupRayEnd(atmosphere, params, rayDir, shadowDistance / SETTING_ATM_D_SCALE)) {
            result = raymarchAerialPerspective(atmosphere, params, scatteringParams, startShadow, endShadow, noiseV);
        }
    } else {
        params.rayEnd = atmosphere_viewToAtm(atmosphere, endViewPos);
        result = raymarchAerialPerspective(atmosphere, params, scatteringParams, startShadow, endShadow, noiseV);
    }

    return result;
}