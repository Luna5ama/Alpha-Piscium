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
#include "/rtwsm/RTWSM.glsl"
#include "/util/Celestial.glsl"

uniform sampler2D usam_rtwsm_imap;
const bool shadowHardwareFiltering0 = true;
uniform sampler2DShadow shadowtex0HW;

float atmosphere_sample_shadow(vec3 shadowPos) {
    vec3 sampleTexCoord = shadowPos;
    sampleTexCoord.xy = rtwsm_warpTexCoord(usam_rtwsm_imap, sampleTexCoord.xy);
    return rtwsm_sampleShadowDepth(shadowtex0HW, sampleTexCoord, 0.0);
}

#define ATMOSPHERE_RAYMARCHING_SKY
#define ATMOSPHERE_RAYMARCHING_AERIAL_PERSPECTIVE
#include "Raymarching.glsl"

// originView: ray origin in view space
// endView: ray end in view space
ScatteringResult computeSingleScattering(AtmosphereParameters atmosphere, vec3 originView, vec3 endView, float stepJitter, float multiSctrFactor) {
    ScatteringResult result = scatteringResult_init();

    mat3 vectorView2World = mat3(gbufferModelViewInverse);

    vec3 viewDirView = normalize(endView - originView);
    vec3 viewDirWorld = normalize(vectorView2World * viewDirView);

    vec3 rayDir = viewDirWorld;

    RaymarchParameters params;
    params.stepJitter = stepJitter;
    params.rayStart = atmosphere_viewToAtm(atmosphere, originView);
    LightParameters sunParam = lightParameters_init(atmosphere, SUN_ILLUMINANCE, uval_sunDirWorld, rayDir);
    LightParameters moonParams = lightParameters_init(atmosphere, MOON_ILLUMINANCE, uval_moonDirWorld, rayDir);
    ScatteringParameters scatteringParams = scatteringParameters_init(sunParam, moonParams, multiSctrFactor);

    if (endView.z == -65536.0) {
        params.rayStart.y = max(params.rayStart.y, atmosphere.bottom + 0.5);
        params.steps = SETTING_SKY_SAMPLES;

        if (setupRayEnd(atmosphere, params, rayDir)) {
            result = raymarchSky(atmosphere, params, scatteringParams);
        }
    } else {
        params.rayEnd = atmosphere_viewToAtm(atmosphere, endView);

        vec4 originScene = gbufferModelViewInverse * vec4(originView, 1.0);
        vec4 endScene = gbufferModelViewInverse * vec4(endView, 1.0);

        vec4 originShadowCS = global_shadowRotationMatrix * shadowProjection * shadowModelView * originScene;
        vec4 endShadowCS = global_shadowRotationMatrix * shadowProjection * shadowModelView * endScene;

        vec3 startShadow = originShadowCS.xyz / originShadowCS.w;
        startShadow = startShadow * 0.5 + 0.5;
        vec3 endShadow = endShadowCS.xyz / endShadowCS.w;
        endShadow = endShadow * 0.5 + 0.5;

        params.steps = SETTING_LIGHT_SHAFT_SAMPLES;
        result = raymarchAerialPerspective(atmosphere, params, scatteringParams, startShadow, endShadow);
    }

    return result;
}