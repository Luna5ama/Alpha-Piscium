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

//uniform sampler2D usam_rtwsm_imap;
//const bool shadowHardwareFiltering0 = true;
//uniform sampler2DShadow shadowtex0HW;
//uniform sampler2D usam_gbufferViewZ;
//uniform usampler2D usam_packedZN;

float atmosphere_sample_shadow(vec3 shadowPos) {
    vec3 sampleTexCoord = shadowPos;
    sampleTexCoord.xy = rtwsm_warpTexCoord(usam_rtwsm_imap, sampleTexCoord.xy);
    return rtwsm_sampleShadowDepth(shadowtex0HW, sampleTexCoord, 0.0);
}

#define ATMOSPHERE_RAYMARCHING_SKY  a
#define ATMOSPHERE_RAYMARCHING_AERIAL_PERSPECTIVE a
#include "Raymarching.glsl"

const vec3 ORIGIN_VIEW = vec3(0.0);

ScatteringResult computeSingleScattering(vec2 screenPos, float viewZ, float noiseV) {
    AtmosphereParameters atmosphere = getAtmosphereParameters();
    ScatteringResult result = scatteringResult_init();

    vec3 viewPos = coords_toViewCoord(screenPos, viewZ, gbufferProjectionInverse);
    ivec2 texelPos = ivec2(screenPos * global_mainImageSize);
    ivec2 texePos2x2 = texelPos >> 1;
    float lmCoordSky = abs(unpackHalf2x16(texelFetch(usam_packedZN, texePos2x2 + ivec2(0, global_mipmapSizesI[1].y), 0).y).y);
    float multiSctrFactor = max(lmCoordSky, linearStep(0.0, 240.0, float(eyeBrightnessSmooth.y)));

    mat3 vectorView2World = mat3(gbufferModelViewInverse);
    vec3 viewDirWorld = normalize(vectorView2World * (viewPos - ORIGIN_VIEW));

    vec3 rayDir = viewDirWorld;

    RaymarchParameters params = raymarchParameters_init();
    params.rayStart = atmosphere_viewToAtm(atmosphere, ORIGIN_VIEW);
    LightParameters sunParam = lightParameters_init(atmosphere, SUN_ILLUMINANCE * PI, uval_sunDirWorld, rayDir);
    LightParameters moonParams = lightParameters_init(atmosphere, MOON_ILLUMINANCE, uval_moonDirWorld, rayDir);
    ScatteringParameters scatteringParams = scatteringParameters_init(sunParam, moonParams, multiSctrFactor);

    if (viewZ == -65536.0) {
        scatteringParams.multiSctrFactor = 1.0;

        params.rayStart.y = max(params.rayStart.y, atmosphere.bottom + 0.5);
        params.steps = SETTING_SKY_SAMPLES;

        if (setupRayEnd(atmosphere, params, rayDir)) {
            result = raymarchSky(atmosphere, params, scatteringParams);
        }
    } else {
        params.rayEnd = atmosphere_viewToAtm(atmosphere, viewPos);

        vec4 originScene = gbufferModelViewInverse * vec4(ORIGIN_VIEW, 1.0);
        vec4 endScene = gbufferModelViewInverse * vec4(viewPos, 1.0);

        vec4 originShadowCS = global_shadowRotationMatrix * shadowProjection * shadowModelView * originScene;
        vec4 endShadowCS = global_shadowRotationMatrix * shadowProjection * shadowModelView * endScene;

        vec3 startShadow = originShadowCS.xyz / originShadowCS.w;
        startShadow = startShadow * 0.5 + 0.5;
        vec3 endShadow = endShadowCS.xyz / endShadowCS.w;
        endShadow = endShadow * 0.5 + 0.5;

        params.steps = SETTING_LIGHT_SHAFT_SAMPLES;
        result = raymarchAerialPerspective(atmosphere, params, scatteringParams, startShadow, endShadow, noiseV);
    }

    return result;
}