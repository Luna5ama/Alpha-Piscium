// Contains code adopted from:
// https://github.com/sebh/UnrealEngineSkyAtmosphere
// MIT License
// Copyright (c) 2020 Epic Games, Inc.
//
// https://github.com/GameTechDev/OutdoorLightScattering
// Apache License 2.0
// Copyright (c) 2017 Intel Corporation
//
// You can find full license texts in /licenses
#include "Common.glsl"
#include "/rtwsm/RTWSM.glsl"

uniform sampler2D usam_rtwsm_imap;
const bool shadowHardwareFiltering0 = true;
uniform sampler2DShadow shadowtex0HW;

float sampleShadow(vec3 shadowPos) {
    vec3 sampleTexCoord = shadowPos;
    sampleTexCoord.xy = rtwsm_warpTexCoord(usam_rtwsm_imap, sampleTexCoord.xy);
    return rtwsm_sampleShadowDepth(shadowtex0HW, sampleTexCoord, 0.0);
}

ScatteringResult raymarchSingleScatteringShadowed(
AtmosphereParameters atmosphere, RaymarchParameters params, LightParameters sunParams, LightParameters moonParams,
vec3 shadowStart, vec3 shadowEnd, float stepJitter
) {
    ScatteringResult result = ScatteringResult(vec3(1.0), vec3(0.0));

    float rcpSteps = 1.0 / float(params.steps);
    vec3 rayStepDelta = (params.rayEnd - params.rayStart) * rcpSteps;
    float rayStepLength = length(params.rayEnd - params.rayStart) * rcpSteps;
    vec3 shaodwStepDelta = (shadowEnd - shadowStart) * rcpSteps;

    vec3 totalInSctr = vec3(0.0);
    vec3 tSampleToOrigin = vec3(1.0);

    float shadowIsSun = float(all(equal(sunPosition, shadowLightPosition)));

    for (uint stepIndex = 0u; stepIndex < params.steps; stepIndex++) {
        float stepIndexF = float(stepIndex) + stepJitter;
        vec3 samplePos = params.rayStart + stepIndexF * rayStepDelta;
        float sampleHeight = length(samplePos);

        vec3 sampleDensity = sampleParticleDensity(atmosphere, sampleHeight);
        vec3 sampleExtinction = computeOpticalDepth(atmosphere, sampleDensity);
        vec3 sampleOpticalDepth = sampleExtinction * rayStepLength;
        vec3 sampleTransmittance = exp(-sampleOpticalDepth);

        vec3 rayleighInSctr = sampleDensity.x * atmosphere.rayleighSctrCoeff;
        vec3 mieInSctr = sampleDensity.y * atmosphere.mieSctrCoeff;

        vec3 sampleShadowPos = shadowStart + stepIndexF * shaodwStepDelta;
        float shadowSample = sampleShadow(sampleShadowPos);

        {
            vec3 tSunToSample = sampleTransmittanceLUT(atmosphere, sunParams.cosZenith, sampleHeight);
            vec3 multiSctrLuminance = sampleMultiSctrLUT(atmosphere, sunParams.cosZenith, sampleHeight);

            float shadow = mix(1.0, shadowSample, shadowIsSun);
            vec3 sampleInSctr = shadow * tSunToSample * computeTotalInSctr(atmosphere, sunParams, sampleDensity);
            sampleInSctr += multiSctrLuminance * (rayleighInSctr + mieInSctr);

            // See slide 28 at http://www.frostbite.com/2015/08/physically-based-unified-volumetric-rendering-in-frostbite/
            vec3 sampleInSctrInt = (sampleInSctr - sampleInSctr * sampleTransmittance) / sampleExtinction;
            totalInSctr += tSampleToOrigin * sampleInSctrInt * sunParams.radiance;
        }

        {
            vec3 tMoonToSample = sampleTransmittanceLUT(atmosphere, moonParams.cosZenith, sampleHeight);
            vec3 multiSctrLuminance = sampleMultiSctrLUT(atmosphere, moonParams.cosZenith, sampleHeight);

            float shadow = mix(shadowSample, 1.0, shadowIsSun);
            vec3 sampleInSctr = shadow * tMoonToSample * computeTotalInSctr(atmosphere, moonParams, sampleDensity);
            sampleInSctr += multiSctrLuminance * (rayleighInSctr + mieInSctr);

            vec3 sampleInSctrInt = (sampleInSctr - sampleInSctr * sampleTransmittance) / sampleExtinction;
            totalInSctr += tSampleToOrigin * sampleInSctrInt * moonParams.radiance;
        }

        tSampleToOrigin *= sampleTransmittance;
    }

    result.transmittance = tSampleToOrigin;
    result.inScattering = totalInSctr;

    return result;
}


// originView: ray origin in view space
// endView: ray end in view space
ScatteringResult computeSingleScattering(AtmosphereParameters atmosphere, vec3 originView, vec3 endView, float stepJitter) {
    ScatteringResult result = ScatteringResult(vec3(1.0), vec3(0.0));

    mat3 vectorView2World = mat3(gbufferModelViewInverse);

    vec3 viewDirView = normalize(endView - originView);
    vec3 viewDirWorld = normalize(vectorView2World * viewDirView);

    vec3 rayDir = viewDirWorld;
    vec3 sunRadiance = global_sunRadiance.rgb * global_sunRadiance.a;
    vec3 moonRadiance = sunRadiance * MOON_RADIANCE_MUL;

    RaymarchParameters params;
    params.rayStart = atmosphere_viewToAtm(atmosphere, originView);
    LightParameters sunParams;
    lightParameters_setup(atmosphere, sunParams, sunRadiance, uval_sunDirWorld, rayDir);
    LightParameters moonParams;
    lightParameters_setup(atmosphere, moonParams, moonRadiance, uval_moonDirWorld, rayDir);

    if (endView.z == -65536.0) {
        params.rayStart.y = max(params.rayStart.y, atmosphere.bottom + 0.5);
        vec3 earthCenter = vec3(0.0);

        // Check if ray origin is outside the atmosphere
        if (length(params.rayStart) > atmosphere.top) {
            float tTop = raySphereIntersectNearest(params.rayStart, rayDir, earthCenter, atmosphere.top);
            if (tTop < 0.0) {
                return result; // No intersection with atmosphere: stop right away
            }
            params.rayStart += rayDir * (tTop + 0.001);
        }

        float tBottom = raySphereIntersectNearest(params.rayStart, rayDir, earthCenter, atmosphere.bottom);
        float tTop = raySphereIntersectNearest(params.rayStart, rayDir, earthCenter, atmosphere.top);
        float rayLen = 0.0;

        if (tBottom < 0.0) {
            if (tTop < 0.0) {
                return result; // No intersection with earth nor atmosphere: stop right away
            } else {
                rayLen = tTop;
            }
        } else {
            if (tTop > 0.0) {
                rayLen = min(tTop, tBottom);
            }
        }

        params.rayEnd = params.rayStart + rayDir * rayLen;
        params.steps = SETTING_SKY_SAMPLES;
        return raymarchSingleScattering(atmosphere, params, sunParams, moonParams);
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
        return raymarchSingleScatteringShadowed(atmosphere, params, sunParams, moonParams, startShadow, endShadow, stepJitter);
    }
}