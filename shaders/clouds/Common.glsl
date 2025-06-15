/*
    References:
        [HIL16] Hillaire, Sébastien. "Physically Based Sky, Atmosphere and Cloud Rendering in Frostbite".
            SIGGRAPH 2016. 2016.
            https://www.ea.com/frostbite/news/physically-based-sky-atmosphere-and-cloud-rendering
        [SCH15] Schneider, Andrew. "The Real-Time Volumetric Cloudscapes Of Horizon: Zero Dawn"
            SIGGRAPH 2015. 2015.
            https://www.guerrilla-games.com/read/the-real-time-volumetric-cloudscapes-of-horizon-zero-dawn
        [QIU25] QiuTang98. "Flower Engine"
            MIT License. Copyright (c) 2025 QiuTang98.
            https://github.com/qiutang98/flower
        [QIU23] QiuTang98. "实时体积云渲染的光照细节". 2023
            https://qiutang98.github.io/post/%E5%AE%9E%E6%97%B6%E6%B8%B2%E6%9F%93%E5%BC%80%E5%8F%91/%E5%AE%9E%E6%97%B6%E4%BD%93%E7%A7%AF%E4%BA%91%E6%B8%B2%E6%9F%93/

        You can find full license texts in /licenses
*/
#ifndef INCLUDE_clouds_Common_glsl
#define INCLUDE_clouds_Common_glsl a

#include "/atmosphere/Common.glsl"
#include "Constants.glsl"

struct CloudRayParams {
    vec3 rayStart;
    vec3 rayDir;
    vec3 rayEnd;
    float rayStartHeight;
    float rayEndHeight;
};

struct CloudRenderParams {
    vec3 lightDir;
    float LDotV;
};

CloudRenderParams cloudRenderParams_init(CloudRayParams rayParam, vec3 lightDir) {
    CloudRenderParams params;
    params.lightDir = normalize(lightDir);
    params.LDotV = -dot(rayParam.rayDir, lightDir);
    return params;
}

struct CloudParticpatingMedium {
    vec3 scattering;
    vec3 extinction;
    vec3 phase;
};

CloudParticpatingMedium clouds_cirrus_medium(CloudRenderParams renderParams) {
    CloudParticpatingMedium medium;
    medium.scattering = CLOUDS_CIRRUS_SCATTERING;
    medium.extinction = CLOUDS_CIRRUS_EXTINCTION;
    medium.phase = cornetteShanksPhase(renderParams.LDotV, CLOUDS_CIRRUS_ASYM);
    return medium;
}

struct CloudRaymarchAccumState {
    vec3 totalInSctr;
    vec3 totalTransmittance;
};

CloudRaymarchAccumState clouds_raymarchAccumState_init() {
    CloudRaymarchAccumState state;
    state.totalInSctr = vec3(0.0);
    state.totalTransmittance = vec3(1.0);
    return state;
}

struct CloudRaymarchStepState {
    vec3 samplePos;
    float sampleHeight;
    vec3 upVector;
    float sampleDensity;
};

CloudRaymarchStepState clouds_raymarchStepState_init(vec3 samplePos, float sampleDensity) {
    CloudRaymarchStepState state;
    state.samplePos = samplePos;
    state.sampleHeight = length(samplePos);
    state.upVector = samplePos / state.sampleHeight;
    state.sampleDensity = sampleDensity;
    return state;
}

void clouds_computeLighting(
    AtmosphereParameters atmosphere,
    CloudRenderParams renderParams,
    CloudParticpatingMedium medium,
    CloudRaymarchStepState stepState,
    float rayStepLength,
    inout CloudRaymarchAccumState accumState
) {
    float cosLightZenith = dot(stepState.upVector, renderParams.lightDir);
    vec3 tLightToSample = sampleTransmittanceLUT(atmosphere, cosLightZenith, stepState.sampleHeight);

    vec3 sampleExtinction = medium.extinction * stepState.sampleDensity;
    vec3 sampleOpticalDepth = sampleExtinction * rayStepLength;
    vec3 sampleTransmittance = exp(-sampleOpticalDepth);

    vec3 sampleInSctr = medium.phase * medium.scattering * stepState.sampleDensity;
    sampleInSctr *= tLightToSample;
    // See slide 28 at http://www.frostbite.com/2015/08/physically-based-unified-volumetric-rendering-in-frostbite/
    vec3 sampleInSctrInt = (sampleInSctr - sampleInSctr * sampleTransmittance) / sampleExtinction;

    accumState.totalInSctr += sampleInSctrInt;
    accumState.totalTransmittance *= sampleTransmittance;
}

#endif