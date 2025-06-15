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

#include "/util/Colors.glsl"
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
    vec3 lightIrradiance;
    float LDotV;
};

CloudRenderParams cloudRenderParams_init(CloudRayParams rayParam, vec3 lightDir, vec3 lightIrradiance) {
    CloudRenderParams params;
    params.lightDir = normalize(lightDir);
    params.lightIrradiance = lightIrradiance;
    params.LDotV = -dot(rayParam.rayDir, lightDir);
    return params;
}

struct CloudParticpatingMedium {
    vec3 scattering;
    vec3 extinction;
    vec3 phase;
};

// See https://www.desmos.com/calculator/vzc8sfwbfv
vec3 samplePhaseLUT(float cosTheta, float type) {
    const vec4 COEFFS = vec4(0.0189677, 0.351847, 0.0946675, 0.147379);
    float x1 = -cosTheta;
    float x2 = x1 * x1;
    float x3 = x2 * x1;
    float x4 = x3 * x1;
    vec4 x = vec4(x4, x3, x2, x1);
    float u = dot(COEFFS, x) + 0.384577;
    float v = (type + 0.5) / 3.0;
    return colors_LogLuv32ToSRGB(texture(usam_cloudPhases, vec2(u, v)));
}

CloudParticpatingMedium clouds_cirrus_medium(float cosTheta) {
    CloudParticpatingMedium medium;
    medium.scattering = CLOUDS_CIRRUS_SCATTERING;
    medium.extinction = CLOUDS_CIRRUS_EXTINCTION;
    medium.phase = samplePhaseLUT(cosTheta, 0.0);
    return medium;
}

struct CloudRaymarchLayerParam {
    CloudParticpatingMedium medium;
    float rayStepLength;
    vec3 ambientIrradiance;
};

CloudRaymarchLayerParam clouds_raymarchLayerParam_init(
    CloudParticpatingMedium medium,
    float rayStepLength,
    vec3 ambientIrradiance
) {
    CloudRaymarchLayerParam param;
    param.medium = medium;
    param.rayStepLength = rayStepLength;
    param.ambientIrradiance = ambientIrradiance;
    return param;
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
    CloudRaymarchLayerParam layerParam,
    CloudRaymarchStepState stepState,
    inout CloudRaymarchAccumState accumState
) {
    float cosLightZenith = dot(stepState.upVector, renderParams.lightDir);
    vec3 tLightToSample = sampleTransmittanceLUT(atmosphere, cosLightZenith, stepState.sampleHeight);

    vec3 sampleExtinction = layerParam.medium.extinction * stepState.sampleDensity;
    vec3 sampleOpticalDepth = sampleExtinction * layerParam.rayStepLength;
    vec3 sampleTransmittance = exp(-sampleOpticalDepth);

    vec3 sampleTotalInSctrCoeff = layerParam.medium.scattering * stepState.sampleDensity;

    vec3 sampleInSctr = renderParams.lightIrradiance * tLightToSample * layerParam.medium.phase;
    sampleInSctr += layerParam.ambientIrradiance * accumState.totalTransmittance;
    sampleInSctr *= sampleTotalInSctrCoeff;
    // See slide 28 at http://www.frostbite.com/2015/08/physically-based-unified-volumetric-rendering-in-frostbite/
    vec3 sampleInSctrInt = (sampleInSctr - sampleInSctr * sampleTransmittance) / sampleExtinction;

    accumState.totalInSctr += sampleInSctrInt;
    accumState.totalTransmittance *= sampleTransmittance;
}

#endif