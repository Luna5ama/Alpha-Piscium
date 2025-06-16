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

// See https://www.desmos.com/calculator/yerfmyqpuh
vec3 samplePhaseLUT(float cosTheta, float type) {
    const float a0 = 0.672617934627;
    const float a1 = -0.0713555761181;
    const float a2 = 0.0299320735609;
    const float b = 0.264767018876;
    float x1 = acos(-cosTheta);
    float x2 = x1 * x1;
    float u = saturate((a0 + a1 * x1 + a2 * x2) * pow(x1, b));
    float v = (type + 0.5) / 3.0;
    return colors_LogLuv32ToSRGB(texture(usam_cloudPhases, vec2(u, v)));
}

CloudParticpatingMedium clouds_cirrus_medium(float cosTheta) {
    CloudParticpatingMedium medium;
    medium.scattering = CLOUDS_CIRRUS_SCATTERING;
    medium.extinction = CLOUDS_CIRRUS_EXTINCTION;
    medium.phase = mix(cornetteShanksPhase(cosTheta, CLOUDS_CIRRUS_ASYM), samplePhaseLUT(cosTheta, 0.0), SETTING_CIRRUS_PHASE_RATIO);
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

const vec4 _CLOUDS_MS_FALLOFFS = vec4(
    SETTING_CLOUDS_MS_FALLOFF_SCTTERING,
    SETTING_CLOUDS_MS_FALLOFF_EXTINCTION,
    SETTING_CLOUDS_MS_FALLOFF_PHASE,
    SETTING_CLOUDS_MS_FALLOFF_AMB
);

void clouds_computeLighting(
    AtmosphereParameters atmosphere,
    CloudRenderParams renderParams,
    CloudRaymarchLayerParam layerParam,
    CloudRaymarchStepState stepState,
    inout CloudRaymarchAccumState accumState
) {
    float cosLightZenith = dot(stepState.upVector, renderParams.lightDir);
    vec3 tLightToSample = sampleTransmittanceLUT(atmosphere, cosLightZenith, stepState.sampleHeight);

    vec3 sampleLightIrradiance = renderParams.lightIrradiance * tLightToSample;
    vec3 sampleAmbientIrradiance = layerParam.ambientIrradiance * accumState.totalTransmittance;

    vec3 sampleScattering = layerParam.medium.scattering * stepState.sampleDensity;
    vec3 sampleExtinction = layerParam.medium.extinction * stepState.sampleDensity;
    vec3 sampleOpticalDepth = sampleExtinction * layerParam.rayStepLength;

    vec3 sampleScatteringMS = sampleScattering;
    vec3 sampleOpticalDepthMS = sampleOpticalDepth;
    vec3 samplePhaseMS = layerParam.medium.phase;
    vec4 multSctrFalloffs = _CLOUDS_MS_FALLOFFS;

    // See [HIL16] and [QIU25]
    for (uint i = 0; i < SETTING_CLOUDS_MS_ORDER; i++) {
        vec3 sampleTransmittanceMS = exp(-sampleOpticalDepthMS);

        vec3 sampleInSctr = sampleLightIrradiance * samplePhaseMS;
        sampleInSctr += sampleAmbientIrradiance * multSctrFalloffs.w;
        sampleInSctr *= sampleScatteringMS;
        // See slide 28 at http://www.frostbite.com/2015/08/physically-based-unified-volumetric-rendering-in-frostbite/
        vec3 sampleInSctrInt = (sampleInSctr - sampleInSctr * sampleTransmittanceMS) / sampleOpticalDepthMS;

        accumState.totalInSctr += sampleInSctrInt;

        sampleScatteringMS *= multSctrFalloffs.x;
        sampleOpticalDepthMS *= multSctrFalloffs.y;
        samplePhaseMS = mix(vec3(UNIFORM_PHASE), layerParam.medium.phase, multSctrFalloffs.z);
        multSctrFalloffs *= multSctrFalloffs;
    }

    accumState.totalTransmittance *= exp(-sampleOpticalDepth);
}

#endif