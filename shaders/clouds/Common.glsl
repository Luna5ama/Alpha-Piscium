/*
    References:
        [HIL16] Hillaire, Sébastien. "Physically Based Sky, Atmosphere and Cloud Rendering in Frostbite".
            SIGGRAPH 2016. 2016.
            https://www.ea.com/frostbite/news/physically-based-sky-atmosphere-and-cloud-rendering
        [SCH15] Schneider, Andrew. "The Real-Time Volumetric Cloudscapes Of Horizon: Zero Dawn"
            SIGGRAPH 2015. 2015.
            https://www.guerrilla-games.com/read/the-real-time-volumetric-cloudscapes-of-horizon-zero-dawn
        [SCH16] Schneider, Andrew. "Real-Time Volumetric Cloudscapes"
            GPU Pro 7. 2016.
        [SCH17] Schneider, Andrew. "Nubis: Authoring Real-Time Volumetric Cloudscapes with the Decima Engine"
            SIGGRAPH 17. 2017.
            https://www.guerrilla-games.com/read/nubis-authoring-real-time-volumetric-cloudscapes-with-the-decima-engine
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
#include "Mediums.glsl"

struct CloudMainRayParams {
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

CloudRenderParams cloudRenderParams_init(CloudMainRayParams rayParam, vec3 lightDir, vec3 lightIrradiance) {
    CloudRenderParams params;
    params.lightDir = normalize(lightDir);
    params.lightIrradiance = lightIrradiance;
    params.LDotV = -dot(rayParam.rayDir, lightDir);
    return params;
}

struct CloudRaymarchLayerParam {
    CloudParticpatingMedium medium;
    vec3 ambientIrradiance;
    vec2 layerRange;
    vec3 rayStart;
    vec3 rayEnd;
    vec4 rayStep;
};

CloudRaymarchLayerParam clouds_raymarchLayerParam_init(
    CloudMainRayParams mainRayParam,
    CloudParticpatingMedium medium,
    vec3 ambientIrradiance,
    vec2 layerRange,
    float origin2RayOffset,
    float rayLength,
    float rayRcpStepCount
) {
    CloudRaymarchLayerParam param;
    param.medium = medium;
    param.ambientIrradiance = ambientIrradiance;
    param.layerRange = layerRange;
    param.rayStart = mainRayParam.rayStart + mainRayParam.rayDir * origin2RayOffset;
    param.rayEnd = param.rayStart + mainRayParam.rayDir * rayLength;
    param.rayStep = vec4(param.rayEnd - param.rayStart, rayLength) * rayRcpStepCount;
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
    vec4 position;
    float height;
    vec4 rayStep;
    vec3 upVector;
};

CloudRaymarchStepState clouds_raymarchStepState_init(CloudRaymarchLayerParam layerParam) {
    CloudRaymarchStepState state;
    state.position = vec4(layerParam.rayStart, 0.0);
    state.height = length(state.position.xyz);
    state.upVector = state.position.xyz / state.height;
    state.rayStep = layerParam.rayStep;
    return state;
}

void clouds_raymarchStepState_update(
    inout CloudRaymarchStepState state
) {
    state.position += state.rayStep;
    state.height = length(state.position.xyz);
    state.upVector = state.position.xyz / state.height;
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
    float sampleDensity,
    vec3 lightTransmittance,
    inout CloudRaymarchAccumState accumState
) {
    float cosLightZenith = dot(stepState.upVector, renderParams.lightDir);
    vec3 tLightToSample = sampleTransmittanceLUT(atmosphere, cosLightZenith, stepState.height);

    vec3 sampleLightIrradiance = renderParams.lightIrradiance;
    sampleLightIrradiance *= tLightToSample * lightTransmittance;
    vec3 sampleAmbientIrradiance = layerParam.ambientIrradiance;
    sampleAmbientIrradiance *= accumState.totalTransmittance * mix(lightTransmittance, vec3(1.0), 0.5);

    vec3 sampleScattering = layerParam.medium.scattering * sampleDensity;
    vec3 sampleExtinction = layerParam.medium.extinction * sampleDensity;
    vec3 sampleOpticalDepth = sampleExtinction * stepState.rayStep.w;
    // See [SCH17]
    vec3 sampleTransmittance = max(exp(-sampleOpticalDepth), exp(-sampleOpticalDepth * 0.25) * 0.7);

    vec4 multSctrFalloffs = vec4(1.0);

    vec3 sampleTotalInSctr = vec3(0.0);

    // See [HIL16] and [QIU25]
    for (uint i = 0; i < SETTING_CLOUDS_MS_ORDER; i++) {
        vec3 sampleScatteringMS = sampleScattering * multSctrFalloffs.x;
        vec3 sampleExtinctionMS = sampleExtinction * multSctrFalloffs.y;
        vec3 samplePhaseMS = mix(vec3(UNIFORM_PHASE), layerParam.medium.phase, multSctrFalloffs.z);
        vec3 sampleAmbientIrradianceMS = sampleAmbientIrradiance * multSctrFalloffs.w;

        vec3 sampleInSctr = sampleLightIrradiance * samplePhaseMS;
        sampleInSctr += sampleAmbientIrradianceMS;
        sampleInSctr *= sampleScatteringMS;
        // See slide 28 at http://www.frostbite.com/2015/08/physically-based-unified-volumetric-rendering-in-frostbite/
        vec3 sampleInSctrInt = (sampleInSctr - sampleInSctr * sampleTransmittance) / sampleExtinctionMS;

        sampleTotalInSctr += sampleInSctrInt;
        multSctrFalloffs *= _CLOUDS_MS_FALLOFFS;
    }

    accumState.totalInSctr += sampleTotalInSctr * accumState.totalTransmittance;
    accumState.totalTransmittance *= sampleTransmittance;
}

#endif