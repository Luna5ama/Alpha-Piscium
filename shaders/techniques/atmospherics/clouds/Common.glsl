#ifndef INCLUDE_clouds_Common_glsl
#define INCLUDE_clouds_Common_glsl a
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
        [WDT22] Wo De Tian (oh my god). "一个简单的体积云多重散射近似方法" (A Simple Multiple Scattering Approximation for Volumetric Clouds). 2022.
            https://zhuanlan.zhihu.com/p/457997155

        You can find full license texts in /licenses
*/

#include "Mediums.glsl"
#include "/techniques/atmospherics/air/Constants.glsl"
#include "/techniques/atmospherics/air/lut/API.glsl"
#include "/util/Colors.glsl"

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
    float cosLightTheta;
};

CloudRenderParams cloudRenderParams_init(CloudMainRayParams rayParam, vec3 lightDir, vec3 lightIrradiance) {
    CloudRenderParams params;
    params.lightDir = normalize(lightDir);
    params.lightIrradiance = lightIrradiance;
    params.cosLightTheta = dot(rayParam.rayDir, lightDir);
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
    uint stepCount
) {
    CloudRaymarchLayerParam param;
    param.medium = medium;
    param.ambientIrradiance = ambientIrradiance;
    param.layerRange = layerRange;
    param.rayStart = mainRayParam.rayStart + mainRayParam.rayDir * origin2RayOffset;
    param.rayEnd = param.rayStart + mainRayParam.rayDir * rayLength;
    param.rayStep = vec4(param.rayEnd - param.rayStart, rayLength) * rcp(float(stepCount + 1u));
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
    vec4 origin;
    vec4 position;
    float height;
    vec4 rayStep;
    vec3 upVector;
};

CloudRaymarchStepState clouds_raymarchStepState_init(CloudRaymarchLayerParam layerParam) {
    CloudRaymarchStepState state;
    state.origin = vec4(layerParam.rayStart, 0.0);
    state.position = state.origin;
    state.height = length(state.position.xyz);
    state.upVector = state.position.xyz / state.height;
    state.rayStep = layerParam.rayStep;
    return state;
}

void clouds_raymarchStepState_update(inout CloudRaymarchStepState state, float stepCount) {
    state.position = state.origin + state.rayStep * stepCount;
    state.height = length(state.position.xyz);
    state.upVector = state.position.xyz / state.height;
}

void clouds_computeLighting(
    AtmosphereParameters atmosphere,
    CloudRenderParams renderParams,
    CloudRaymarchLayerParam layerParam,
    CloudRaymarchStepState stepState,
    float sampleDensity,
    vec3 lightOpticalDepth,
    inout CloudRaymarchAccumState accumState
) {
    float cosLightZenith = dot(stepState.upVector, renderParams.lightDir);
    vec3 tLightToSample = atmospherics_air_lut_sampleTransmittance(atmosphere, cosLightZenith, stepState.height);

    vec3 sampleScattering = layerParam.medium.scattering * sampleDensity;
    vec3 sampleExtinction = layerParam.medium.extinction * sampleDensity;
    vec3 sampleOpticalDepth = sampleExtinction * stepState.rayStep.w;
    // See [SCH17]
    vec3 sampleTransmittance = exp(-sampleOpticalDepth);

    vec3 sampleLightIrradiance = renderParams.lightIrradiance;
    sampleLightIrradiance *= tLightToSample * exp(-lightOpticalDepth);
    vec3 sampleAmbientIrradiance = layerParam.ambientIrradiance;

    vec3 ambLightOpticalDepth = lightOpticalDepth;
    ambLightOpticalDepth += -log(accumState.totalTransmittance);
    ambLightOpticalDepth += sampleOpticalDepth;
    ambLightOpticalDepth /= 3.0;
    // See [SCH17]
    vec3 ambientTransmittance = max(exp(-ambLightOpticalDepth), exp(-ambLightOpticalDepth * 0.25) * 0.7);

    sampleAmbientIrradiance *= ambientTransmittance;

    vec4 multSctrFalloffs = vec4(1.0);

    vec3 sampleTotalInSctr = vec3(0.0);

    vec3 sampleInSctr = sampleLightIrradiance * layerParam.medium.phase;
    sampleInSctr += sampleAmbientIrradiance;

    const float D = SETTING_CLOUDS_MS_RADIUS;
    vec3 fMS = (sampleScattering / sampleExtinction) * (1.0 - exp(-D * sampleExtinction));
    fMS = mix(fMS, fMS * 0.99, smoothstep(0.99, 1.0, fMS));
    vec3 sampleMSIrradiance = sampleLightIrradiance;
    sampleMSIrradiance += sampleAmbientIrradiance;
    sampleMSIrradiance *= UNIFORM_PHASE;
    sampleMSIrradiance *= fMS / (1.0 - fMS);
    sampleInSctr += sampleMSIrradiance;

    sampleInSctr *= sampleScattering;

    sampleInSctr += sampleMSIrradiance;

    vec3 sampleInSctrInt = (sampleInSctr - sampleInSctr * sampleTransmittance) / sampleExtinction;
    sampleTotalInSctr += sampleInSctrInt;

    accumState.totalInSctr += sampleTotalInSctr * accumState.totalTransmittance;
    accumState.totalTransmittance *= sampleTransmittance;
}

#endif