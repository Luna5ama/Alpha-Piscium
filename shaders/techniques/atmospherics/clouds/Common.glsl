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
        [ASH25] AshenOneArt. "HanPi Volume Cloud" isotropic multiple-scattering derivation.
            https://github.com/AshenOneArt/HPVolumeCloud/blob/27e799914493de9fa527179312ed72a39d08e225/Docs/PhiFwd_FromRTE.md

        You can find full license texts in /licenses
*/

#include "Mediums.glsl"
#include "/util/Celestial.glsl"
#include "/techniques/atmospherics/air/Constants.glsl"
#include "/techniques/atmospherics/air/lut/API.glsl"
#include "/util/Colors.glsl"

struct CloudRenderParams {
    vec3 lightDir;
    vec3 lightIrradiance;
    float cosLightTheta;
};

CloudRenderParams clouds_renderParams(vec3 direction, float jitter) {
    float angle = fract(sunAngle + 0.25);
    float daylight = smoothstep(0.23035, 0.24035, angle) * (1.0 - smoothstep(0.75965, 0.76965, angle));
    bool sun = daylight + (jitter - 0.5) * 0.2 >= 0.5;
    vec3 lightDir = sun ? uval_sunDirWorld : uval_moonDirWorld;
    return CloudRenderParams(lightDir, sun ? SUN_ILLUMINANCE : MOON_ILLUMINANCE, dot(direction, lightDir));
}

ScatteringResult clouds_computeLighting(
    AtmosphereParameters atmosphere,
    CloudRenderParams renderParams,
    CloudParticpatingMedium medium,
    vec3 ambientIrradiance,
    vec3 atmPos,
    float stepLength,
    float sampleDensity,
    float sampleDensityLod,
    vec3 lightOpticalDepth,
    vec3 sampleIsotropicMSIrradiance
) {
    float height = length(atmPos);
    float cosLightZenith = dot(atmPos / height, renderParams.lightDir);
    vec3 tLightToSample = atmospherics_air_lut_sampleTransmittance(atmosphere, cosLightZenith, height);

    vec3 sampleScattering = medium.scattering * sampleDensity;
    vec3 sampleExtinction = medium.extinction * sampleDensity;
    vec3 sampleOpticalDepth = sampleExtinction * stepLength;
    // See [SCH17]
    vec3 sampleTransmittance = exp(-sampleOpticalDepth);

    vec3 sampleLightIrradiance = renderParams.lightIrradiance;
    vec3 sampleLightTransmittance = exp(-lightOpticalDepth);
    sampleLightIrradiance *= tLightToSample * sampleLightTransmittance;

    float D = exp2(SETTING_CLOUDS_MS_RADIUS);

    // TODO: Expose the density mix as setting
    float ambLightDesnity = mix(sampleDensityLod, sampleDensity, 0.8);
    vec3 ambLightOpticalDepth = medium.extinction * ambLightDesnity;
    vec3 ambientTransmittance = exp(-ambLightOpticalDepth);

    vec3 sampleAmbientIrradiance = ambientIrradiance;
    sampleAmbientIrradiance *= ambientTransmittance;

    vec3 sampleIrradiance = sampleLightIrradiance;
    sampleIrradiance *= medium.phase;
    sampleIrradiance += sampleAmbientIrradiance;
    vec3 msPhase = mix(vec3(UNIFORM_PHASE), medium.phase, 0.7);
    sampleIrradiance += renderParams.lightIrradiance * tLightToSample * sampleIsotropicMSIrradiance * msPhase;

    vec3 fMS = (sampleScattering / sampleExtinction) * (1.0 - exp(-D * sampleExtinction));
    fMS = mix(fMS, fMS * 0.95, linearStep(0.9, 1.0, fMS));
    vec3 sampleMSIrradiance = sampleLightIrradiance;
    sampleMSIrradiance *= UNIFORM_PHASE;
    sampleMSIrradiance += sampleAmbientIrradiance;
    sampleMSIrradiance *= fMS / (1.0 - fMS);
    sampleIrradiance += sampleMSIrradiance;

    vec3 sampleInSctr = sampleIrradiance * sampleScattering;

    vec3 sampleInSctrInt = (sampleInSctr - sampleInSctr * sampleTransmittance) / sampleExtinction;

    return ScatteringResult(sampleTransmittance, sampleInSctrInt);
}


#endif
