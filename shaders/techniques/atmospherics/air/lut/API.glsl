/*
    References:
        [EPI20] Epic Games, Inc. "Unreal Engine Sky Atmosphere Rendering Technique". 2020.
            MIT License. Copyright (c) 2020 Epic Games, Inc.
            https://github.com/sebh/UnrealEngineSkyAtmosphere
        [HIL20] Hillaire, Sébastien. "A Scalable and Production Ready Sky and Atmosphere Rendering Technique".
            EGSR 2020. 2020.
            https://sebh.github.io/publications/egsr2020.pdf

        You can find full license texts in /licenses
*/
#ifndef INCLUDE_atmosphere_lut_API_glsl
#define INCLUDE_atmosphere_lut_API_glsl a

#include "Common.glsl"

vec3 atmospherics_air_lut_sampleTransmittance(AtmosphereParameters atmosphere, float cosLightZenith, float sampleAltitude) {
    vec2 tLUTUV;
    _atmospherics_air_lut_lutTransmittanceParamsToUv(atmosphere, sampleAltitude, cosLightZenith, tLUTUV);
    uint cond = uint(any(lessThan(tLUTUV, vec2(0.0))));
    cond |= uint(any(greaterThan(tLUTUV, vec2(1.0))));
    if (bool(cond)) {
        return vec3(0.0);
    }
    return texture(usam_transmittanceLUT, tLUTUV).rgb;
}

vec3 atmospherics_air_lut_sampleMultiSctr(AtmosphereParameters atmosphere, float cosLightZenith, float sampleAltitude) {
    vec2 uv = saturate(vec2(cosLightZenith * 0.5 + 0.5, sampleAltitude / (atmosphere.top - atmosphere.bottom)));
    uv = _atmospherics_air_lut_fromUnitToSubUvs(uv, vec2(MULTI_SCTR_LUT_SIZE));
    // Hacky twilight multiple scattering fix
    return texture(usam_multiSctrLUT, uv).rgb * pow6(linearStep(-0.2, 0.0, cosLightZenith));
}

vec3 _atmospherics_air_lut_sampleSkyViewSlice(vec2 sliceUV, float sliceIndex) {
    vec3 sampleUV = vec3(sliceUV, (sliceIndex + 0.5) / SKYVIEW_LUT_DEPTH);
    return colors_LogLuv32ToSRGB(texture(usam_skyViewLUT, sampleUV));
}

ScatteringResult atmospherics_air_lut_sampleSkyView(
AtmosphereParameters atmosphere,
    bool intersectGround,
    float viewZenithCosAngle,
    float sunViewCosAngle,
    float MoonViewCosAngle,
    float viewHeight,
    float layerIndex
) {
    vec2 sunSliceUV;
    _atmospherics_air_lut_skyViewLutParamsToUv(
        atmosphere,
        intersectGround,
        viewZenithCosAngle,
        sunViewCosAngle,
        viewHeight,
        sunSliceUV
    );
    vec2 moonSliceUV;
    _atmospherics_air_lut_skyViewLutParamsToUv(
        atmosphere,
        intersectGround,
        viewZenithCosAngle,
        MoonViewCosAngle,
        viewHeight,
        moonSliceUV
    );
    float sunSlice = layerIndex * 3;
    float moonSlice = sunSlice + 1;
    float tSlice = sunSlice + 2;

    ScatteringResult result = scatteringResult_init();
    result.inScattering = _atmospherics_air_lut_sampleSkyViewSlice(sunSliceUV, sunSlice);
    result.inScattering += _atmospherics_air_lut_sampleSkyViewSlice(moonSliceUV, moonSlice);
    result.transmittance = _atmospherics_air_lut_sampleSkyViewSlice(sunSliceUV, tSlice);
    return result;
}

#endif