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

#define MULTI_SCTR_LUT_QUANTIZATION_MUL 32.0

vec3 atmospherics_air_lut_sampleMultiSctr(AtmosphereParameters atmosphere, float cosLightZenith, float sampleAltitude) {
    vec2 uv = vec2(saturate(cosLightZenith * 0.5 + 0.5), linearStep(atmosphere.bottom, atmosphere.top, sampleAltitude));
    uv = _atmospherics_air_lut_fromUnitToSubUvs(uv, vec2(MULTI_SCTR_LUT_SIZE));
    return texture(usam_multiSctrLUT, uv).rgb / MULTI_SCTR_LUT_QUANTIZATION_MUL;
}

vec3 _atmospherics_air_lut_sampleSkyViewSlice(vec2 sliceUV, float sliceIndex) {
    vec3 sampleUV = vec3(sliceUV, (sliceIndex + 0.5) / SKYVIEW_LUT_DEPTH);
    return texture(usam_skyViewLUT, sampleUV).rgb;
}
#include "/util/Celestial.glsl"

ScatteringResult _atmospherics_air_lut_sampleSkyView(
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
    result.inScattering = _atmospherics_air_lut_sampleSkyViewSlice(sunSliceUV, sunSlice) * SUN_ILLUMINANCE;
    result.inScattering += _atmospherics_air_lut_sampleSkyViewSlice(moonSliceUV, moonSlice) * MOON_ILLUMINANCE;
    result.transmittance = _atmospherics_air_lut_sampleSkyViewSlice(sunSliceUV, tSlice);
    return result;
}

struct SkyViewLutParams {
    bool intersectGround;
    float viewZenithCosAngle;
    float sunViewCosAngle;
    float moonViewCosAngle;
    float viewHeight;
};

SkyViewLutParams atmospherics_air_lut_setupSkyViewLutParams(
    AtmosphereParameters atmosphere,
    vec3 rayDir
) {
    vec3 rayStart = atmosphere_viewToAtm(atmosphere, vec3(0.0));
    float viewHeight = length(rayStart);
    vec3 upVector = rayStart / viewHeight;

    float viewZenithCosAngle = dot(rayDir, upVector);

    const vec3 earthCenter = vec3(0.0);
    float tBottom = raySphereIntersectNearest(rayStart, rayDir, earthCenter, atmosphere.bottom);

    vec3 sideVector = normalize(cross(upVector, rayDir));		// assumes non parallel vectors
    vec3 forwardVector = normalize(cross(sideVector, upVector));	// aligns toward the sun light but perpendicular to up vector

    vec2 sunOnPlane = vec2(dot(uval_sunDirWorld, forwardVector), dot(uval_sunDirWorld, sideVector));
    sunOnPlane = normalize(sunOnPlane);
    float sunViewCosAngle = sunOnPlane.x;

    vec2 moonOnPlane = vec2(dot(uval_moonDirWorld, forwardVector), dot(uval_moonDirWorld, sideVector));
    moonOnPlane = normalize(moonOnPlane);
    float moonViewCosAngle = moonOnPlane.x;

    float horizonZenthCosAngle = -sqrt(saturate(1.0 - pow2(atmosphere.bottom / viewHeight)));
    bool intersectGround = viewZenithCosAngle < (horizonZenthCosAngle);

    return SkyViewLutParams(
        intersectGround,
        viewZenithCosAngle,
        sunViewCosAngle,
        moonViewCosAngle,
        viewHeight
    );
}

ScatteringResult atmospherics_air_lut_sampleSkyViewLUT(AtmosphereParameters atmosphere, SkyViewLutParams params, float layerIndex) {
    return _atmospherics_air_lut_sampleSkyView(
        atmosphere,
        params.intersectGround,
        params.viewZenithCosAngle,
        params.sunViewCosAngle,
        params.moonViewCosAngle,
        params.viewHeight,
        layerIndex
    );
}

#endif