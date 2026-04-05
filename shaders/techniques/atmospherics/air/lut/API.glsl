#ifndef INCLUDE_atmosphere_lut_API_glsl
#define INCLUDE_atmosphere_lut_API_glsl a
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

#include "Common.glsl"

vec3 atmospherics_air_lut_sampleTransmittance(AtmosphereParameters atmosphere, float cosLightZenith, float sampleAltitude) {
    vec2 tLUTUV;
    _atmospherics_air_lut_lutTransmittanceParamsToUv(atmosphere, sampleAltitude, cosLightZenith, tLUTUV);
    uint cond = uint(any(lessThan(tLUTUV, vec2(0.0))));
    cond |= uint(any(greaterThan(tLUTUV, vec2(1.0))));
    if (bool(cond)) {
        return vec3(0.0);
    }
    return persistent_transmittanceLUT_sample(tLUTUV).rgb;
}

vec3 atmospherics_air_lut_sampleMultiSctr(AtmosphereParameters atmosphere, float cosLightZenith, float sampleAltitude) {
    vec2 uv = vec2(saturate(cosLightZenith * 0.5 + 0.5), linearStep(atmosphere.bottom, atmosphere.top, sampleAltitude));
    uv = _atmospherics_air_lut_fromUnitToSubUvs(uv, vec2(MULTI_SCTR_LUT_SIZE));
    return persistent_multiSctrLUT_sample(uv).rgb / LUT_QUANTIZATION_MUL;
}

vec3 _atmospherics_air_lut_sampleSkyViewSlice(vec2 sliceUV, float sliceIndex) {
    vec3 sampleUV = vec3(sliceUV, (sliceIndex + 0.5) / SKYVIEW_LUT_DEPTH);
    return texture(usam_skyViewLUT, sampleUV).rgb;
}
#include "/util/Celestial.glsl"

// Sky View LUT lat/lon parameterisation
// Latitude  : elevation angle from horizon, in [-PI_HALF, PI_HALF]
// Longitude : stored pre-mapped to [0, 1] (u = lon directly)
//             lon = atan(-rayDir.x, rayDir.z) * RCP_PI_2 * 0.5 + 0.5
//             so Z+ (south, MC yaw=0) -> 0.5, X+ (east, yaw=-90) -> 0.25
ScatteringResult _atmospherics_air_lut_sampleSkyView(
    float lat,
    float lon,
    float layerIndex
) {
    vec2 sliceUV;
    _atmospherics_air_lut_skyViewLonLatToUv(lat, lon, sliceUV);

    float sctrSlice = layerIndex * 2.0;
    float tSlice = layerIndex * 2.0 + 1.0;

    ScatteringResult result = scatteringResult_init();
    result.inScattering = _atmospherics_air_lut_sampleSkyViewSlice(sliceUV, sctrSlice);
    result.transmittance = _atmospherics_air_lut_sampleSkyViewSlice(sliceUV, tSlice);
    return result;
}

struct SkyViewLutParams {
    float latitude;
    float longitude;
    float viewHeight;
};

SkyViewLutParams atmospherics_air_lut_setupSkyViewLutParams(
    AtmosphereParameters atmosphere,
    vec3 rayDirWorld
) {
    vec3 rayStart = atmosphere_viewToAtm(atmosphere, vec3(0.0));
    float viewHeight = length(rayStart);

    float lat, lon;
    _atmospherics_air_lut_rayDirToSkyViewLonLat(rayDirWorld, lat, lon);

    return SkyViewLutParams(lat, lon, viewHeight);
}

ScatteringResult atmospherics_air_lut_sampleSkyViewLUT(AtmosphereParameters atmosphere, SkyViewLutParams params, float layerIndex) {
    return _atmospherics_air_lut_sampleSkyView(params.latitude, params.longitude, layerIndex);
}

#endif