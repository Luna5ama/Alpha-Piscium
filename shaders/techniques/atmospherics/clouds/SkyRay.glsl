#ifndef INCLUDE_clouds_SkyRay_glsl
#define INCLUDE_clouds_SkyRay_glsl

#include "CumulusRay.glsl"

ScatteringResult clouds_skyRay(AtmosphereParameters atmosphere, vec3 sceneOrigin, vec3 direction) {
    vec3 atmOrigin = atmosphere_sceneToAtm(atmosphere, sceneOrigin);
    AtmosphereParameters boundary = atmosphere;
    boundary.bottom -= 8.0;
    RaymarchParameters ray = RaymarchParameters(atmOrigin, atmOrigin, SETTING_SKY_SAMPLES);
    if (!setupRayEnd(boundary, ray, direction)) return scatteringResult_init();
    ScatteringParameters scattering = scatteringParameters_init(
        lightParameters_init(atmosphere, SUN_ILLUMINANCE, uval_sunDirWorld, direction),
        lightParameters_init(atmosphere, MOON_ILLUMINANCE, uval_moonDirWorld, direction), 1.0);
    ScatteringResult sky = raymarchSky(atmosphere, ray, scattering, 8.0);
    #ifdef SETTING_CLOUDS_CI
    float t = clouds_ci_distance(atmosphere, atmOrigin, direction);
    if (t >= 0.0 && t < length(ray.rayEnd - atmOrigin)) {
        ScatteringResult cirrus = clouds_ci_layer(atmosphere, atmOrigin + direction * t,
            clouds_renderParams(direction, 0.5), cloods_amblut_uv(direction, vec2(0.5)));
        ScatteringResult prefix = clouds_cu_airPrefix(atmosphere, scattering, atmOrigin, direction * t);
        sky.inScattering = cirrus.transmittance * sky.inScattering + (1.0 - cirrus.transmittance)
            * prefix.inScattering + prefix.transmittance * cirrus.inScattering;
        sky.transmittance *= cirrus.transmittance;
    }
    #endif
    #ifdef SETTING_CLOUDS_CU
    float meanDistance;
    ScatteringResult cloud = clouds_cu_render(atmosphere, clouds_cu_origin(sceneOrigin), atmOrigin,
        direction, vec2(0.0, CU_RANGE), vec4(0.5), true, meanDistance);
    sky = scatteringResult_blendLayer(sky, cloud, true);
    #endif
    return sky;
}

vec3 clouds_offscreenRadiance(vec3 sceneOrigin, vec3 direction) {
    AtmosphereParameters atmosphere = getAtmosphereParameters();
    SkyViewLutParams sky = atmospherics_air_lut_setupSkyViewLutParams(atmosphere, direction);
    #ifdef SETTING_CLOUDS_CU
    vec3 eye = gbufferModelViewInverse[3].xyz;
    vec3 origin = clouds_cu_origin(eye);
    vec3 displacement = (sceneOrigin - eye) * 0.001;
    vec2 eyeInterval;
    vec2 rayInterval;
    bool eyeHits = clouds_cu_interval(origin, direction, CU_RANGE, eyeInterval);
    bool rayHits = clouds_cu_interval(origin + displacement, direction, CU_RANGE, rayInterval);
    if (eyeHits != rayHits) return clouds_skyRay(atmosphere, sceneOrigin, direction).inScattering;
    if (rayHits && any(notEqual(displacement, vec3(0.0)))) {
        if (min(eyeInterval.x, rayInterval.x) <= length(displacement)) {
            return clouds_skyRay(atmosphere, sceneOrigin, direction).inScattering;
        }
        // Check parallax in the LUT's actual longitude and warped-latitude coordinates at the first possible cloud.
        vec3 apparent = normalize(displacement + direction * rayInterval.x);
        float lat, lon;
        _atmospherics_air_lut_rayDirToSkyViewLonLat(apparent, lat, lon);
        vec2 uv, apparentUV;
        _atmospherics_air_lut_skyViewLonLatToUv(sky.latitude, sky.longitude, uv);
        _atmospherics_air_lut_skyViewLonLatToUv(lat, lon, apparentUV);
        vec2 delta = abs(apparentUV - uv);
        delta.x = min(delta.x, 1.0 - delta.x);
        if (any(greaterThan(delta * SKYVIEW_LUT_SIZE_F, vec2(0.5)))) {
            return clouds_skyRay(atmosphere, sceneOrigin, direction).inScattering;
        }
    }
    #endif
    return atmospherics_air_lut_sampleSkyViewLUT(atmosphere, sky, 0.0).inScattering;
}

#endif
