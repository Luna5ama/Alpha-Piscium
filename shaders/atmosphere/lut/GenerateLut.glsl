#ifndef INCLUDE_atmosphere_lut_GenerateLut_glsl
#define INCLUDE_atmosphere_lut_GenerateLut_glsl a

#include "/atmosphere/Common.glsl"
#include "/atmosphere/lut/Common.glsl"
#include "/atmosphere/Raymarching.glsl"
#include "/util/Celestial.glsl"
#include "/util/Coords.glsl"

// originView: ray origin in view space
// endView: ray end in view space
ScatteringResult lut_computeSingleScattering(AtmosphereParameters atmosphere, vec3 rayDir, vec3 origin) {
    ScatteringResult result = scatteringResult_init();
    if (all(equal(rayDir, vec3(0.0)))) {
        return result;
    }

    vec3 originView = origin;

    RaymarchParameters params = raymarchParameters_init();
    params.rayStart = atmosphere_viewToAtm(atmosphere, originView);
    params.rayStart.y = max(params.rayStart.y, atmosphere.bottom + 0.5);
    params.steps = SETTING_SKY_SAMPLES;

    LightParameters sunParams = lightParameters_init(atmosphere, SUN_ILLUMINANCE * PI, uval_sunDirWorld, rayDir);
    LightParameters moonParams = lightParameters_init(atmosphere, MOON_ILLUMINANCE, uval_moonDirWorld, rayDir);
    ScatteringParameters scatteringParams = scatteringParameters_init(sunParams, moonParams, 1.0);

    const vec3 earthCenter = vec3(0.0);

    if (setupRayEnd(atmosphere, params, rayDir)) {
        result = raymarchSky(atmosphere, params, scatteringParams);
    }

    return result;
}

#endif