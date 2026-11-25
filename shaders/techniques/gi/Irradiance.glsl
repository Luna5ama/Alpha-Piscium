#ifndef INCLUDE_techniques_restir_Irradiance_glsl
#define INCLUDE_techniques_restir_Irradiance_glsl a

#include "/techniques/atmospherics/air/lut/API.glsl"
#include "/techniques/EnvProbe.glsl"
#include "/util/Colors.glsl"
#include "/util/Material.glsl"

vec3 gi_environmentRadiance(ivec2 texelPos, vec3 rayOriginScene, vec3 worldDirection) {
    AtmosphereParameters atmosphere = getAtmosphereParameters();
    SkyViewLutParams skyParams = atmospherics_air_lut_setupSkyViewLutParams(atmosphere, worldDirection);
    vec3 skyIrradiance = atmospherics_air_lut_sampleSkyViewLUT(atmosphere, skyParams, 0.0).inScattering;
    #ifdef SETTING_GI_MC_SKYLIGHT_ATTENUATION
    float lmCoordSky = transient_lmCoord_fetch(texelPos).y;
    float skyLightFactor = max(lmCoordSky, linearStep(0.0, 240.0, float(eyeBrightnessSmooth.y)));
    skyIrradiance *= skyLightFactor;
    #endif
    #if SETTING_GI_USE_REFERENCE == 2
    return skyIrradiance;
    #else
    vec2 uv;
    vec2 face;
    coords_cubeMapForward(worldDirection, uv, face);
    ivec2 pixel = ivec2(face) * ENV_PROBE_SIZEI + clamp(ivec2(uv * ENV_PROBE_SIZE), ivec2(0), ENV_PROBE_SIZEI - 1);
    EnvProbeData probe = envProbe_decode(persistent_envProbe_fetch(pixel));
    if (envProbe_isSky(probe)) return skyIrradiance;
    float distanceToProbe = distance(probe.scenePos, rayOriginScene);
    return mix(probe.radiance, skyIrradiance, linearStep(SETTING_GI_PROBE_FADE_START, SETTING_GI_PROBE_FADE_END, distanceToProbe));
    #endif
}

#endif
