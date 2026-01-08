#ifndef INCLUDE_techniques_restir_Irradiance_glsl
#define INCLUDE_techniques_restir_Irradiance_glsl a

#include "/techniques/atmospherics/air/lut/API.glsl"
#include "/techniques/EnvProbe.glsl"
#include "/util/Colors.glsl"
#include "/util/Material.glsl"

// xyz: radiance
// w: hit distance (or -1.0 for sky)
vec3 restir_irradiance_sampleIrradianceMiss(ivec2 texelPos, vec3 rayOriginScene, vec3 worldDirection) {
    vec2 envSliceUV = vec2(-1.0);
    vec2 envSliceID = vec2(-1.0);
    coords_cubeMapForward(worldDirection, envSliceUV, envSliceID);
    ivec2 envTexel = ivec2((envSliceUV + envSliceID) * ENV_PROBE_SIZE);
    EnvProbeData envData = envProbe_decode(texelFetch(usam_envProbe, envTexel, 0));

    float envProbeDistance = distance(envData.scenePos, rayOriginScene);

    AtmosphereParameters atmosphere = getAtmosphereParameters();
    SkyViewLutParams skyParams = atmospherics_air_lut_setupSkyViewLutParams(atmosphere, worldDirection);
    vec3 skyIrradiance = atmospherics_air_lut_sampleSkyViewLUT(atmosphere, skyParams, 0.0).inScattering;
    #ifdef SETTING_GI_MC_SKYLIGHT_ATTENUATION
    float lmCoordSky = transient_lmCoord_fetch(texelPos).y;
    float skyLightFactor = max(lmCoordSky, linearStep(0.0, 240.0, float(eyeBrightnessSmooth.y)));
    skyIrradiance *= skyLightFactor;
    #endif

    vec3 result = vec3(0.0);
    if (envProbe_isSky(envData)) {
        result.rgb = skyIrradiance;
    } else {
        float skyMixWeight = linearStep(SETTING_GI_PROBE_FADE_START, SETTING_GI_PROBE_FADE_END, length(rayOriginScene));
        result.rgb = mix(envData.radiance.rgb, skyIrradiance, skyMixWeight);
    }

    return result;
}

vec3 restir_irradiance_sampleIrradiance(ivec2 texelPos, Material selfMaterial, ivec2 hitTexelPos, vec3 outgoingDirection) {
    vec4 hitGeomNormalData = transient_geomViewNormal_fetch(hitTexelPos);
    uvec4 hitRadianceData = transient_giRadianceInputs_fetch(hitTexelPos);

    vec3 hitGeomNormal = normalize(hitGeomNormalData.xyz * 2.0 - 1.0);
    vec3 hitRadiance = colors_FP16LuvToWorkingColor(hitRadianceData.x);
    vec3 hitEmissive = colors_FP16LuvToWorkingColor(hitRadianceData.y);
    float hitCosTheta = saturate(dot(hitGeomNormal, outgoingDirection));

    return hitRadiance * float(hitCosTheta > 0.0) + hitEmissive * float(all(lessThan(selfMaterial.emissive, vec3(0.0001))));
}

#endif