#ifndef INCLUDE_techniques_restir_Irradiance_glsl
#define INCLUDE_techniques_restir_Irradiance_glsl a

#include "/techniques/atmospherics/air/lut/API.glsl"
#include "/util/GBufferData.glsl"

vec3 restir_irradiance_sampleIrradianceMiss(vec3 worldDirection) {
    AtmosphereParameters atmosphere = getAtmosphereParameters();
    SkyViewLutParams skyParams = atmospherics_air_lut_setupSkyViewLutParams(atmosphere, worldDirection);
    return atmospherics_air_lut_sampleSkyViewLUT(atmosphere, skyParams, 0.0).inScattering;
}

vec3 restir_irradiance_sampleIrradiance(ivec2 texelPos, ivec2 hitTexelPos, vec3 outgoingDirection) {
    GBufferData hitGData = gbufferData_init();
    gbufferData1_unpack(texelFetch(usam_gbufferData1, hitTexelPos, 0), hitGData);

    float hitCosTheta = saturate(dot(hitGData.geomNormal, outgoingDirection));
    vec3 hitRadiance = transient_giRadianceInput1_fetch(hitTexelPos).rgb;
    vec3 hitEmissive = transient_giRadianceInput2_fetch(hitTexelPos).rgb;
    vec3 selfHitEmissive = transient_giRadianceInput2_fetch(texelPos).rgb;

    return hitRadiance * float(hitCosTheta > 0.0) + hitEmissive * float(all(lessThan(selfHitEmissive, vec3(0.0001))));
}

#endif