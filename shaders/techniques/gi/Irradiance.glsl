#ifndef INCLUDE_techniques_restir_Irradiance_glsl
#define INCLUDE_techniques_restir_Irradiance_glsl a

#include "/techniques/atmospherics/air/lut/API.glsl"
#include "/util/Material.glsl"

vec3 restir_irradiance_sampleIrradianceMiss(vec3 worldDirection) {
    AtmosphereParameters atmosphere = getAtmosphereParameters();
    SkyViewLutParams skyParams = atmospherics_air_lut_setupSkyViewLutParams(atmosphere, worldDirection);
    return atmospherics_air_lut_sampleSkyViewLUT(atmosphere, skyParams, 0.0).inScattering;
}

vec3 restir_irradiance_sampleIrradiance(ivec2 texelPos, Material selfMaterial, ivec2 hitTexelPos, vec3 outgoingDirection) {
    vec4 hitGeomNormalData = transient_geomViewNormal_fetch(hitTexelPos);
    vec4 hitRadianceData = transient_giRadianceInput1_fetch(hitTexelPos);
    vec4 hitEmissiveData = transient_giRadianceInput2_fetch(hitTexelPos);

    vec3 hitGeomNormal = normalize(hitGeomNormalData.xyz * 2.0 - 1.0);
    vec3 hitRadiance = hitRadianceData.rgb;
    vec3 hitEmissive = hitEmissiveData.rgb;
    float hitCosTheta = saturate(dot(hitGeomNormal, outgoingDirection));

    return hitRadiance * float(hitCosTheta > 0.0) + hitEmissive * float(all(lessThan(selfMaterial.emissive, vec3(0.0001))));
}

#endif