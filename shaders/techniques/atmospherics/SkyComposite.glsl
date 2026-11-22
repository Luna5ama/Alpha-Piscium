#include "clouds/Cirrus.glsl"
#include "clouds/ss/Common.glsl"

ScatteringResult atmospherics_skyComposite(ivec2 texelPos, vec3 viewPos) {
    ScatteringResult result = scatteringResult_init();
    #ifdef SETTING_CLOUDS_CU
    uvec4 encoded = transient_lowCloudAccumulated_fetch(texelPos);
    history_lowCloud_store(texelPos, encoded);
    #endif
    if (viewPos.z > -65536.0) return result;

    vec3 direction = normalize(mat3(gbufferModelViewInverse) * viewPos);
    AtmosphereParameters atmosphere = getAtmosphereParameters();
    SkyViewLutParams sky = atmospherics_air_lut_setupSkyViewLutParams(atmosphere, direction);
    vec3 origin = atmosphere_viewToAtm(atmosphere, vec3(0.0));
    bool aboveCirrus = length(origin) >= atmosphere.bottom + SETTING_CLOUDS_CI_HEIGHT;
    ScatteringResult below = atmospherics_air_lut_sampleSkyViewLUT(atmosphere, sky, 1.0);
    ScatteringResult above = atmospherics_air_lut_sampleSkyViewLUT(atmosphere, sky, 2.0);
    result = below;
    #ifdef SETTING_CLOUDS_CI
    float ciDistance = clouds_ci_distance(atmosphere, origin, direction);
    if (ciDistance >= 0.0) {
        vec2 ambUV = cloods_amblut_uv(direction, rand_stbnVec2(texelPos, frameCounter));
        ScatteringResult cirrus = clouds_ci_layer(atmosphere, origin + direction * ciDistance,
            clouds_renderParams(direction, 0.5), ambUV);
        result = scatteringResult_blendLayer(result, cirrus, aboveCirrus);
    }
    #endif
    result = scatteringResult_blendLayer(result, above, aboveCirrus);
    #ifdef SETTING_CLOUDS_CU
    CloudSSHistoryData cloud;
    clouds_ss_historyData_unpack(encoded, cloud);
    result = scatteringResult_blendLayer(result, ScatteringResult(cloud.transmittance, cloud.inScattering), true);
    #endif
    return result;
}
