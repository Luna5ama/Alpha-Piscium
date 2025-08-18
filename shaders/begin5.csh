#version 460 compatibility

#include "/techniques/atmospherics/air/lut/API.glsl"

layout(local_size_x = 8, local_size_y = 16) in;
const ivec3 workGroups = ivec3(SKYVIEW_RES_D16, SKYVIEW_RES_D16, 1);

layout(rgba8) restrict uniform image3D uimg_skyViewLUT;

const float[3] BOUND = {
    -100.0,
    SETTING_CLOUDS_CU_HEIGHT,
    SETTING_CLOUDS_CI_HEIGHT
};

void main() {
    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);
    vec2 sliceUV = (vec2(texelPos) + vec2(0.5)) / SKYVIEW_LUT_SIZE_F;

    vec3 sunInSctr = vec3(0.0);
    vec3 moonInSctr = vec3(0.0);
    vec3 transmittance = vec3(1.0);

    AtmosphereParameters atmosphere = getAtmosphereParameters();
    vec3 rayStart = atmosphere_viewToAtm(atmosphere, vec3(0.0));
    float viewHeight = length(rayStart) - atmosphere.bottom;

    for (int i = 0; i < 3; i++) {
        int sliceIndex = (i + 1) * 3;
        vec3 layerSunInSctr = _atmospherics_air_lut_sampleSkyViewSlice(sliceUV, float(sliceIndex));
        vec3 layerMoonInSctr = _atmospherics_air_lut_sampleSkyViewSlice(sliceUV, float(sliceIndex + 1));
        vec3 layerTransmittance = _atmospherics_air_lut_sampleSkyViewSlice(sliceUV, float(sliceIndex + 2));

        float bound = BOUND[i];

        if (viewHeight >= bound) {
            sunInSctr = sunInSctr * layerTransmittance + layerSunInSctr;
            moonInSctr = moonInSctr * layerTransmittance + layerMoonInSctr;
            transmittance *= layerTransmittance;
        } else {
            sunInSctr = layerSunInSctr * transmittance + sunInSctr;
            moonInSctr = layerMoonInSctr * transmittance + moonInSctr;
            transmittance *= layerTransmittance;
        }
    }

    ivec3 writePos = ivec3(texelPos, 4 * 3);
    imageStore(uimg_skyViewLUT, writePos, colors_sRGBToLogLuv32(sunInSctr));
    writePos.z += 1;
    imageStore(uimg_skyViewLUT, writePos, colors_sRGBToLogLuv32(moonInSctr));
    writePos.z += 1;
    imageStore(uimg_skyViewLUT, writePos, colors_sRGBToLogLuv32(transmittance));
}