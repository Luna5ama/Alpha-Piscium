#ifndef INCLUDE_clouds_ss_Common_glsl
#define INCLUDE_clouds_ss_Common_glsl a

#include "/util/Colors.glsl"
#include "/util/Rand.glsl"

#if SETTING_CLOUDS_LOW_UPSCALE_FACTOR == 0
#define UPSCALE_FACTOR 1.0
#define DOWNSCALE_DIVIDE(x) x
#define RENDER_MULTIPLIER 1.0
#define CHECK_MIP_LEVEL 3
#define CHECK_MIP_FACTOR 8.0

#elif SETTING_CLOUDS_LOW_UPSCALE_FACTOR == 1
#define UPSCALE_FACTOR 1.5
#define DOWNSCALE_DIVIDE(x) (x * 2 / 3)
#define RENDER_MULTIPLIER 0.6666666667
#define CHECK_MIP_LEVEL 3
#define CHECK_MIP_FACTOR 8.0

#elif SETTING_CLOUDS_LOW_UPSCALE_FACTOR == 2
#define UPSCALE_FACTOR 2.0
#define DOWNSCALE_DIVIDE(x) (x / 2)
#define RENDER_MULTIPLIER 0.5
#define CHECK_MIP_LEVEL 4
#define CHECK_MIP_FACTOR 16.0

#elif SETTING_CLOUDS_LOW_UPSCALE_FACTOR == 3
#define UPSCALE_FACTOR 2.5
#define DOWNSCALE_DIVIDE(x) (x * 2 / 5)
#define RENDER_MULTIPLIER 0.4
#define CHECK_MIP_LEVEL 4
#define CHECK_MIP_FACTOR 16.0

#elif SETTING_CLOUDS_LOW_UPSCALE_FACTOR == 4
#define UPSCALE_FACTOR 3.0
#define DOWNSCALE_DIVIDE(x) (x / 3)
#define RENDER_MULTIPLIER 0.3333333333
#define CHECK_MIP_LEVEL 5
#define CHECK_MIP_FACTOR 32.0

#elif SETTING_CLOUDS_LOW_UPSCALE_FACTOR == 5
#define UPSCALE_FACTOR 3.5
#define DOWNSCALE_DIVIDE(x) (x * 2 / 7)
#define RENDER_MULTIPLIER 0.2857142857
#define CHECK_MIP_LEVEL 5
#define CHECK_MIP_FACTOR 32.0

#elif SETTING_CLOUDS_LOW_UPSCALE_FACTOR == 6
#define UPSCALE_FACTOR 4.0
#define DOWNSCALE_DIVIDE(x) (x / 4)
#define RENDER_MULTIPLIER 0.25
#define CHECK_MIP_LEVEL 6
#define CHECK_MIP_FACTOR 64.0

#endif

ivec2 renderSize = DOWNSCALE_DIVIDE(uval_mainImageSizeI);

vec2 clouds_ss_upscaleoffset() {
    return rand_r2Seq2(frameCounter);
}

vec2 clouds_ss_upscaledTexelCenter(ivec2 texelPosDownScale) {
    vec2 texelPos1x1F = vec2(texelPosDownScale * UPSCALE_FACTOR);
    vec2 offset = clouds_ss_upscaleoffset() * UPSCALE_FACTOR;
    return clamp(texelPos1x1F + offset, vec2(0.0), uval_mainImageSize);
}

#define CLOUDS_SS_MAX_ACCUM SETTING_CLOUDS_LOW_MAX_ACCUM

struct CloudSSHistoryData {
    vec3 inScattering;
    vec3 transmittance;
    float hLen;
};

CloudSSHistoryData clouds_ss_historyData_init() {
    CloudSSHistoryData data;
    data.inScattering = vec3(0.0);
    data.transmittance = vec3(1.0);
    data.hLen = 0.0;
    return data;
}

void clouds_ss_historyData_pack(out uvec4 packedData, CloudSSHistoryData data) {
    data.inScattering = clamp(data.inScattering, 0.0, FP16_MAX);
    packedData.x = packHalf2x16(data.inScattering.xy);
    packedData.y = packHalf2x16(vec2(data.inScattering.z, data.transmittance.x));
    packedData.z = packHalf2x16(data.transmittance.yz);
    packedData.w = floatBitsToUint(data.hLen);
}

void clouds_ss_historyData_unpack(uvec4 packedData, out CloudSSHistoryData data) {
    if (packedData == uvec4(0u)) {
        data = clouds_ss_historyData_init();
    }
    vec2 temp1 = unpackHalf2x16(packedData.x);
    vec2 temp2 = unpackHalf2x16(packedData.y);
    vec2 temp3 = unpackHalf2x16(packedData.z);
    data.inScattering = vec3(temp1.xy, temp2.x);
    data.transmittance = vec3(temp2.y, temp3.xy);
    data.hLen = uintBitsToFloat(packedData.w);
}

#endif