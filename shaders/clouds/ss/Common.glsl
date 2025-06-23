#ifndef INCLUDE_clouds_ss_Common_glsl
#define INCLUDE_clouds_ss_Common_glsl a

#include "/util/Colors.glsl"
#include "/util/Rand.glsl"
#include "/textile/CSRGBA32UI.glsl"

#define UPSCALE_FACTOR 4

#if UPSCALE_FACTOR == 1
ivec2 renderSize = global_mipmapSizesI[0];
#define RENDER_MULTIPLIER 1.0
#define UPSCALE_BLOCK_SIZE 1

#elif UPSCALE_FACTOR == 2
ivec2 renderSize = global_mipmapSizesI[1];
#define UPSCALE_BLOCK_SIZE 4
#define RENDER_MULTIPLIER 0.5

#elif UPSCALE_FACTOR == 4
ivec2 renderSize = global_mipmapSizesI[2];
#define UPSCALE_BLOCK_SIZE 16
#define RENDER_MULTIPLIER 0.25

#elif UPSCALE_FACTOR == 8
ivec2 renderSize = global_mipmapSizesI[3];
#define UPSCALE_BLOCK_SIZE 64
#define RENDER_MULTIPLIER 0.125

#endif

vec2 getTexelPos1x1(ivec2 texelPosDownScale) {
    vec2 texelPos1x1F = vec2(texelPosDownScale * UPSCALE_FACTOR);
    vec2 offset = rand_r2Seq2(frameCounter);
    offset *= UPSCALE_FACTOR;
    offset = mod(offset, vec2(UPSCALE_FACTOR));
    return clamp(texelPos1x1F + offset, vec2(0.5), global_mainImageSize - 0.5);
}

#define CLOUDS_SS_MAX_ACCUM 8

struct CloudSSHistoryData {
    vec3 inScattering;
    vec3 transmittance;
    float viewZ; // in unit km, positive
    float hLen;
};

CloudSSHistoryData clouds_ss_historyData_init() {
    CloudSSHistoryData data;
    data.inScattering = vec3(0.0);
    data.transmittance = vec3(1.0);
    data.viewZ = 0.0;
    data.hLen = 0.0;
    return data;
}

void clouds_ss_historyData_pack(out uvec4 packedData, CloudSSHistoryData data) {
    packedData.x = packUnorm4x8(colors_sRGBToLogLuv32(data.inScattering));
    packedData.y = packUnorm2x16(data.transmittance.xy);
    packedData.z = packUnorm2x16(vec2(data.transmittance.z, saturate(data.hLen / float(CLOUDS_SS_MAX_ACCUM))));
    packedData.w = floatBitsToUint(data.viewZ);
}

void clouds_ss_historyData_unpack(uvec4 packedData, out CloudSSHistoryData data) {
    data.inScattering = colors_LogLuv32ToSRGB(unpackUnorm4x8(packedData.x));
    data.transmittance.xy = unpackUnorm2x16(packedData.y);
    vec2 temp = unpackUnorm2x16(packedData.z);
    data.transmittance.z = temp.x;
    data.hLen = temp.y * float(CLOUDS_SS_MAX_ACCUM);
    data.viewZ = uintBitsToFloat(packedData.w);
}

#endif