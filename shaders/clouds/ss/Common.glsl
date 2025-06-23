#ifndef INCLUDE_clouds_ss_Common_glsl
#define INCLUDE_clouds_ss_Common_glsl a

#include "/util/Colors.glsl"
#include "/textile/CSRGBA32UI.glsl"

#define CLOUDS_SS_MAX_ACCUM 64

struct CloudSSHistoryData {
    vec3 inScattering;
    vec3 transmittance;
    float viewZ;
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