#ifndef INCLUDE_techniques_gi_Common_glsl
#define INCLUDE_techniques_gi_Common_glsl a

#include "/util/BitPacking.glsl"
#include "/util/Colors.glsl"
#include "/util/Colors2.glsl"
#include "/util/NZPacking.glsl"

#ifdef SETTING_GI_USE_REFERENCE
#define USE_REFERENCE 1
#else
#define USE_REFERENCE 0
#endif
#define SKIP_FRAMES 0
#define MAX_FRAMES 0x7fffffff
#define RANDOM_FRAME (frameCounter - SKIP_FRAMES)
#define MC_SPP 16

const float HISTORY_LENGTH = float(SETTING_DENOISER_HISTORY_LENGTH);
const float TOTAL_HISTORY_LENGTH = 255.0;
const float FAST_HISTORY_LENGTH = float(SETTING_DENOISER_FAST_HISTORY_LENGTH);

#define GI_MB 1.0
/*
    Diffuse Color : 48 bits         (1)
    Diffuse fast color: 48 bits     (2)
    Diffuse Moment2: 16 bits        (1)

    Specular Color: 48 bits         (3)
    Specular fast color: 48 bits    (4)
    Specular Moment2: 16 bits       (3)

    History Length: 8 bits          (5)
    ~ Edge Mask: 8 bits             (5)

    Shadow: 16 bits                 (2)
*/

const float MAX_HIT_DISTANCE = 64.0;

struct GIHistoryData {
    vec3 diffuseColor;
    vec3 diffuseFastColor;
    float diffuseHitDistance;

    vec3 specularColor;
    vec3 specularFastColor;
    float specularHitDistance;

    float historyLength;
    float realHistoryLength;
    float edgeMask;
    float glazingAngleFactor;

    float shadow;
};

GIHistoryData gi_historyData_init()  {
    GIHistoryData data;
    data.diffuseColor = vec3(0.0);
    data.diffuseFastColor = vec3(0.0);
    data.diffuseHitDistance = MAX_HIT_DISTANCE;

    data.specularColor = vec3(0.0);
    data.specularFastColor = vec3(0.0);
    data.specularHitDistance = MAX_HIT_DISTANCE;

    data.historyLength = 0.0;
    data.realHistoryLength = 0.0;
    data.edgeMask = 0.0;
    data.shadow = 0.0;
    data.glazingAngleFactor = 0.0;
    return data;
}

void gi_historyData_unpack1(inout GIHistoryData data, vec4 packedData) {
    data.diffuseColor = packedData.xyz;
    data.shadow = packedData.w;
}

void gi_historyData_unpack2(inout GIHistoryData data, vec4 packedData) {
    data.diffuseFastColor = packedData.xyz;
    data.diffuseHitDistance = packedData.w;
}

void gi_historyData_unpack3(inout GIHistoryData data, vec4 packedData) {
    data.specularColor = packedData.xyz;
}

void gi_historyData_unpack4(inout GIHistoryData data, vec4 packedData) {
    data.specularFastColor = packedData.xyz;
    data.specularHitDistance = packedData.w;
}

void gi_historyData_unpack5(inout GIHistoryData data, vec4 packedData) {
    data.historyLength = packedData.x;
    data.realHistoryLength = packedData.y;
    data.edgeMask = packedData.z;
    data.glazingAngleFactor = packedData.w;
}

vec4 gi_historyData_pack1(GIHistoryData data) {
    return vec4(data.diffuseColor, data.shadow);
}

vec4 gi_historyData_pack2(GIHistoryData data) {
    return vec4(data.diffuseFastColor, data.diffuseHitDistance);
}

vec4 gi_historyData_pack3(GIHistoryData data) {
    return vec4(data.specularColor, 0.0);
}

vec4 gi_historyData_pack4(GIHistoryData data) {
    return vec4(data.specularFastColor, data.specularHitDistance);
}

vec4 gi_historyData_pack5(GIHistoryData data) {
    return vec4(data.historyLength, data.realHistoryLength, data.edgeMask, data.glazingAngleFactor);
}

float gi_planeDistance(vec3 pos1, vec3 normal1, vec3 pos2, vec3 normal2) {
    vec3 posDiff = pos1 - pos2;
    float planeDist1 = abs(dot(posDiff, normal2));
    float planeDist2 = abs(dot(posDiff, normal1));
    float maxPlaneDist = max(planeDist1, planeDist2);
    return maxPlaneDist;
}

struct ReprojectInfo {
    vec4 bilateralWeights;
    float historyResetFactor;
    vec2 curr2PrevScreenPos;
};

ReprojectInfo reprojectInfo_init() {
    ReprojectInfo info;
    info.bilateralWeights = vec4(0.0);
    info.historyResetFactor = 0.0;
    info.curr2PrevScreenPos = vec2(-1.0);
    return info;
}

ReprojectInfo reprojectInfo_unpack(uvec4 packedData) {
    ReprojectInfo info;
    info.bilateralWeights = unpackUnorm4x16(packedData.xy);
    info.historyResetFactor = uintBitsToFloat(packedData.z);
    info.curr2PrevScreenPos = unpackUnorm2x16(packedData.w);
    return info;
}

uvec4 reprojectInfo_pack(ReprojectInfo info) {
    uvec4 packedData;
    packedData.xy = packUnorm4x16(info.bilateralWeights);
    packedData.z = floatBitsToUint(info.historyResetFactor);
    packedData.w = packUnorm2x16(info.curr2PrevScreenPos);
    return packedData;
}

vec2 _gi_mirrorUV(vec2 uv) {
    return 1.0 - abs(1.0 - (fract(uv * 0.5) * 2.0));
}

#endif