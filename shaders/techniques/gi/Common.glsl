#ifndef INCLUDE_techniques_gi_Common_glsl
#define INCLUDE_techniques_gi_Common_glsl a

#include "/util/BitPacking.glsl"
#include "/util/Colors.glsl"
#include "/util/Colors2.glsl"

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

struct GIHistoryData {
    vec3 diffuseColor;
    vec3 diffuseFastColor;
    float diffuseMoment2;

    vec3 specularColor;
    vec3 specularFastColor;
    float specularMoment2;

    float historyLength;
    float edgeMask;
    float glazingAngleFactor;

    float shadow;
};

GIHistoryData gi_historyData_init()  {
    GIHistoryData data;
    data.diffuseColor = vec3(0.0);
    data.diffuseFastColor = vec3(0.0);
    data.diffuseMoment2 = 0.0;

    data.specularColor = vec3(0.0);
    data.specularFastColor = vec3(0.0);
    data.specularMoment2 = 0.0;

    data.historyLength = 0.0;
    data.edgeMask = 0.0;

    data.shadow = 0.0;
    return data;
}

void gi_historyData_unpack1(inout GIHistoryData data, vec4 packedData) {
    data.diffuseColor = packedData.xyz;
    data.diffuseMoment2 = packedData.w;
}

void gi_historyData_unpack2(inout GIHistoryData data, vec4 packedData) {
    data.diffuseFastColor = packedData.xyz;
    data.shadow = packedData.w;
}

void gi_historyData_unpack3(inout GIHistoryData data, vec4 packedData) {
    data.specularColor = packedData.xyz;
    data.specularMoment2 = packedData.w;
}

void gi_historyData_unpack4(inout GIHistoryData data, vec4 packedData) {
    data.specularFastColor = packedData.xyz;
}

void gi_historyData_unpack5(inout GIHistoryData data, vec4 packedData) {
    data.historyLength = packedData.x;
    data.edgeMask = packedData.y;
    data.glazingAngleFactor = packedData.z;
}

vec4 gi_historyData_pack1(GIHistoryData data) {
    return vec4(data.diffuseColor, data.diffuseMoment2);
}

vec4 gi_historyData_pack2(GIHistoryData data) {
    return vec4(data.diffuseFastColor, data.shadow);
}

vec4 gi_historyData_pack3(GIHistoryData data) {
    return vec4(data.specularColor, data.specularMoment2);
}

vec4 gi_historyData_pack4(GIHistoryData data) {
    return vec4(data.specularFastColor, 0.0);
}

vec4 gi_historyData_pack5(GIHistoryData data) {
    return vec4(data.historyLength, data.edgeMask, data.glazingAngleFactor, 0.0);
}

float gi_planeDistance(vec3 pos1, vec3 normal1, vec3 pos2, vec3 normal2) {
    vec3 posDiff = pos1 - pos2;
    float planeDist1 = abs(dot(posDiff, normal2));
    float planeDist2 = abs(dot(posDiff, normal1));
    float maxPlaneDist = max(planeDist1, planeDist2);
    return maxPlaneDist;
}

#endif