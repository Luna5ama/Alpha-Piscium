#ifndef INCLUDE_techniques_gi_Common_glsl
#define INCLUDE_techniques_gi_Common_glsl a

#include "/util/BitPacking.glsl"
#include "/util/Colors.glsl"
#include "/util/Colors2.glsl"

/*
    Diffuse Color : 48 bits         (1) (1)
    Diffuse fast color: 48 bits     (1) (2)
    Diffuse Moment2: 16 bits        (1) (1)

    Specular Color: 48 bits         (2) (3)
    Specular fast color: 48 bits    (2) (4)
    Specular Moment2: 16 bits       (2) (3)

    History Length: 8 bits          (1) (5)
    Fast history Length: 8 bits     (1) (5)

    Shadow: 16 bits                 (2) (2)
*/

struct GIHistoryData {
    vec3 diffuseColor;
    vec3 diffuseFastColor;
    float diffuseMoments;

    vec3 specularColor;
    vec3 specularFastColor;
    float specularMoments;

    float historyLength;
    float fastHistoryLength;

    float shadow;
};

GIHistoryData gi_historyData_init()  {
    GIHistoryData data;
    data.diffuseColor = vec3(0.0);
    data.diffuseFastColor = vec3(0.0);
    data.diffuseMoments = 0.0;

    data.specularColor = vec3(0.0);
    data.specularFastColor = vec3(0.0);
    data.specularMoments = 0.0;

    data.historyLength = 0.0;
    data.fastHistoryLength = 0.0;

    data.shadow = 0.0;
    return data;
}

void gi_historyData_unpack1(inout GIHistoryData data, uvec4 packedData) {
    vec2 temp1 = unpackHalf2x16(packedData.x);
    vec2 temp2 = unpackHalf2x16(packedData.y);
    vec2 temp3 = unpackHalf2x16(packedData.z);
    vec2 temp4 = unpackHalf2x16(packedData.w);
    vec2 temp5 = unpackUnorm4x8(packedData.w).zw;

    data.diffuseColor.rg = temp1;
    data.diffuseColor.b = temp2.x;
    data.diffuseFastColor.r = temp2.y;
    data.diffuseFastColor.gb = temp3.xy;
    data.diffuseMoments = temp4.x;

    data.historyLength = temp5.x;
    data.fastHistoryLength = temp5.y;
}

void gi_historyData_pack1(GIHistoryData data, out uvec4 packedData) {
    packedData.x = packHalf2x16(data.diffuseColor.rg);
    packedData.y = packHalf2x16(vec2(data.diffuseColor.b, data.diffuseFastColor.r));
    packedData.z = packHalf2x16(data.diffuseFastColor.gb);

    uint temp = packHalf2x16(vec2(data.diffuseMoments, 0.0)) & 0xFFFFu;
    temp |= packUnorm4x8(vec4(0.0, 0.0, vec2(data.historyLength, data.fastHistoryLength))) & 0xFFFF0000u;
    packedData.w = temp;
}

void gi_historyData_unpack2(inout GIHistoryData data, uvec4 packedData) {
    vec2 temp1 = unpackHalf2x16(packedData.x);
    vec2 temp2 = unpackHalf2x16(packedData.y);
    vec2 temp3 = unpackHalf2x16(packedData.z);
    vec2 temp4 = unpackHalf2x16(packedData.w);

    data.specularColor.rg = temp1;
    data.specularColor.b = temp2.x;
    data.specularFastColor.r = temp2.y;
    data.specularFastColor.gb = temp3.xy;
    data.specularMoments = temp4.x;

    data.shadow = temp4.y;
}

void gi_historyData_pack2(GIHistoryData data, out uvec4 packedData) {
    packedData.x = packHalf2x16(data.specularColor.rg);
    packedData.y = packHalf2x16(vec2(data.specularColor.b, data.specularFastColor.r));
    packedData.z = packHalf2x16(data.specularFastColor.gb);
    packedData.w = packHalf2x16(vec2(data.specularMoments, data.shadow));
}

#endif