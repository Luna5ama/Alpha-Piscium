#ifndef INCLUDE_util_GBuffers.glsl
#define INCLUDE_util_GBuffers.glsl
#include "../_Base.glsl"
#include "BitPacking.glsl"

const uint MATERIAL_ID_UNDEFINED = 65535u;

// gbuffer:
// r:
// albedo: 8 x 3 = 24 bits
// materialAO: 8 bits
// g:
// pbrS: 32 bits
// b:
// normal: 11 + 11 + 10 = 32 bits
// a:
// lmCoord: 8 x 2 = 16 bits
// materialID: 16 bits

// Extra:
// viewZ: 32 bits

struct GBufferData {
    vec3 albedo;
    float materialAO;
    vec4 pbrSpecular;
    vec3 normal;
    vec2 lmCoord;
    uint materialID;
    bool isHand;
};

void gbuffer_pack(out uvec4 packedData, GBufferData gData) {
    packedData.r = packUnorm4x8(vec4(gData.albedo, gData.materialAO));
    packedData.g = packUnorm4x8(vec4(gData.pbrSpecular));

    packedData.b = packS10(gData.normal.x);
    packedData.b |= packS10(gData.normal.y) << 10;
    packedData.b |= packS10(gData.normal.z) << 20;
    packedData.b |= uint(gData.isHand) << 30;

    packedData.a = packUnorm4x8(vec4(gData.lmCoord, 0.0, 0.0)) & 0x0000FFFFu;
    packedData.a |= (gData.materialID & 0xFFFFu) << 16;
}

void gbuffer_unpack(uvec4 packedData, out GBufferData gData) {
    vec4 tempR = unpackUnorm4x8(packedData.r);
    gData.albedo = tempR.rgb;
    gData.materialAO = tempR.a;

    gData.pbrSpecular = unpackUnorm4x8(packedData.g);

    gData.normal.x = unpackS10(packedData.b & 0x3FFu);
    gData.normal.y = unpackS10((packedData.b >> 10) & 0x3FFu);
    gData.normal.z = unpackS10((packedData.b >> 20) & 0x3FFu);
    gData.isHand = bool((packedData.b >> 30) & 1u);

    gData.lmCoord = unpackUnorm4x8(packedData.a).xy;
    gData.materialID = (packedData.a >> 16) & 0xFFFFu;
}

#endif