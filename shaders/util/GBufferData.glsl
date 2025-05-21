#ifndef INCLUDE_util_GBuffers_glsl
#define INCLUDE_util_GBuffers_glsl a
#include "BitPacking.glsl"
#include "Coords.glsl"
#include "NZPacking.glsl"

const uint MATERIAL_ID_UNDEFINED = 65535u;

// GBuffer Data 32UI (RGBA32UI):
// r:
// geomertryNromal: 32 bits
// g:
// pbrSpecular: 32 bits
// b:
// normal: 32 bits
// a:
// lmCoord: 8 x 2 = 16 bits
// materialID: 16 bits
//
// GBuffer Data 8UN (RGBA8UN):
// Albedo: 24 bits
// isHand: 1 bit
//
// GBuffer ViewZ (R32F):
// viewZ: 32 bits

struct GBufferData {
    vec3 geometryNormal;
    vec4 pbrSpecular;
    vec3 normal;
    vec2 lmCoord;
    uint materialID;

    vec3 albedo;
    bool isHand;
};

void gbufferData1_pack(out uvec4 packedData, GBufferData gData) {
    packedData.r = nzpacking_packNormal(gData.geometryNormal);
    packedData.g = packUnorm4x8(vec4(gData.pbrSpecular));
    packedData.b = nzpacking_packNormal(gData.normal);
    packedData.a = packUnorm4x8(vec4(gData.lmCoord, 0.0, 0.0)) & 0x0000FFFFu;
    packedData.a |= (gData.materialID & 0xFFFFu) << 16;
}

void gbufferData1_unpack(uvec4 packedData, inout GBufferData gData) {
    gData.geometryNormal = nzpacking_unpackNormal(packedData.r);
    gData.pbrSpecular = unpackUnorm4x8(packedData.g);
    gData.normal = nzpacking_unpackNormal(packedData.b);
    gData.lmCoord = unpackUnorm4x8(packedData.a).xy;
    gData.materialID = (packedData.a >> 16) & 0xFFFFu;
}

void gbufferData2_pack(out vec4 packedData, GBufferData gData) {
    packedData.rgb = gData.albedo;
    packedData.a = float(uint(gData.isHand));
}

void gbufferData2_unpack(vec4 packedData, inout GBufferData gData) {
    gData.albedo = packedData.rgb;
    gData.isHand = bool(uint(packedData.a));
}

#endif