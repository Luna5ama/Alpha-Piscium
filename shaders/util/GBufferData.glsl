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
    vec3 geomNormal;
    vec3 geomTangent;
    vec4 pbrSpecular;
    vec3 normal;
    vec2 lmCoord;
    uint materialID;

    vec3 albedo; // Still in its input space (Typically gamma encoded sRGB)
    bool isHand;
    int bitangentSign;
};

GBufferData gbufferData_init() {
    GBufferData gData;
    gData.geomNormal = vec3(0.0);
    gData.geomTangent = vec3(0.0);
    gData.pbrSpecular = vec4(0.0);
    gData.normal = vec3(0.0);
    gData.lmCoord = vec2(0.0);
    gData.materialID = MATERIAL_ID_UNDEFINED;

    gData.albedo = vec3(0.0);
    gData.isHand = false;

    return gData;
}

void gbufferData1_pack(out uvec4 packedData, GBufferData gData) {
    nzpacking_packNormalOct16(packedData.r, coords_dir_viewToWorld(gData.geomNormal), coords_dir_viewToWorld(gData.geomTangent));
    packedData.g = packUnorm4x8(vec4(gData.pbrSpecular));
    packedData.b = nzpacking_packNormalOct32(coords_dir_viewToWorld(gData.normal));
    packedData.a = packUnorm4x8(vec4(gData.lmCoord, 0.0, 0.0)) & 0x0000FFFFu;
    packedData.a |= (gData.materialID & 0xFFFFu) << 16;
}

void gbufferData1_unpack_world(uvec4 packedData, inout GBufferData gData) {
    nzpacking_unpackNormalOct16(packedData.r, gData.geomNormal, gData.geomTangent);
    gData.geomNormal = gData.geomNormal;
    gData.geomTangent = gData.geomTangent;
    gData.pbrSpecular = unpackUnorm4x8(packedData.g);
    gData.normal = nzpacking_unpackNormalOct32(packedData.b);
    gData.lmCoord = unpackUnorm4x8(packedData.a).xy;
    gData.materialID = (packedData.a >> 16) & 0xFFFFu;
}

void gbufferData1_unpack(uvec4 packedData, inout GBufferData gData) {
    gbufferData1_unpack_world(packedData, gData);
    gData.geomNormal = coords_dir_worldToView(gData.geomNormal);
    gData.geomTangent = coords_dir_worldToView(gData.geomTangent);
    gData.normal = coords_dir_worldToView(gData.normal);
}

void gbufferData2_pack(out uvec4 packedData, GBufferData gData) {
    packedData.r = packUnorm4x8(vec4(gData.albedo, 0.0)) & 0x00FFFFFFu;
    packedData.r = bitfieldInsert(packedData.r, uint(gData.isHand), 24, 1);
    packedData.r = bitfieldInsert(packedData.r, uint(clamp(gData.bitangentSign, 0, 1)), 25, 1);
}

void gbufferData2_unpack(uvec4 packedData, inout GBufferData gData) {
    gData.albedo = unpackUnorm4x8(packedData.r).rgb;
    gData.isHand = bool(bitfieldExtract(packedData.r, 24, 1));
    gData.bitangentSign = int(bitfieldExtract(packedData.r, 25, 1)) * 2 - 1;
}

#endif