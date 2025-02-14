#ifndef INCLUDE_util_GBuffers_glsl
#define INCLUDE_util_GBuffers_glsl a
#include "BitPacking.glsl"
#include "Coords.glsl"

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
    packedData.r = packUnorm4x8(vec4(gData.albedo, gData.materialAO * 0.5)) & 0x7FFFFFFFu;
    packedData.r |= uint(gData.isHand) << 31;
    packedData.g = packUnorm4x8(vec4(gData.pbrSpecular));

    packedData.b = packSnorm2x16(coords_octEncode11(gData.normal));

    packedData.a = packUnorm4x8(vec4(gData.lmCoord, 0.0, 0.0)) & 0x0000FFFFu;
    packedData.a |= (gData.materialID & 0xFFFFu) << 16;
}

void gbuffer_unpack(uvec4 packedData, out GBufferData gData) {
    vec4 tempR = unpackUnorm4x8(packedData.r & 0x7FFFFFFFu);
    gData.albedo = tempR.rgb;
    gData.materialAO = tempR.a;
    gData.isHand = bool(packedData.r >> 31u);

    gData.pbrSpecular = unpackUnorm4x8(packedData.g);

    gData.normal = coords_octDecode11(unpackSnorm2x16(packedData.b));

    gData.lmCoord = unpackUnorm4x8(packedData.a).xy;
    gData.materialID = (packedData.a >> 16) & 0xFFFFu;
}

#endif