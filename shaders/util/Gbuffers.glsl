#ifndef INCLUDE_util_GBuffers.glsl
#define INCLUDE_util_GBuffers.glsl
#include "../_Base.glsl"
#include "BitPacking.glsl"

// gbuffer:
// r:
// albedo: 8 x 3 = 24 bits
// roughness: 8 bits
// g:
// f0: 8 bits
// emissive: 8 bits
// porosity/sss: 8 x 1 = 8 bits
// ???: 8 bits
// b:
// normal: 11 + 11 + 10 = 32 bits
// a:
// lmCoord: 8 x 2 = 16 bits
// materialID: 16 bits

// Extra:
// viewZ: 32 bits

struct GBufferData {
    vec3 albedo;
    float roughness;
    float f0;
    float emissive;
    float porositySSS;
    vec3 normal;
    vec2 lmCoord;
    uint materialID;
};

void gbuffer_pack(out uvec4 packedData, GBufferData gData) {
    packedData.r = packUnorm4x8(vec4(gData.albedo, gData.roughness));
    packedData.g = packUnorm4x8(vec4(gData.f0, gData.emissive, gData.porositySSS, 0.0));

    packedData.b = packS11(gData.normal.x);
    packedData.b |= packS11(gData.normal.y) << 11;
    packedData.b |= packS10(gData.normal.z) << 22;

    packedData.a = packUnorm4x8(vec4(gData.lmCoord, 0.0, 0.0)) & 0x0000FFFFu;
    packedData.a |= (gData.materialID & 0xFFFFu) << 16;
}

void gbuffer_unpack(uvec4 packedData, out GBufferData gData) {
    vec4 tempR = unpackUnorm4x8(packedData.r);
    const float a0 = 0.000570846;
    const float a1 = -0.0403863;
    const float a2 = 0.862127;
    const float a3 = 0.178572;
    vec3 x = max(tempR.rgb, 0.0232545);
    vec3 x2 = x * x;
    vec3 x3 = x2 * x;
    gData.albedo = a0 + a1 * x + a2 * x2 + a3 * x3;
    gData.roughness = tempR.a;

    vec4 tempG = unpackUnorm4x8(packedData.g);
    gData.f0 = tempG.r;
    gData.emissive = tempG.g;
    gData.porositySSS = tempG.b;

    gData.normal.x = unpackS11(packedData.b & 0x7FFu);
    gData.normal.y = unpackS11((packedData.b >> 11) & 0x7FFu);
    gData.normal.z = unpackS10((packedData.b >> 22) & 0x3FFu);

    gData.lmCoord = unpackUnorm4x8(packedData.a).xy;
    gData.materialID = (packedData.a >> 16) & 0xFFFFu;
}

#endif