#include "Packing.glsl"

struct PackedGBufferData {
//    vec2 viewCoord;
    vec3 viewCoord;
    uvec4 specularParams;
    vec4 albedo;
    vec3 worldNormal;
    vec2 lightMapCoord;
    uint materialID;
};

void pgbuffer_restoreNormalZ(inout vec3 normal, uint signBitIn) {
    normal.z = sqrt(abs(1.0 - dot(normal.xy, normal.xy)));
    int signBit = int(signBitIn);
    signBit = (-signBit) * 2 + 1;
    normal.z = float(signBit) * normal.z;
}

uint pgbuffer_extractSignBit(float value) {
    return uint(int(sign(value))) & 0x80000000u;
}

void pgbuffer_pack(out uvec4 packed1, out uvec4 packed2, in PackedGBufferData data) {
    packed1.x = packHalf2x16(data.viewCoord.xy);

    packed1.y = data.specularParams.x & 0xFFu |
    (data.specularParams.y & 0xFFu) << 8u |
    (data.specularParams.z & 0xFFu) << 16u |
    (data.specularParams.w & 0xFFu) << 24u;

    packed1.z = packUnorm2x16(data.lightMapCoord);

    packed1.w = packing_packU12(data.albedo.r) |
    (packing_packU12(data.albedo.g) << 12u) |
    (packing_packU8(data.albedo.a) << 24u);

    packed2.x = packing_packS10(data.worldNormal.x) |
    (packing_packS10(data.worldNormal.y) << 10u) |
    (pgbuffer_extractSignBit(data.worldNormal.z) << 20u) |
    (packing_packU11(data.albedo.b) << 21u);

//    packed2.y = data.materialID;
    packed1.y = floatBitsToUint(data.viewCoord.z);
}

void pgbuffer_unpack(uvec4 packed1, uvec4 packed2, out PackedGBufferData data) {
    data.viewCoord.xy = unpackHalf2x16(packed1.x);

    data.specularParams.x = packed1.y & 0xFFu;
    data.specularParams.y = (packed1.y >> 8u) & 0xFFu;
    data.specularParams.z = (packed1.y >> 16u) & 0xFFu;
    data.specularParams.w = (packed1.y >> 24u) & 0xFFu;

    data.lightMapCoord = unpackUnorm2x16(packed1.z);

    data.albedo.r = packing_unpackU12(packed1.w & 4095u);
    data.albedo.g = packing_unpackU12((packed1.w >> 12u) & 4095u);
    data.albedo.a = packing_unpackU8((packed1.w >> 24u) & 255u);

    data.worldNormal.x = packing_unpackS10(packed2.x & 1023u);
    data.worldNormal.y = packing_unpackS10((packed2.x >> 10u) & 1023u);
    pgbuffer_restoreNormalZ(data.worldNormal, (packed2.x >> 20u) & 1u);
    data.albedo.b = packing_unpackU11((packed2.x >> 21u) & 2047u);

//    data.materialID = packed2.y;
    data.viewCoord.z = uintBitsToFloat(packed1.y);
}