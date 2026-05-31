#ifndef INCLUDE_util_HardcodedPBR_glsl
#define INCLUDE_util_HardcodedPBR_glsl a

#include "BitPacking.glsl"

struct HardcodedPBR {
    float sss;
    float emissive;
    float ior;
    float roughness;
    int emissiveMultiplier;
    bool isFullCube;
    bool isSmallFoliage;
};

HardcodedPBR hardcodedpbr_decode(uint materialID) {
    if (materialID >= textureSize(usam_pbrLUT0, 0).x) {
        materialID = 0u;
    }
    uvec4 rawData = uvec4(texelFetch(usam_pbrLUT0, int(materialID), 0));
    HardcodedPBR pbr;
    pbr.sss = unpackU4(bitfieldExtract(rawData.x, 0, 4));
    pbr.emissive = unpackU4(bitfieldExtract(rawData.x, 4, 4));
    pbr.ior = unpackU8(bitfieldExtract(rawData.x, 8, 8)) * 3.0;
    pbr.roughness = unpackU8(bitfieldExtract(rawData.x, 24, 8));
    int temp = int(bitfieldExtract(rawData.x, 24, 4));
    pbr.emissiveMultiplier = temp | (0 - (temp & 0x8));
    pbr.isFullCube = bitfieldExtract(rawData.x, 28, 1) == 1u;
    pbr.isSmallFoliage = bitfieldExtract(rawData.x, 29, 1) == 1u;
    return pbr;
}

#endif