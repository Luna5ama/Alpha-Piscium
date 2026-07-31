#ifndef INCLUDE_util_HardcodedPBR_glsl
#define INCLUDE_util_HardcodedPBR_glsl a

#include "BitPacking.glsl"

struct HardcodedPBR {
    float sss;
    float emissive;
    float ior;
    float roughness;
    float dielectric;
    uint metalIndex;
    int emissiveMultiplier;
    bool isFullCube;
    bool isSmallFoliage;
    uint blockModelMetadata;
    bool isKnown;
};

HardcodedPBR hardcodedpbr_decode(uint materialID) {
    bool isKnown = materialID != 0u &&
        materialID < textureSize(usam_pbrLUT0, 0).x &&
        materialID < textureSize(usam_pbrLUT1, 0).x &&
        materialID < textureSize(usam_pbrLUT2, 0).x;
    if (!isKnown) {
        materialID = 0u;
    }
    uvec4 materialData = uvec4(texelFetch(usam_pbrLUT0, int(materialID), 0));
    uint flagData = texelFetch(usam_pbrLUT1, int(materialID), 0).r;
    HardcodedPBR pbr;
    pbr.sss = unpackU4(bitfieldExtract(materialData.x, 0, 4));
    pbr.emissive = unpackU4(bitfieldExtract(materialData.x, 4, 4));
    pbr.ior = unpackU8(bitfieldExtract(materialData.x, 8, 8)) * 3.0;
    pbr.roughness = unpackU8(bitfieldExtract(materialData.x, 16, 8));
    pbr.metalIndex = bitfieldExtract(materialData.x, 24, 4);
    pbr.dielectric = unpackU4(bitfieldExtract(materialData.x, 28, 4));
    pbr.isKnown = isKnown;
    int temp = int(bitfieldExtract(flagData, 0, 4));
    pbr.emissiveMultiplier = temp | (0 - (temp & 0x8));
    pbr.isFullCube = bitfieldExtract(flagData, 4, 1) == 1u;
    pbr.isSmallFoliage = bitfieldExtract(flagData, 5, 1) == 1u;
    pbr.blockModelMetadata = texelFetch(usam_pbrLUT2, int(materialID), 0).x;
    return pbr;
}

#endif
