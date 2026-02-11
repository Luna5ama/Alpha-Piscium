#ifndef INCLUDE_util_HardcodedPBR_glsl
#define INCLUDE_util_HardcodedPBR_glsl a

#include "BitPacking.glsl"

struct HardcodedPBR {
    float sss;
    float emissive;
    float ior;
    float roughness;
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
    pbr.roughness = unpackU8(bitfieldExtract(rawData.x, 16, 8));
    pbr.isSmallFoliage = bitfieldExtract(rawData.x, 31, 1) == 1u;
    return pbr;
}

#endif