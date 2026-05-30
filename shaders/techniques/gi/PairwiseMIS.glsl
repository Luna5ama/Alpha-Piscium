#ifndef INCLUDE_techniques_gi_PairwiseMIS_glsl
#define INCLUDE_techniques_gi_PairwiseMIS_glsl a

#include "/util/BitPacking.glsl"

struct PairwiseMISMetadata {
    ivec2 selectedTexel;
    float accumM;
    uint numValidNeighbors;
    float mc;
    float spatialWSum;
};

PairwiseMISMetadata pairwiseMISMetadata_init(ivec2 texel) {
    PairwiseMISMetadata metadata;
    metadata.selectedTexel = texel;
    metadata.accumM = 0.0;
    metadata.numValidNeighbors = 0u;
    metadata.mc = 1.0;
    metadata.spatialWSum = 0.0;
    return metadata;
}

uint pairwiseMISMetadata_packMcAndNumValidNeighbors(float mc, uint numValidNeighbors) {
    uint mcBits = floatBitsToUint(max(mc, 0.0));
    uint packedMc = (mcBits << 1) & 0xFFFFFFF0u;
    return packedMc | (min(numValidNeighbors, 15u) & 0xFu);
}

float pairwiseMISMetadata_unpackMc(uint packedData) {
    return uintBitsToFloat((packedData & 0xFFFFFFF0u) >> 1);
}

uint pairwiseMISMetadata_unpackNumValidNeighbors(uint packedData) {
    return packedData & 0xFu;
}

PairwiseMISMetadata pairwiseMISMetadata_unpack(uvec4 packedData) {
    PairwiseMISMetadata metadata;
    metadata.selectedTexel = ivec2(unpackUInt2x16(packedData.x));
    metadata.accumM = uintBitsToFloat(packedData.y);
    metadata.numValidNeighbors = pairwiseMISMetadata_unpackNumValidNeighbors(packedData.z);
    metadata.mc = pairwiseMISMetadata_unpackMc(packedData.z);
    metadata.spatialWSum = uintBitsToFloat(packedData.w);
    return metadata;
}

uvec4 pairwiseMISMetadata_pack(PairwiseMISMetadata metadata) {
    uvec4 packedData;
    packedData.x = packUInt2x16(uvec2(metadata.selectedTexel));
    packedData.y = floatBitsToUint(metadata.accumM);
    packedData.z = pairwiseMISMetadata_packMcAndNumValidNeighbors(metadata.mc, metadata.numValidNeighbors);
    packedData.w = floatBitsToUint(metadata.spatialWSum);
    return packedData;
}

#endif
