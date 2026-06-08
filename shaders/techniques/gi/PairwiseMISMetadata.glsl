#ifndef INCLUDE_techniques_gi_PairwiseMISMetadata_glsl
#define INCLUDE_techniques_gi_PairwiseMISMetadata_glsl a

#include "/util/BitPacking.glsl"

struct PairwiseMISMetadata {
    ivec2 selectedTexelDelta;
    float accumM;
    uint numValidNeighbors;
    float mc;
    float spatialWSum;
};

PairwiseMISMetadata pairwiseMISMetadata_init() {
    PairwiseMISMetadata metadata;
    metadata.selectedTexelDelta = ivec2(0);
    metadata.accumM = 0.0;
    metadata.numValidNeighbors = 0u;
    metadata.mc = 1.0;
    metadata.spatialWSum = 0.0;
    return metadata;
}

uint pairwiseMISMetadata_packSelectedTexelDeltaAndNumValidNeighbors(ivec2 selectedTexelDelta, uint numValidNeighbors) {
    uint result = uint(selectedTexelDelta.x) & 0x3FFu;
    result = bitfieldInsert(result, uint(selectedTexelDelta.y) & 0x3FFu, 10, 10);
    // Max reuse candidates count is 4 x 7 = 28 so no need clamping
    result = bitfieldInsert(result, numValidNeighbors, 20, 12);
    return result;
}

ivec2 pairwiseMISMetadata_unpackSelectedTexelDelta(uint packedData) {
    ivec2 result;
    result.x = (int(bitfieldExtract(packedData, 0, 10)) << 22) >> 22;
    result.y = (int(bitfieldExtract(packedData, 10, 10)) << 22) >> 22;
    return result;
}

uint pairwiseMISMetadata_unpackNumValidNeighbors(uint packedData) {
    return bitfieldExtract(packedData, 20, 12);
}

PairwiseMISMetadata pairwiseMISMetadata_unpack(uvec4 packedData) {
    PairwiseMISMetadata metadata;
    metadata.selectedTexelDelta = pairwiseMISMetadata_unpackSelectedTexelDelta(packedData.x);
    metadata.accumM = uintBitsToFloat(packedData.y);
    metadata.numValidNeighbors = pairwiseMISMetadata_unpackNumValidNeighbors(packedData.x);
    metadata.mc = uintBitsToFloat(packedData.z);
    metadata.spatialWSum = uintBitsToFloat(packedData.w);
    return metadata;
}

uvec4 pairwiseMISMetadata_pack(PairwiseMISMetadata metadata) {
    uvec4 packedData;
    packedData.x = pairwiseMISMetadata_packSelectedTexelDeltaAndNumValidNeighbors(metadata.selectedTexelDelta, metadata.numValidNeighbors);
    packedData.y = floatBitsToUint(metadata.accumM);
    packedData.z = floatBitsToUint(max(metadata.mc, 0.0));
    packedData.w = floatBitsToUint(metadata.spatialWSum);
    return packedData;
}

#endif
