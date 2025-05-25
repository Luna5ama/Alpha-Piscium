#ifndef INCLUDE_util_NZPacking_glsl
#define INCLUDE_util_NZPacking_glsl a

#include "/util/Coords.glsl"

vec2 nzpacking_fullResGatherUV(vec2 gatherTexelPos) {
    vec2 clampedGatherTexelPos = clamp(gatherTexelPos, vec2(1.0), global_mainImageSize - 1);
    clampedGatherTexelPos.y += global_mainImageSize.y;
    vec2 sizeRcp = global_mainImageSizeRcp;
    sizeRcp.y *= 0.5;
    return clampedGatherTexelPos * sizeRcp;
}

uint nzpacking_packNormalOct32(vec3 normal) {
    return packSnorm2x16(coords_octEncode11(normal));
}

vec3 nzpacking_unpackNormalOct32(uint packedNormal) {
    return coords_octDecode11(unpackSnorm2x16(packedNormal));
}

void nzpacking_pack(out uvec2 packedData, vec3 normal, float depth) {
    packedData.x = nzpacking_packNormalOct32(normal);
    packedData.y = floatBitsToUint(depth);
}

void nzpacking_unpack(uvec2 packedData, out vec3 normal, out float depth) {
    normal = nzpacking_unpackNormalOct32(packedData.x);
    depth = uintBitsToFloat(packedData.y);
}

#endif