#include "/_Base.glsl"

vec2 svgf_gatherUV1(vec2 gatherTexelPos) {
    vec2 clampedGatherTexelPos = clamp(gatherTexelPos, vec2(1.0), global_mainImageSize - 1);
    vec2 sizeRcp = global_mainImageSizeRcp;
    sizeRcp *= 0.5;
    return clampedGatherTexelPos * sizeRcp;
}

vec2 svgf_gatherUV2(vec2 gatherTexelPos) {
    vec2 clampedGatherTexelPos = clamp(gatherTexelPos, vec2(1.0), global_mainImageSize - 1);
    clampedGatherTexelPos.x += global_mainImageSize.x;
    vec2 sizeRcp = global_mainImageSizeRcp;
    sizeRcp *= 0.5;
    return clampedGatherTexelPos * sizeRcp;
}

vec2 svgf_gatherUV3(vec2 gatherTexelPos) {
    vec2 clampedGatherTexelPos = clamp(gatherTexelPos, vec2(1.0), global_mainImageSize - 1);
    clampedGatherTexelPos += global_mainImageSize;
    vec2 sizeRcp = global_mainImageSizeRcp;
    sizeRcp *= 0.5;
    return clampedGatherTexelPos * sizeRcp;
}

vec2 svgf_gatherUV4(vec2 gatherTexelPos) {
    vec2 clampedGatherTexelPos = clamp(gatherTexelPos, vec2(1.0), global_mainImageSize - 1);
    clampedGatherTexelPos.y += global_mainImageSize.y;
    vec2 sizeRcp = global_mainImageSizeRcp;
    sizeRcp *= 0.5;
    return clampedGatherTexelPos * sizeRcp;
}

void svgf_pack(out uvec4 packedData, vec4 colorHLen, vec2 moments) {
    packedData.x = packHalf2x16(colorHLen.rg);
    packedData.y = packHalf2x16(colorHLen.ba);
    packedData.z = packHalf2x16(moments);
    packedData.w = 0u;
}

void svgf_unpack(uvec4 packedData, out vec4 colorHLen, out vec2 moments) {
    vec2 temp1 = unpackHalf2x16(packedData.x);
    vec2 temp2 = unpackHalf2x16(packedData.y);
    vec2 temp3 = unpackHalf2x16(packedData.z);
//    vec2 temp4 = unpackHalf2x16(packedData.w); // unused

    colorHLen.rg = temp1;
    colorHLen.ba = temp2;
    moments = temp3;
}