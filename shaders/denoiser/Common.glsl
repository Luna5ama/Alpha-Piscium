#include "/_Base.glsl"

vec2 svgf_gatherUV1(vec2 gatherTexelPos) {
    vec2 clampedGatherTexelPos = clamp(gatherTexelPos, vec2(1.0), global_mainImageSize - 1);
    vec2 sizeRcp = global_mainImageSizeRcp;
    sizeRcp.y *= 0.5;
    return clampedGatherTexelPos * sizeRcp;
}

vec2 svgf_gatherUV2(vec2 gatherTexelPos) {
    vec2 clampedGatherTexelPos = clamp(gatherTexelPos, vec2(1.0), global_mainImageSize - 1);
    clampedGatherTexelPos.y += global_mainImageSize.y;
    vec2 sizeRcp = global_mainImageSizeRcp;
    sizeRcp.y *= 0.5;
    return clampedGatherTexelPos * sizeRcp;
}

ivec2 svgf_texelPos1(ivec2 texelPos) {
    vec2 clampedTexelPos = clamp(texelPos, ivec2(0), global_mainImageSizeI - 1);
    return ivec2(clampedTexelPos);
}

ivec2 svgf_texelPos2(ivec2 texelPos) {
    vec2 clampedTexelPos = clamp(texelPos, ivec2(0), global_mainImageSizeI - 1);
    clampedTexelPos.y += global_mainImageSizeI.y;
    return ivec2(clampedTexelPos);
}

void svgf_pack(out uvec4 packedData, vec3 color, vec3 fastColor, vec2 moments, float hLen) {
    packedData.x = packHalf2x16(color.rg);
    packedData.y = packHalf2x16(vec2(color.b, 0.0));
    packedData.z = packHalf2x16(moments);
    packedData.w = floatBitsToUint(hLen);
}

void svgf_unpack(uvec4 packedData, out vec3 color, out vec3 fastColor, out vec2 moments, out float hLen) {
    vec2 temp1 = unpackHalf2x16(packedData.x);
    vec2 temp2 = unpackHalf2x16(packedData.y);
    vec2 temp3 = unpackHalf2x16(packedData.z);
    color = vec3(temp1, temp2.x);
    fastColor = vec3(0.0);
    moments = temp3;
    hLen = uintBitsToFloat(packedData.w);
}