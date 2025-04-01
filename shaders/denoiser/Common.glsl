#include "/_Base.glsl"
#include "/util/Colors.glsl"

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

void svgf_packNoColor(out uvec4 packedData, vec3 fastColor, vec2 moments, float hLen) {
    packedData.x = 0u;
    packedData.y = packUnorm4x8(colors_SRGBToLogLuv(fastColor));
    packedData.z = packHalf2x16(moments);
    packedData.w = floatBitsToUint(hLen);
}

void svgf_pack(out uvec4 packedData, vec3 color, vec3 fastColor, vec2 moments, float hLen) {
    packedData.x = packUnorm4x8(colors_SRGBToLogLuv(color));
    packedData.y = packUnorm4x8(colors_SRGBToLogLuv(fastColor));
    packedData.z = packHalf2x16(moments);
    packedData.w = floatBitsToUint(hLen);
}

void svgf_unpack(uvec4 packedData, out vec3 color, out vec3 fastColor, out vec2 moments, out float hLen) {
    color = colors_LogLuvToSRGB(unpackUnorm4x8(packedData.x));
    fastColor = colors_LogLuvToSRGB(unpackUnorm4x8(packedData.y));
    moments = unpackHalf2x16(packedData.z);
    hLen = uintBitsToFloat(packedData.w);
}