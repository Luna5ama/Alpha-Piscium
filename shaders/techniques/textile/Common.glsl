#ifndef INCLUDE_textile_Common_glsl
#define INCLUDE_textile_Common_glsl a

#include "/util/Math.glsl"

ivec2 textile_texelToTexel(ivec2 texelPos, ivec2 tileOffset, ivec2 tileSize) {
    return clamp(texelPos, ivec2(0), tileSize - 1) + tileOffset;
}

vec2 textile_texelToUV(ivec2 texelPos, ivec2 tileOffset, ivec2 tileSize, vec2 textureSizeRcp) {
    ivec2 textureTexelPos = clamp(texelPos, ivec2(0), tileSize - 1) + tileOffset;
    return saturate((vec2(textureTexelPos) + 0.5) * textureSizeRcp);
}

vec2  textile_texelToGatherUV(vec2 texelPos, vec2 tileOffsetF, vec2 tileSizeF, vec2 textureSizeRcp) {
    vec2 textureTexelPos = clamp(texelPos, vec2(1.0), tileSizeF - 1.0) + tileOffsetF;
    return saturate(textureTexelPos * textureSizeRcp);
}

ivec2 textile_uvToTexel(vec2 uv, ivec2 tileOffset, ivec2 tileSizeI, vec2 tileSizeF) {
    return textile_texelToTexel(ivec2(uv * tileSizeF), tileOffset, tileSizeI);
}

vec2 textile_uvToUV(vec2 uv, vec2 tileOffsetF, vec2 tileSizeF, vec2 textureSizeRcp) {
    vec2 textureTexelPos = clamp(uv * tileSizeF, vec2(0.0), tileSizeF - 1.0) + tileOffsetF;
    return saturate(textureTexelPos * textureSizeRcp);
}

#endif