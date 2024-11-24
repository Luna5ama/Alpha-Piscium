#ifndef INCLUDE_rtwsm_RTWSM.glsl
#define INCLUDE_rtwsm_RTWSM.glsl

#include "../_Util.glsl"

#if SETTING_RTWSM_IMAP_SIZE == 128

#define IMAP_SIZE_D2 64
#define IMAP_SIZE_D16 8
#define IMAP_SIZE_D128 1

#elif SETTING_RTWSM_IMAP_SIZE == 256

#define IMAP_SIZE_D2 128
#define IMAP_SIZE_D16 16
#define IMAP_SIZE_D128 2

#elif SETTING_RTWSM_IMAP_SIZE == 512

#define IMAP_SIZE_D2 256
#define IMAP_SIZE_D16 32
#define IMAP_SIZE_D128 4

#elif SETTING_RTWSM_IMAP_SIZE == 1024

#define IMAP_SIZE_D2 512
#define IMAP_SIZE_D16 64
#define IMAP_SIZE_D128 8

#endif

vec2 rtwsm_warpTexCoord(sampler2D warpingMap, vec2 uv) {
    vec2 result = uv;
    vec4 warp = vec4(texture(warpingMap, vec2(result.x, 0.25)).rg, texture(warpingMap, vec2(result.y, 0.75)).rg);
    result += warp.xz;
    return result;
}

vec2 rtwsm_warpTexCoordTexelSize(sampler2D warpingMap, vec2 uv, out vec2 texelSize) {
    vec2 result = uv;
    vec4 warp = vec4(texture(warpingMap, vec2(result.x, 0.25)).rg, texture(warpingMap, vec2(result.y, 0.75)).rg);
    result += warp.xz;
    texelSize = warp.yw;
    return result;
}

float rtwsm_sampleShadowDepth(sampler2DShadow shadowMap, vec3 coord, float lod) {
    vec2 ndcCoord = coord.xy * 2.0 - 1.0;
    float edgeCoord = max(abs(ndcCoord.x), abs(ndcCoord.y));
    return mix(textureLod(shadowMap, coord, lod), 1.0, linearStep(1.0 - SHADOW_MAP_SIZE.y * 16, 1.0, edgeCoord));
}

vec4 rtwsm_sampleShadowColor(sampler2D shadowColor, vec2 coord, float lod) {
    return textureLod(shadowColor, coord, lod);
}

float rtwsm_sampleShadowDepth(sampler2D shadowMap, vec3 coord, float lod) {
    uint flag = uint(any(lessThan(coord.xy, vec2(0.0))));
    flag |= uint(any(greaterThan(coord.xy, vec2(1.0))));
    return mix(textureLod(shadowMap, coord.xy, lod).r, coord.z, float(flag));
}

float rtwsm_linearDepth(float d) {
    return (d * 2.0 - 1.0) * 4.0 * shadowProjectionInverse[2][2] + shadowProjectionInverse[3][2];
}
#endif