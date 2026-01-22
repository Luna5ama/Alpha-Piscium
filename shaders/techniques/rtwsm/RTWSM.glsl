#ifndef INCLUDE_rtwsm_RTWSM_glsl
#define INCLUDE_rtwsm_RTWSM_glsl a

#include "/util/Math.glsl"

#define RTWSM_IMAP_SIZE 256

#define IMAP_SIZE_D2 128
#define IMAP_SIZE_D16 16
#define IMAP_SIZE_D32 8
#define IMAP_SIZE_D128 2

#define IMAP1D_X_V (0.5 / 2.0)
#define IMAP1D_Y_V (1.5 / 2.0)

#define WARP_X_V (0.5 / 2.0)
#define WARP_Y_V (1.5 / 2.0)

#define TEXELSIZE_X_V (0.5 / 2.0)
#define TEXELSIZE_Y_V (1.5 / 2.0)

vec2 rtwsm_warpTexCoord(vec2 uv) {
    vec2 result = uv;
    vec4 warpTexelSize = persistent_rtwsm_warpTexelSize_sample(uv);
    result.xy += warpTexelSize.xy * 2.0 - 1.0;
    return result;
}

vec2 rtwsm_warpTexCoordTexelSize(vec2 uv, out vec2 texelSize) {
    vec2 result = uv;
    vec4 warpTexelSize = persistent_rtwsm_warpTexelSize_sample(uv);
    result.xy += warpTexelSize.xy * 2.0 - 1.0;
    texelSize = warpTexelSize.zw;
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
    return mix(-global_shadowAABBMaxPrev.z - 512.0, -global_shadowAABBMinPrev.z + 16.0, d);
}

float rtwsm_linearDepthInverse(float depth) {
    return linearStep(-global_shadowAABBMaxPrev.z - 512.0, -global_shadowAABBMinPrev.z + 16.0, depth);
}

float rtwsm_linearDepthOffset(float zOffset) {
    float range = (-global_shadowAABBMinPrev.z + 16.0) - (-global_shadowAABBMaxPrev.z - 512.0);
    return zOffset * range;
}

float rtwsm_linearDepthOffsetInverse(float zOffset) {
    float range = (-global_shadowAABBMinPrev.z + 16.0) - (-global_shadowAABBMaxPrev.z - 512.0);
    return zOffset / range;
}
#endif