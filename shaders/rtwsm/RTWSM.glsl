#ifndef INCLUDE_rtwsm_RTWSM.glsl
#define INCLUDE_rtwsm_RTWSM.glsl

#include "../_Util.glsl"

#if SETTING_RTWSM_IMAP_SIZE == 256

#define IMAP_SIZE_D2 128
#define IMAP_SIZE_D16 16
#define IMAP_SIZE_D32 8
#define IMAP_SIZE_D128 2

#define IMAP_SIZE_Y 262
#define IMAP2D_V_RANGE (256.0 / 262.0)
#define IMAP2D_V_CLAMP (255.5 / 262.0)

#define IMAP1D_X_Y 256
#define IMAP1D_X_V (256.5 / 262.0)
#define IMAP1D_Y_Y 257
#define IMAP1D_Y_V (257.5 / 262.0)

#define WARP_X_Y 258
#define WARP_X_V (258.5 / 262.0)
#define WARP_Y_Y 259
#define WARP_Y_V (259.5 / 262.0)

#define TEXELSIZE_X_Y 260
#define TEXELSIZE_X_V (260.5 / 262.0)
#define TEXELSIZE_Y_Y 261
#define TEXELSIZE_Y_V (261.5 / 262.0)

#elif SETTING_RTWSM_IMAP_SIZE == 512

#define IMAP_SIZE_D2 256
#define IMAP_SIZE_D16 32
#define IMAP_SIZE_D32 16
#define IMAP_SIZE_D128 4

#define IMAP_SIZE_Y 518
#define IMAP2D_V_RANGE (512.0 / 518.0)
#define IMAP2D_V_CLAMP (511.5 / 518.0)

#define IMAP1D_X_Y 512
#define IMAP1D_X_V (512.5 / 518.0)
#define IMAP1D_Y_Y 513
#define IMAP1D_Y_V (513.5 / 518.0)

#define WARP_X_Y 514
#define WARP_X_V (514.5 / 518.0)
#define WARP_Y_Y 515
#define WARP_Y_V (515.5 / 518.0)

#define TEXELSIZE_X_Y 516
#define TEXELSIZE_X_V (516.5 / 518.0)
#define TEXELSIZE_Y_Y 517
#define TEXELSIZE_Y_V (517.5 / 518.0)

#elif SETTING_RTWSM_IMAP_SIZE == 1024

#define IMAP_SIZE_D2 512
#define IMAP_SIZE_D16 64
#define IMAP_SIZE_D32 32
#define IMAP_SIZE_D128 8

#define IMAP_SIZE_Y 1030

#define IMAP2D_V_RANGE (1024.0 / 1030.0)
#define IMAP2D_V_CLAMP (1023.5 / 1030.0)

#define IMAP1D_X_Y 1024
#define IMAP1D_X_V (1024.5 / 1030.0)
#define IMAP1D_Y_Y 1025
#define IMAP1D_Y_V (1025.5 / 1030.0)

#define WARP_X_Y 1026
#define WARP_X_V (1026.5 / 1030.0)
#define WARP_Y_Y 1027
#define WARP_Y_V (1027.5 / 1030.0)

#define TEXELSIZE_X_Y 1028
#define TEXELSIZE_X_V (1028.5 / 1030.0)
#define TEXELSIZE_Y_Y 1029
#define TEXELSIZE_Y_V (1029.5 / 1030.0)

#endif

vec2 rtwsm_warpTexCoord(sampler2D warpingMap, vec2 uv) {
    vec2 result = uv;
    result.x += texture(warpingMap, vec2(uv.x, WARP_X_V)).r;
    result.y += texture(warpingMap, vec2(uv.y, WARP_Y_V)).r;
    return result;
}

vec2 rtwsm_warpTexCoordTexelSize(sampler2D warpingMap, vec2 uv, out vec2 texelSize) {
    vec2 result = uv;
    result.x += texture(warpingMap, vec2(uv.x, WARP_X_V)).r;
    result.y += texture(warpingMap, vec2(uv.y, WARP_Y_V)).r;
    texelSize.x = texture(warpingMap, vec2(uv.x, TEXELSIZE_X_V)).r;
    texelSize.y = texture(warpingMap, vec2(uv.y, TEXELSIZE_Y_V)).r;
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

float rtwsm_linearDepthInverse(float depth) {
    return ((depth - shadowProjectionInverse[3][2]) / (4.0 * shadowProjectionInverse[2][2]) + 1.0) / 2.0;
}
#endif