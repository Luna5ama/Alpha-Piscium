#ifndef INCLUDE_RTWSM.glsl
#define INCLUDE_RTWSM.glsl

#include "../utils/Settings.glsl"

#if RTWSM_IMAP_SIZE == 128

#define WORKGROUP16_COUNT 8
#define WORKGROUP128_COUNT 1

#elif RTWSM_IMAP_SIZE == 256

#define WORKGROUP16_COUNT 16
#define WORKGROUP128_COUNT 2

#elif RTWSM_IMAP_SIZE == 512

#define WORKGROUP16_COUNT 32
#define WORKGROUP128_COUNT 4

#elif RTWSM_IMAP_SIZE == 1024

#define WORKGROUP16_COUNT 64
#define WORKGROUP128_COUNT 8

#elif RTWSM_IMAP_SIZE == 2048

#define WORKGROUP16_COUNT 128
#define WORKGROUP128_COUNT 16

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
    return textureLod(shadowMap, coord, lod);
}

float rtwsm_sampleShadowDepth(sampler2D shadowMap, vec2 coord, float lod) {
    return textureLod(shadowMap, coord, lod).r;
}

float rtwsm_sampleShadowDepthOffset(sampler2DShadow shadowMap, vec3 coord, float lod, vec2 offsetPixels) {
    vec3 offsetPos = coord;
    offsetPos.xy += offsetPixels * SHADOW_MAP_SIZE.zw;
    return textureLod(shadowMap, offsetPos, lod);
}

float rtwsm_sampleShadowDepthOffset(sampler2D shadowMap, vec2 coord, float lod, vec2 offsetPixels) {
    vec2 offsetPos = coord;
    offsetPos += offsetPixels * SHADOW_MAP_SIZE.zw;
    return textureLod(shadowMap, offsetPos, lod).r;
}

//float rtwsm_linearDepth(float d) {
//    return mix(uShadowMap.planeZ.x, uShadowMap.planeZ.y, d);
//}
#endif