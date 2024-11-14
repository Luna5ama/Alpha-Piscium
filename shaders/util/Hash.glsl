// MIT License
// Copyright © 2017,2024 Inigo Quilez
// You can find the full license text in /licenses/MIT.txt
#ifndef INCLUDE_Hash.glsl
#define INCLUDE_Hash.glsl
#include "../_Base.glsl"

uint hash41(uvec4 x) {
    uint h = x.x * 0x8da6b343;
    h = 1103515245u * (h ^ (h >> 31)) + x.y;
    h = 1103515245u * (h ^ (h >> 31)) + x.z;
    h = 1103515245u * (h ^ (h >> 31)) + x.w;
    return h ^ (h >> 31);
}

uint hash31(uvec3 x) {
    uint h = x.x * 0x8da6b343;
    h = 1103515245u * (h ^ (h >> 31)) + x.y;
    h = 1103515245u * (h ^ (h >> 31)) + x.z;
    return h ^ (h >> 31);
}

uint hash21(uvec2 x) {
    uint h = x.x * 0x8da6b343;
    h = 1103515245u * (h ^ (h >> 31)) + x.y;
    return h ^ (h >> 31);
}

uint hash11(uint n) {
    // integer hash copied from Hugo Elias
    n = (n << 13U) ^ n;
    n = n * (n * n * 15731U + 789221U) + 1376312589U;
    return n;
}

uvec3 hash13(uint n) {
    // integer hash copied from Hugo Elias
    n = (n << 13U) ^ n;
    n = n * (n * n * 15731U + 789221U) + 1376312589U;
    uvec3 k = n * uvec3(n, n * 16807U, n * 48271U);
    return k;
}

uvec3 hash33(uvec3 x) {
    x = ((x >> 8U) ^ x.yzx) * 1103515245u;
    x = ((x >> 8U) ^ x.yzx) * 1103515245u;
    x = ((x >> 8U) ^ x.yzx) * 1103515245u;

    return x;
}

float hash_uintToFloat(uint v) {
    return float(v) * (1.0 / float(0xffffffffU));
}

vec2 hash_uintToFloat(uvec2 v) {
    return vec2(v) * (1.0 / float(0xffffffffU));
}

vec3 hash_uintToFloat(uvec3 v) {
    return vec3(v) * (1.0 / float(0xffffffffU));
}

vec4 hash_uintToFloat(uvec4 v) {
    return vec4(v) * (1.0 / float(0xffffffffU));
}

uint hash_floatToUint(float v) {
    return uint(v * float(0xffffffffU));
}

uvec2 hash_floatToUint(vec2 v) {
    return uvec2(v * float(0xffffffffU));
}

uvec3 hash_floatToUint(vec3 v) {
    return uvec3(v * float(0xffffffffU));
}

uvec4 hash_floatToUint(vec4 v) {
    return uvec4(v * float(0xffffffffU));
}

#endif