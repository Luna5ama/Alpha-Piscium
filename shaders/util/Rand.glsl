/*
    References:
        [JIM17] Jimenez, Jorge. "Interleaved Gradient Noise". 2017.
            https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare/
        [ROB18] Roberts, Martine. "The Unreasonable Effectiveness of Quasirandom Sequences". 2018.
            https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
        [WOL22] Wolfe, Alan. "Interleaved Gradient Noise: A Different Kind of Low Discrepancy Sequence". 2022.
            https://blog.demofox.org/2022/01/01/interleaved-gradient-noise-a-different-kind-of-low-discrepancy-sequence/.
            (MIT License)
        [QUI] Quilez, Inigo. "Integer Hash Functions".
            (MIT License)

    You can find full license texts in /licenses
*/

#ifndef INCLUDE_util_Rand_glsl
#define INCLUDE_util_Rand_glsl a
#include "/_Base.glsl"

uniform sampler3D usam_stbnVec1;
uniform sampler3D usam_stbnUnitVec2;

float rand_stbnVec1(ivec2 texelPos, uint frame) {
    return texelFetch(usam_stbnVec1, ivec3(texelPos, frame) & ivec3(127, 127, 63), 0).x;
}

vec2 rand_stbnUnitVec201(ivec2 texelPos, uint frame) {
    return normalize(texelFetch(usam_stbnUnitVec2, ivec3(texelPos, frame) & ivec3(127, 127, 63), 0).xy);
}

vec2 rand_stbnUnitVec211(ivec2 texelPos, uint frame) {
    return normalize(texelFetch(usam_stbnUnitVec2, ivec3(texelPos, frame) & ivec3(127, 127, 63), 0).xy * 2.0 - 1.0);
}

// --------------------------------------------------------------------------------------------------------------------
// Interleaved Gradient Noise
// See [JIM17] and [WOL22]
float rand_IGN(vec2 v) {
    return fract(52.9829189 * fract(0.06711056 * v.x + 0.00583715 * v.y));
}

// See [JIM17] and [WOL22]
float rand_IGN(vec2 v, uint frame) {
    frame = frame % 64u;
    v = v + 5.588238 * float(frame);
    return fract(52.9829189 * fract(0.06711056 * v.x + 0.00583715 * v.y));
}

// --------------------------------------------------------------------------------------------------------------------
// R2 Sequence by Martine Roberts
// See [ROB18]
float rand_r2Seq1(uint idx) {
    const float g = 1.32471795724474602596;
    const float a = 1.0 / g;
    return fract(0.5 + a * idx);
}

// See [ROB18]
vec2 rand_r2Seq2(uint idx) {
    const float g = 1.32471795724474602596;
    const vec2 a = vec2(1.0 / g, 1.0 / (g * g));
    return fract(0.5 + a * idx);
}

vec3 rand_r2Seq3(uint idx) {
//    const float g = 1.32471795724474602596;
//    const vec3 a = vec3(1.0 / g, 1.0 / (g * g), 1.0 / (g * g * g));
    const vec3 a = vec3(0.7548776662466927600500267982588, 0.56984029099805326591218186327522, 0.43015970900194673408948540598911);
    return fract(a * idx + 0.5);
}

// --------------------------------------------------------------------------------------------------------------------
// Integer Hash Functions
//
uint rand_hash41(uvec4 x) {
    uint h = x.x * 0x8da6b343;
    h = 1103515245u * (h ^ (h >> 31)) + x.y;
    h = 1103515245u * (h ^ (h >> 31)) + x.z;
    h = 1103515245u * (h ^ (h >> 31)) + x.w;
    return h ^ (h >> 31);
}

uint rand_hash31(uvec3 x) {
    uint h = x.x * 0x8da6b343;
    h = 1103515245u * (h ^ (h >> 31)) + x.y;
    h = 1103515245u * (h ^ (h >> 31)) + x.z;
    return h ^ (h >> 31);
}

// https://www.shadertoy.com/view/4tXyWN
uint rand_hash21(uvec2 x) {
    uint h = x.x * 0x8da6b343;
    h = 1103515245u * (h ^ (h >> 31)) + x.y;
    return h ^ (h >> 31);
}

// https://www.shadertoy.com/view/llGSzw
uint rand_hash11(uint n) {
    // integer hash copied from Hugo Elias
    n = (n << 13U) ^ n;
    n = n * (n * n * 15731U + 789221U) + 1376312589U;
    return n;
}

// https://www.shadertoy.com/view/llGSzw
uvec3 rand_hash13(uint n) {
    // integer hash copied from Hugo Elias
    n = (n << 13U) ^ n;
    n = n * (n * n * 15731U + 789221U) + 1376312589U;
    uvec3 k = n * uvec3(n, n * 16807U, n * 48271U);
    return k;
}

// https://www.shadertoy.com/view/XlXcW4
uvec3 rand_hash33(uvec3 x) {
    x = ((x >> 8U) ^ x.yzx) * 1103515245u;
    x = ((x >> 8U) ^ x.yzx) * 1103515245u;
    x = ((x >> 8U) ^ x.yzx) * 1103515245u;

    return x;
}

float rand_uintToFloat(uint v) {
    return float(v) * (1.0 / float(0xffffffffU));
}

vec2 rand_uintToFloat(uvec2 v) {
    return vec2(v) * (1.0 / float(0xffffffffU));
}

vec3 rand_uintToFloat(uvec3 v) {
    return vec3(v) * (1.0 / float(0xffffffffU));
}

vec4 rand_uintToFloat(uvec4 v) {
    return vec4(v) * (1.0 / float(0xffffffffU));
}

uint rand_floatToUint(float v) {
    return uint(v * float(0xffffffffU));
}

uvec2 rand_floatToUint(vec2 v) {
    return uvec2(v * float(0xffffffffU));
}

uvec3 rand_floatToUint(vec3 v) {
    return uvec3(v * float(0xffffffffU));
}

uvec4 rand_floatToUint(vec4 v) {
    return uvec4(v * float(0xffffffffU));
}

#endif