#ifndef INCLUDE_util_Rand_glsl
#define INCLUDE_util_Rand_glsl a
/*
    References:
        [GIL23] Gilcher, Pascal. "A Better R2 Sequence". 2023.
            https://www.martysmods.com/a-better-r2-sequence/
        [JIM17] Jimenez, Jorge. "Interleaved Gradient Noise". 2017.
            https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare/
        [ROB18] Roberts, Martine. "The Unreasonable Effectiveness of Quasirandom Sequences". 2018.
            https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
        [WOL22] Wolfe, Alan. "Interleaved Gradient Noise: A Different Kind of Low Discrepancy Sequence". 2022.
            MIT License. Copyright (c) 2019 Alan Wolfe.
            https://blog.demofox.org/2022/01/01/interleaved-gradient-noise-a-different-kind-of-low-discrepancy-sequence/

        You can find full license texts in /licenses
*/


#include "/Base.glsl"

float rand_stbnVec1(ivec2 texelPos, uint frame) {
    return texelFetch(usam_stbnVec1, ivec3(texelPos, frame) & ivec3(127, 127, 63), 0).x;
}

vec2 rand_stbnVec2(ivec2 texelPos, uint frame) {
    return texelFetch(usam_stbnVec2, ivec3(texelPos, frame) & ivec3(127, 127, 63), 0).xy;
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

// ---------------------------------------------------- R2 Sequence ----------------------------------------------------
// See [ROB18] and [GIL23]
float rand_r2Seq1(uint idx) {
    const float g = 1.6180339887498948482;
    const float a = 1.0 - 1.0 / g;
    return 1.0 - fract(a * idx - 0.5);
}

vec2 rand_r2Seq2(uint idx) {
    const float g = 1.32471795724474602596;
    const vec2 a = 1.0 - vec2(1.0 / g, 1.0 / (g * g));
    return 1.0 - fract(a * idx - 0.5);
}

vec3 rand_r2Seq3(uint idx) {
    const float g = 1.22074408460575947536;
    const vec3 a = 1.0 - vec3(1.0 / g, 1.0 / (g * g), 1.0 / (g * g * g));
    return 1.0 - fract(a * idx - 0.5);
}

vec3 rand_sampleInCone(vec3 center, float coneHalfAngle, vec2 rand) {
    // Random azimuth angle
    float phi = PI_2 * rand.x;

    // Uniform sampling on spherical cap
    float cosTheta = cos(coneHalfAngle);
    float cosAlpha = mix(1.0, cosTheta, rand.y);
    float sinAlpha = sqrt(1.0 - cosAlpha * cosAlpha);

    // Build orthonormal basis (u, v, center)
    vec3 other = abs(center.x) < 0.9 ? vec3(1, 0, 0) : vec3(0, 1, 0);
    vec3 u = normalize(cross(center, other));
    vec3 v = cross(center, u);

    // Final direction
    return normalize(cosAlpha * center + sinAlpha * (cos(phi) * u + sin(phi) * v));
}

#endif