/*
    References:
        [QUI17] Quilez, Inigo. "Integer Hash Functions". 2017.
            MIT License. Copyright (c) 2017,2024 Inigo Quilez.
            https://www.shadertoy.com/view/XlXcW4 (hash33)
            https://www.shadertoy.com/view/llGSzw (hash11)
            https://www.shadertoy.com/view/4tXyWN (hash21)
        [JAR20] Jarzynski, Mark. Olano, Marc. "Hash Functions for GPU Rendering". 2020.
            https://jcgt.org/published/0009/03/02/

        You can find full license texts in /licenses
*/

#ifndef INCLUDE_util_Hash_glsl
#define INCLUDE_util_Hash_glsl a

// --------------------------------------------------- Intenal Utils ---------------------------------------------------
uint _hash_sum(uvec2 v) {
    return v.x + v.y;
}

uint _hash_sum(uvec3 v) {
    return v.x + v.y + v.z;
}

uint _hash_sum(uvec4 v) {
    return v.x + v.y + v.z + v.w;
}

const uvec4 _HASH_LINEAR_COMB_MAGIC_MUL = uvec4(19u, 47u, 101u, 131u);
const uint _HASH_LINEAR_COMB_MAGIC_ADD = 173u;

uint _hash_linear_comb(uvec2 v) {
    return _hash_sum(_HASH_LINEAR_COMB_MAGIC_MUL.xy * v) + _HASH_LINEAR_COMB_MAGIC_ADD;
}

uint _hash_linear_comb(uvec3 v) {
    return _hash_sum(_HASH_LINEAR_COMB_MAGIC_MUL.xyz * v) + _HASH_LINEAR_COMB_MAGIC_ADD;
}

uint _hash_linear_comb(uvec4 v) {
    return _hash_sum(_HASH_LINEAR_COMB_MAGIC_MUL * v) + _HASH_LINEAR_COMB_MAGIC_ADD;
}


// -------------------------------------------------------- LCG --------------------------------------------------------
uint hash_lcg_11(uint p) {
    return p * 1664525u + 1013904223u;
}

uint hash_lcg_21_linear(uvec2 p) {
    return hash_lcg_11(_hash_linear_comb(p));
}

uint hash_lcg_31_linear(uvec3 p) {
    return hash_lcg_11(_hash_linear_comb(p));
}

uint hash_lcg_41_linear(uvec4 p) {
    return hash_lcg_11(_hash_linear_comb(p));
}

// -------------------------------------------------------- PCG --------------------------------------------------------
uint hash_pcg_11(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) *277803737u;
    return (word >> 22u) ^ word;
}

uvec2 hash_pcg2d_22(uvec2 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> 16u);
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> 16u);
    return v;
}


uvec3 hash_pcg3d16_33(uvec3 v) {
    v = v * 12829u + 47989u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v >>= 16u;
    return v;
}

uvec3 hash_pcg3d_33(uvec3 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v = v ^ (v >> 16u);
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

uvec4 hash_pcg4d_44(uvec4 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    v = v ^ (v >> 16u);
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    return v;
}

// ----------------------------------------------------- XXHASH32 -----------------------------------------------------
const uint _HASH_XXHASH32_PRIME32_2 = 2246822519U;
const uint _HASH_XXHASH32_PRIME32_3 = 3266489917U;
const uint _HASH_XXHASH32_PRIME32_4 = 668265263U;
const uint _HASH_XXHASH32_PRIME32_5 = 374761393U;

uint hash_xxhash32_11(uint p) {
    uint h32 = p + _HASH_XXHASH32_PRIME32_5;
    h32 = _HASH_XXHASH32_PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = _HASH_XXHASH32_PRIME32_2 * (h32 ^ (h32 >> 15));
    h32 = _HASH_XXHASH32_PRIME32_3 * (h32 ^ (h32 >> 13));
    return h32 ^ (h32 >> 16);
}

uint hash_xxhash32_21_multibyte(uvec2 p) {
    uint h32 = p.y + _HASH_XXHASH32_PRIME32_5 + p.x * _HASH_XXHASH32_PRIME32_3;
    h32 = _HASH_XXHASH32_PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = _HASH_XXHASH32_PRIME32_2 * (h32 ^ (h32 >> 15));
    h32 = _HASH_XXHASH32_PRIME32_3 * (h32 ^ (h32 >> 13));
    return h32 ^ (h32 >> 16);
}

uint hash_xxhash32_31_multibyte(uvec3 p) {
    uint h32 = p.z + _HASH_XXHASH32_PRIME32_5 + p.x * _HASH_XXHASH32_PRIME32_3;
    h32 = _HASH_XXHASH32_PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 += p.y * _HASH_XXHASH32_PRIME32_3;
    h32 = _HASH_XXHASH32_PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = _HASH_XXHASH32_PRIME32_2 * (h32 ^ (h32 >> 15));
    h32 = _HASH_XXHASH32_PRIME32_3 * (h32 ^ (h32 >> 13));
    return h32 ^ (h32 >> 16);
}

uint hash_xxhash32_41_multibyte(uvec4 p) {
    uint h32 = p.w + _HASH_XXHASH32_PRIME32_5 + p.x * _HASH_XXHASH32_PRIME32_3;
    h32 = _HASH_XXHASH32_PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 += p.y * _HASH_XXHASH32_PRIME32_3;
    h32 = _HASH_XXHASH32_PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 += p.z * _HASH_XXHASH32_PRIME32_3;
    h32 = _HASH_XXHASH32_PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = _HASH_XXHASH32_PRIME32_2 * (h32 ^ (h32 >> 15));
    h32 = _HASH_XXHASH32_PRIME32_3 * (h32 ^ (h32 >> 13));
    return h32 ^ (h32 >> 16);
}

// -------------------------------------------------------- IQINT --------------------------------------------------------
uint hash_iqint3_21(uvec2 v) {
    uvec2 q = 1103515245U * ((v>>1U) ^ (v.yx));
    uint n = 1103515245U * ((q.x) ^ (q.y >> 3U));
    return n;
}


// 1 to 1
// Not recommended
uint hash_11_q1(uint v) {
    return hash_lcg_11(v);
}

uint hash_11_q2(uint v) {
    return hash_pcg_11(v);
}

uint hash_11_q3(uint v) {
    return hash_xxhash32_11(v);
}

uint hash_11_q4(uint v) {
    return hash_pcg3d_33(uvec3(v, _HASH_XXHASH32_PRIME32_2, _HASH_XXHASH32_PRIME32_3)).x;
}

uint hash_11_q5(uint v) {
    return hash_pcg4d_44(uvec4(v, _HASH_XXHASH32_PRIME32_2, _HASH_XXHASH32_PRIME32_3, _HASH_XXHASH32_PRIME32_4)).x;
}


// 2 to 1
// Not recommended
uint hash_21_q1(uvec2 v) {
    return hash_lcg_21_linear(v);
}

uint hash_21_q2(uvec2 v) {
    return hash_iqint3_21(v);
}

uint hash_21_q3(uvec2 v) {
    return hash_xxhash32_21_multibyte(v);
}

uint hash_21_q4(uvec2 v) {
    return hash_pcg3d_33(uvec3(v, _HASH_XXHASH32_PRIME32_2)).x;
}

uint hash_21_q5(uvec2 v) {
    return hash_pcg4d_44(uvec4(v, _HASH_XXHASH32_PRIME32_2, _HASH_XXHASH32_PRIME32_3)).x;
}


// 3 to 1
// Not recommended
uint hash_31_q1(uvec3 v) {
    return hash_lcg_31_linear(v);
}

uint hash_31_q2(uvec3 v) {
    return hash_pcg3d16_33(v).x;
}

uint hash_31_q3(uvec3 v) {
    return hash_xxhash32_31_multibyte(v);
}

uint hash_31_q4(uvec3 v) {
    return hash_pcg3d_33(v).x;
}

uint hash_31_q5(uvec3 v) {
    return hash_pcg4d_44(uvec4(v, _HASH_XXHASH32_PRIME32_2)).x;
}


// 4 to 1
// Not recommended
uint hash_41_q1(uvec4 v) {
    return hash_lcg_41_linear(v);
}

uint hash_41_q2(uvec4 v) {
    return hash_xxhash32_41_multibyte(v);
}

uint hash_41_q3(uvec4 v) {
    return hash_xxhash32_41_multibyte(v);
}

uint hash_41_q4(uvec4 v) {
    return hash_pcg4d_44(v).x;
}

uint hash_41_q5(uvec4 v) {
    return hash_pcg4d_44(v).x;
}

// 2 to 2
uvec2 hash_22_q1(uvec2 v) {
    return hash_pcg2d_22(v);
}

uvec2 hash_22_q2(uvec2 v) {
    return hash_pcg3d_33(uvec3(v, _HASH_XXHASH32_PRIME32_2)).xy;
}

uvec2 hash_22_q3(uvec2 v) {
    return hash_pcg4d_44(uvec4(v, _HASH_XXHASH32_PRIME32_2, _HASH_XXHASH32_PRIME32_3)).xy;
}

// 3 to 3
uvec3 hash_33_q1(uvec3 v) {
    return hash_pcg3d16_33(v);
}

uvec3 hash_33_q2(uvec3 v) {
    return hash_pcg3d_33(v);
}

uvec3 hash_33_q3(uvec3 v) {
    return hash_pcg4d_44(uvec4(v, _HASH_XXHASH32_PRIME32_2)).xyz;
}

// 4 to 4
uvec4 hash_44_q1(uvec4 v) {
    return hash_pcg4d_44(v);
}

uvec4 hash_44_q2(uvec4 v) {
    return hash_pcg4d_44(v);
}

uvec4 hash_44_q3(uvec4 v) {
    return hash_pcg4d_44(v);
}

// ---------------------------------------------------- Conversions ----------------------------------------------------
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