/*
    References:
        [QUI08] Quilez, Inigo. "Value Noise Derivatives". 2008.
            https://iquilezles.org/articles/morenoise/
        [QUI13a] Quilez, Inigo. "Noise - value - 2D ". 2013.
            MIT License. Copyright (c) 2013 Inigo Quilez.
            https://www.shadertoy.com/view/lsf3WH
        [QUI13b] Quilez, Inigo. "Noise - value - 3D ". 2013.
            MIT License. Copyright (c) 2013 Inigo Quilez.
            https://www.shadertoy.com/view/4sfGzS
        [QUI17a] Quilez, Inigo. "Noise - Value - 2D - Deriv ". 2017.
            MIT License. Copyright (c) 2017 Inigo Quilez.
            https://www.shadertoy.com/view/4dXBRH
        [QUI17b] Quilez, Inigo. "Noise - Value - 3D - Deriv". 2017.
            MIT License. Copyright (c) 2017 Inigo Quilez.
            https://www.shadertoy.com/view/XsXfRH

        You can find full license texts in /licenses
*/

#ifndef INCLUDE_util_GradientNoise_glsl
#define INCLUDE_util_GradientNoise_glsl a
#include "Hash.glsl"

// 0: cubic
// 1: quintic
#define _NOISE_INTERPOLANT 0

struct FBMParameters {
    float frequency;
    float persistence;
    float lacunarity;
    uint octaveCount;
};

#if _NOISE_INTERPOLANT == 1
#define _NOISE_INTERPO(w) (w * w * w * (w * (w * 6.0 - 15.0) + 10.0))
#define _NOISE_INTERPO_GRAD(w) (30.0 * w * w * (w * (w - 2.0) + 1.0))
#else
#define _NOISE_INTERPO(w) (w * w * (3.0 - 2.0 * w))
#define _NOISE_INTERPO_GRAD(w) (6.0 * w * (1.0 - w))
#endif

uvec2 _noise_hash_coord(vec2 x) {
    return uvec2(ivec2(floor(x)));
}

uvec3 _noise_hash_coord(vec3 x) {
    return uvec3(ivec3(floor(x)));
}


// -------------------------------------------------- Value Noise 2D --------------------------------------------------
float _noise_value_2D_hash(uvec2 x) {
    return hash_uintToFloat(hash_21_q3(x)) * 2.0 - 1.0;
}

// [QUI13a]
float noise_value_2D_value(vec2 x) {
    uvec2 i = _noise_hash_coord(x);
    vec2 w = fract(x);

    vec2 u = _NOISE_INTERPO(w);

    float va = _noise_value_2D_hash(i + uvec2(0, 0));
    float vb = _noise_value_2D_hash(i + uvec2(1, 0));
    float vc = _noise_value_2D_hash(i + uvec2(0, 1));
    float vd = _noise_value_2D_hash(i + uvec2(1, 1));

    float xy0 = mix(va, vb, u.x);
    float xy1 = mix(vc, vd, u.x);
    float value = mix(xy0, xy1, u.y);

    return value;
}

// [QUI17a]
vec3 noise_value_2D_valueGrad(vec2 x) {
    uvec2 i = _noise_hash_coord(x);
    vec2 w = fract(x);

    vec2 u = _NOISE_INTERPO(w);
    vec2 du = _NOISE_INTERPO_GRAD(w);

    float va = _noise_value_2D_hash(i + uvec2(0, 0));
    float vb = _noise_value_2D_hash(i + uvec2(1, 0));
    float vc = _noise_value_2D_hash(i + uvec2(0, 1));
    float vd = _noise_value_2D_hash(i + uvec2(1, 1));

    float xy0 = mix(va, vb, u.x);
    float xy1 = mix(vc, vd, u.x);
    float value = mix(xy0, xy1, u.y);

    vec2 grad = du * vec2(
        mix(vb - va, vd - vc, u.y),
        mix(vc - va, vd - vb, u.x)
    );

    return vec3(value, grad);
}

float noise_value_2D_value_fbm(FBMParameters params, vec2 position) {
    float value = 0.0;
    float amplitude = 1.0;
    float currentFrequency = params.frequency;
    for (uint i = 0; i < params.octaveCount; i++) {
        value += noise_value_2D_value(position * currentFrequency) * amplitude;
        amplitude *= params.persistence;
        currentFrequency *= params.lacunarity;
    }
    return value;
}


// -------------------------------------------------- Value Noise 3D --------------------------------------------------
float _noise_value_3D_hash(uvec3 x) {
    return hash_uintToFloat(hash_31_q3(x)) * 2.0 - 1.0;
}

// [QUI13b]
float noise_value_3D_value(vec3 x) {
    uvec3 i = _noise_hash_coord(x);
    vec3 w = fract(x);

    vec3 u = _NOISE_INTERPO(w);

    float va = _noise_value_3D_hash(i + uvec3(0, 0, 0));
    float vb = _noise_value_3D_hash(i + uvec3(1, 0, 0));
    float vc = _noise_value_3D_hash(i + uvec3(0, 1, 0));
    float vd = _noise_value_3D_hash(i + uvec3(1, 1, 0));
    float ve = _noise_value_3D_hash(i + uvec3(0, 0, 1));
    float vf = _noise_value_3D_hash(i + uvec3(1, 0, 1));
    float vg = _noise_value_3D_hash(i + uvec3(0, 1, 1));
    float vh = _noise_value_3D_hash(i + uvec3(1, 1, 1));

    float xy0 = mix(mix(va, vb, u.x), mix(vc, vd, u.x), u.y);
    float xy1 = mix(mix(ve, vf, u.x), mix(vg, vh, u.x), u.y);
    float value = mix(xy0, xy1, u.z);

    return value;
}

// [QUI17b]
vec4 noise_value_3D_valueGrad(vec3 x) {
    uvec3 i = _noise_hash_coord(x);
    vec3 w = fract(x);

    vec3 u = _NOISE_INTERPO(w);
    vec3 du = _NOISE_INTERPO_GRAD(w);

    float va = _noise_value_3D_hash(i + uvec3(0, 0, 0));
    float vb = _noise_value_3D_hash(i + uvec3(1, 0, 0));
    float vc = _noise_value_3D_hash(i + uvec3(0, 1, 0));
    float vd = _noise_value_3D_hash(i + uvec3(1, 1, 0));
    float ve = _noise_value_3D_hash(i + uvec3(0, 0, 1));
    float vf = _noise_value_3D_hash(i + uvec3(1, 0, 1));
    float vg = _noise_value_3D_hash(i + uvec3(0, 1, 1));
    float vh = _noise_value_3D_hash(i + uvec3(1, 1, 1));

    float xy0 = mix(mix(va, vb, u.x), mix(vc, vd, u.x), u.y);
    float xy1 = mix(mix(ve, vf, u.x), mix(vg, vh, u.x), u.y);
    float value = mix(xy0, xy1, u.z);

    vec3 grad = du * vec3(
        mix(mix(vb - va, vd - vc, u.y), mix(vf - ve, vh - vg, u.y), u.z),
        mix(mix(vc - va, vd - vb, u.x), mix(vg - ve, vh - vf, u.x), u.z),
        mix(mix(ve - va, vf - vb, u.x), mix(vg - vc, vh - vd, u.x), u.y)
    );

    return vec4(value, grad);
}

float noise_value_3D_value_fbm(FBMParameters params, vec3 position) {
    float value = 0.0;
    float amplitude = 1.0;
    float currentFrequency = params.frequency;
    for (uint i = 0; i < params.octaveCount; i++) {
        value += noise_value_3D_value(position * currentFrequency) * amplitude;
        amplitude *= params.persistence;
        currentFrequency *= params.lacunarity;
    }
    return value;
}

#endif