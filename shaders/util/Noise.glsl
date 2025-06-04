/*
    References:
        [QUI08] Quilez, Inigo. "Value Noise Derivatives". 2008.
            https://iquilezles.org/articles/morenoise/
        [QUI17a] Quilez, Inigo. "Gradient Noise Derivatives". 2017.
            https://iquilezles.org/articles/gradientnoise/
        [QUI17b] Quilez, Inigo. "Noise - Value - 3D - Deriv". 2017.
            MIT License. Copyright (c) 2017 Inigo Quilez.
            https://www.shadertoy.com/view/XsXfRH
        [QUI17c] Quilez, Inigo. "Noise - Gradient - 3D - Deriv ". 2017.
            MIT License. Copyright (c) 2017 Inigo Quilez.
            https://www.shadertoy.com/view/4dffRH

    Contains code adopted from:
        https://blog.demofox.org/2022/01/01/interleaved-gradient-noise-a-different-kind-of-low-discrepancy-sequence/
        MIT License
        Copyright (c) 2019 Alan Wolfe

        https://www.shadertoy.com/view/XlXcW4 (hash33)
        https://www.shadertoy.com/view/llGSzw (hash11)
        https://www.shadertoy.com/view/4tXyWN (hash21)
        MIT License
        Copyright © 2017,2024 Inigo Quilez

        You can find full license texts in /licenses
*/

#ifndef INCLUDE_util_GradientNoise_glsl
#define INCLUDE_util_GradientNoise_glsl a
#include "Hash.glsl"

float _noise_value_2D_hash(uvec2 x) {
    return hash_uintToFloat(hash_21_q3(x)) * 2.0 - 1.0;
}

float _noise_value_3D_hash(uvec3 x) {
    return hash_uintToFloat(hash_31_q3(x)) * 2.0 - 1.0;
}

// [QUI17b]
vec4 noise_value_3D_valueGrad(in vec3 x) {
    uvec3 i = uvec3(ivec3(floor(x)));
    vec3 w = fract(x);

    vec3 u = w * w * w * (w * (w * 6.0 - 15.0) + 10.0);
    vec3 du = 30.0 * w * w * (w * (w - 2.0) + 1.0);

    float a = _noise_value_3D_hash(i + uvec3(0, 0, 0));
    float b = _noise_value_3D_hash(i + uvec3(1, 0, 0));
    float c = _noise_value_3D_hash(i + uvec3(0, 1, 0));
    float d = _noise_value_3D_hash(i + uvec3(1, 1, 0));
    float e = _noise_value_3D_hash(i + uvec3(0, 0, 1));
    float f = _noise_value_3D_hash(i + uvec3(1, 0, 1));
    float g = _noise_value_3D_hash(i + uvec3(0, 1, 1));
    float h = _noise_value_3D_hash(i + uvec3(1, 1, 1));

    float k0 = a;
    float k1 = b - a;
    float k2 = c - a;
    float k3 = e - a;
    float k4 = a - b - c + d;
    float k5 = a - c - e + g;
    float k6 = a - b - e + f;
    float k7 = -a + b + c - d + e - f - g + h;

    vec4 vdotu1 = vec4(1.0, u.x, u.y, u.z);
    vec4 vdotu2 = u.xyzx * u.yzxy;
    vdotu2.w *= u.z;
    vec4 vdotk1 = vec4(k0, k1, k2, k3);
    vec4 vdotk2 = vec4(k4, k5, k6, k7);

    vec4 dxdotu = vec4(1.0, u.yzy);
    dxdotu.w *= u.z;
    vec4 dxdotk = vec4(k1, k4, k6, k7);

    vec4 dydotu = vec4(1.0, u.zxz);
    dydotu.w *= u.x;
    vec4 dydotk = vec4(k2, k5, k4, k7);

    vec4 dzdotu = vec4(1.0, u.xyx);
    dzdotu.w *= u.y;
    vec4 dzdotk = vec4(k3, k6, k5, k7);

    return vec4(dot(vdotk1, vdotu1) + dot(vdotk2, vdotu2),
        du * vec3(dot(dxdotk, dxdotu), dot(dydotk, dydotu), dot(dzdotk, dzdotu)));
}

// [QUI17b]
float noise_value_3D_value(vec3 x) {
    uvec3 i = uvec3(ivec3(floor(x)));
    vec3 w = fract(x);

    // quintic interpolation
    vec3 u = w * w * w * (w * (w * 6.0 - 15.0) + 10.0);

    float a = _noise_value_3D_hash(i + uvec3(0, 0, 0));
    float b = _noise_value_3D_hash(i + uvec3(1, 0, 0));
    float c = _noise_value_3D_hash(i + uvec3(0, 1, 0));
    float d = _noise_value_3D_hash(i + uvec3(1, 1, 0));
    float e = _noise_value_3D_hash(i + uvec3(0, 0, 1));
    float f = _noise_value_3D_hash(i + uvec3(1, 0, 1));
    float g = _noise_value_3D_hash(i + uvec3(0, 1, 1));
    float h = _noise_value_3D_hash(i + uvec3(1, 1, 1));

    float k0 = a;
    float k1 = b - a;
    float k2 = c - a;
    float k3 = e - a;
    float k4 = a - b - c + d;
    float k5 = a - c - e + g;
    float k6 = a - b - e + f;
    float k7 = -a + b + c - d + e - f - g + h;

    vec4 vdotu1 = vec4(1.0, u.x, u.y, u.z);
    vec4 vdotu2 = u.xyzx * u.yzxy * vec4(1.0, 1.0, 1.0, u.z);

    vec4 vdotk1 = vec4(k0, k1, k2, k3);
    vec4 vdotk2 = vec4(k4, k5, k6, k7);

    return dot(vdotk1, vdotu1) + dot(vdotk2, vdotu2);
}

#endif