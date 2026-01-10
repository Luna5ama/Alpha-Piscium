#ifndef INCLUDE_util_AgxInvertible_glsl
#define INCLUDE_util_AgxInvertible_glsl a

#include "Math.glsl"
#include "Colors.glsl"
#include "Colors2.glsl"

const mat3 agx_mat = mat3(
    0.842479062253094, 0.0423282422610123, 0.0423756549057051,
    0.0784335999999992, 0.878468636469772, 0.0784336,
    0.0792237451477643, 0.0791661274605434, 0.879142973793104
);
const mat3 agx_mat_inv = mat3(
    1.19687900512017, -0.0528968517574562, -0.0529716355144438,
    -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
    -0.0990297440797205, -0.0989611768448433, 1.15107367264116
);

vec3 _agxInvertible_encodeLog2Space(vec3 x) {
    const float EV_RANGE = 16.5;
    const float EV_RANGE_HALF = EV_RANGE * 0.5;
    x = clamp(log2(x), -EV_RANGE_HALF, EV_RANGE_HALF);
    x = (x + EV_RANGE_HALF) / EV_RANGE;
    return x;
}

vec3 _agxInvertible_decodeLog2Space(vec3 x) {
    const float EV_RANGE = 16.5;
    const float EV_RANGE_HALF = EV_RANGE * 0.5;
    x = x * EV_RANGE - EV_RANGE_HALF;
    x = exp2(x);
    return x;
}

vec3 agxInvertible_forward(vec3 x) {
    vec3 y = max(x, 0.0);
    y = agx_mat * y;
    y = _agxInvertible_encodeLog2Space(y);
    return saturate(y);
}

vec3 agxInvertible_inverse(vec3 y) {
    vec3 x = saturate(y);
    x = _agxInvertible_decodeLog2Space(x);
    x = agx_mat_inv * x;
    return max(x, 0.0);
}

#endif