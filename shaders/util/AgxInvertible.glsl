#ifndef INCLUDE_util_AgxInvertible_glsl
#define INCLUDE_util_AgxInvertible_glsl a

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

vec3 _agxInvertible_encodeLog2Space(vec3 x, float evMin, float evMax) {
    float zeroPoint = exp2(evMin);
    return log2(x / zeroPoint + 1.0) / (evMax - evMin);
}

vec3 _agxInvertible_decodeLog2Space(vec3 x, float evMin, float evMax) {
    float zeroPoint = exp2(evMin);
    return (exp2(x * (evMax - evMin)) - 1.0) * zeroPoint;
}

vec3 agxInvertible_forwardRange(vec3 x, float evMin, float evMax) {
    vec3 y = max(x, 0.0);
    y = agx_mat * y;
    y = _agxInvertible_encodeLog2Space(y, evMin, evMax);
    return y;
}

vec3 agxInvertible_inverseRange(vec3 y, float evMin, float evMax) {
    vec3 x = _agxInvertible_decodeLog2Space(y, evMin, evMax);
    x = agx_mat_inv * x;
    return x;
}

vec3 agxInvertible_forward(vec3 x) {
    return agxInvertible_forwardRange(x, -16.5, 16.5);
}

vec3 agxInvertible_inverse(vec3 y) {
    vec3 x = agxInvertible_inverseRange(y, -16.5, 16.5);
    return max(x, 0.0);
}

#endif
