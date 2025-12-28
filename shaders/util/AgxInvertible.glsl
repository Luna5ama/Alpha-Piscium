#ifndef INCLUDE_util_AgxInvertible_glsl
#define INCLUDE_util_AgxInvertible_glsl a

#include "Math.glsl"
#include "Colors2.glsl"

// By GeforceLegends (https://github.com/GeForceLegend) and linlin (https://github.com/bWFuanVzYWth/AgX)
vec3 _agxInvertible_agxCurveFoward(vec3 x) {
    const float scale0 = 59.507875;
    const float scale1 = 69.862789;

    x -= 20.0 / 33.0;
    vec3 type = vec3(floatBitsToUint(x) >> 31);

    vec3 scale = scale1 + (scale0 - scale1) * type;
    vec3 power0 = 13.0 / 4.0 - 0.25 * type;
    vec3 power1 = -4.0 / 13.0 + (4.0 / 13.0 - 1.0 / 3.0) * type;

    x = 2.0 * x * pow(1.0 + scale * pow(abs(x), power0), power1) + 0.5;
    return x;
}

vec3 _agxInvertible_agxCurveInverse(vec3 targetY) {
    const uint ITERATIONS = 4u;
    const float INIT_EPS = 0.25;
    const float ESP_DECAY = 0.15;

    vec3 x = targetY;
    float eps = INIT_EPS;

    for (uint i = 0u; i < ITERATIONS; ++i) {
        vec3 y = _agxInvertible_agxCurveFoward(x);
        vec3 dy = (_agxInvertible_agxCurveFoward(x + eps) - y) / eps;
        x -= (y - targetY) / dy;
        eps *= ESP_DECAY;
    }
    return x;
}
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
    const float EV_RANGE = SETTING_TONE_MAPPING_DYNAMIC_RANGE;
    const float EV_RANGE_HALF = EV_RANGE * 0.5;
    x = clamp(log2(x), -EV_RANGE_HALF, EV_RANGE_HALF);
    x = (x + EV_RANGE_HALF) / EV_RANGE;
    return x;
}

vec3 _agxInvertible_decodeLog2Space(vec3 x) {
    const float EV_RANGE = SETTING_TONE_MAPPING_DYNAMIC_RANGE;
    const float EV_RANGE_HALF = EV_RANGE * 0.5;
    x = x * EV_RANGE - EV_RANGE_HALF;
    x = exp2(x);
    return x;
}

vec3 _agxInvertible_look(vec3 x) {
    // Default
    vec3 offset = vec3(0.0);
    vec3 slope = vec3(1.0);
    vec3 power = vec3(1.0);
    float sat = 1.0;

    #if SETTING_TONE_MAPPING_LOOK == 1
    // Golden
    slope = vec3(1.0, 0.9, 0.5);
    power = vec3(0.8);
    sat = 0.8;
    #elif SETTING_TONE_MAPPING_LOOK == 2
    // Punchy
    slope = vec3(1.0);
    power = vec3(1.35, 1.35, 1.35);
    sat = 1.4;
    #elif SETTING_TONE_MAPPING_LOOK == 3
    // Custom
    offset = vec3(SETTING_TONE_MAPPING_OFFSET_R, SETTING_TONE_MAPPING_OFFSET_G, SETTING_TONE_MAPPING_OFFSET_B);
    slope = vec3(SETTING_TONE_MAPPING_SLOPE_R, SETTING_TONE_MAPPING_SLOPE_G, SETTING_TONE_MAPPING_SLOPE_B);
    power = vec3(SETTING_TONE_MAPPING_POWER_R, SETTING_TONE_MAPPING_POWER_G, SETTING_TONE_MAPPING_POWER_B);
    sat = SETTING_TONE_MAPPING_SATURATION;
    #endif

    // ASC CDL
    x = pow(x * slope + offset, power);
    float luma = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, x);
    return luma + sat * (x - luma);
}

vec3 _agxInvertible_lookInverse(vec3 x) {
    // Default
    vec3 offset = vec3(0.0);
    vec3 slope = vec3(1.0);
    vec3 power = vec3(1.0);
    float sat = 1.0;

    #if SETTING_TONE_MAPPING_LOOK == 1
    // Golden
    slope = vec3(1.0, 0.9, 0.5);
    power = vec3(0.8);
    sat = 0.8;
    #elif SETTING_TONE_MAPPING_LOOK == 2
    // Punchy
    slope = vec3(1.0);
    power = vec3(1.35, 1.35, 1.35);
    sat = 1.4;
    #elif SETTING_TONE_MAPPING_LOOK == 3
    // Custom
    offset = vec3(SETTING_TONE_MAPPING_OFFSET_R, SETTING_TONE_MAPPING_OFFSET_G, SETTING_TONE_MAPPING_OFFSET_B);
    slope = vec3(SETTING_TONE_MAPPING_SLOPE_R, SETTING_TONE_MAPPING_SLOPE_G, SETTING_TONE_MAPPING_SLOPE_B);
    power = vec3(SETTING_TONE_MAPPING_POWER_R, SETTING_TONE_MAPPING_POWER_G, SETTING_TONE_MAPPING_POWER_B);
    sat = SETTING_TONE_MAPPING_SATURATION;
    #endif

    float luma = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, x);
    x = (x - luma) / sat + luma;
    x = pow(x, vec3(1.0) / power) - offset;
    x = x / slope;
    return x;
}

vec3 agxInvertible_forward(vec3 x) {
    vec3 y = max(x, 0.0);
    y = colors2_colorspaces_convert(COLORS2_WORKING_COLORSPACE, COLORS2_DRT_WORKING_COLORSPACE, y);
    y = max(y, 0.0);

    y = agx_mat * y;
    y = _agxInvertible_encodeLog2Space(y);
    y = _agxInvertible_agxCurveFoward(y);
    y = saturate(y);
    y = _agxInvertible_look(y);
    y = agx_mat_inv * y;
    y = colors2_eotf(COLORS2_TF_SRGB, y);

    y = saturate(y);
    y = colors2_colorspaces_convert(COLORS2_DRT_WORKING_COLORSPACE, COLORS2_OUTPUT_COLORSPACE, y);
    y = saturate(y);
    y = colors2_oetf(COLORS2_OUTPUT_TF, y);
    y = saturate(y);

    return y;
}

vec3 agxInvertible_inverse(vec3 y) {
    vec3 x = saturate(y);
    x = colors2_eotf(COLORS2_OUTPUT_TF, x);
    x = saturate(x);
    x = colors2_colorspaces_convert(COLORS2_OUTPUT_COLORSPACE, COLORS2_DRT_WORKING_COLORSPACE, x);
    x = saturate(x);

    x = colors2_oetf(COLORS2_TF_SRGB, x);
    x = agx_mat * x;
    x = _agxInvertible_lookInverse(x);
    x = saturate(x);
    x = _agxInvertible_agxCurveInverse(x);
    x = _agxInvertible_decodeLog2Space(x);
    x = agx_mat_inv * x;

    x = max(x, 0.0);
    x = colors2_colorspaces_convert(COLORS2_DRT_WORKING_COLORSPACE, COLORS2_WORKING_COLORSPACE, x);
    return x;
}

#endif