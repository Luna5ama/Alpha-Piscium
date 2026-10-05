#ifndef INCLUDE_util_Colors_glsl
#define INCLUDE_util_Colors_glsl a
/*
    References:
        [ERI07] Ericson, Christer. "Converting RGB to LogLuv in a fragment shader". 2007.
            https://realtimecollisiondetection.net/blog/?p=15
        [LAR98] Wikipedia. "The LogLuv Encoding for Full Gamut, High Dynamic Range Images". 1998.
            http://www.anyhere.com/gward/papers/jgtpap1.pdf
        [ROS18] Rosseaux, Benjamin. "Matrix-based RGB from/to YCoCg color space conversion". 2018.
            CC0 License (Public Domain).
            https://www.shadertoy.com/view/4dXGzN
        [WIK26a] Wikipedia. "CIELUV". 2026.
            https://en.wikipedia.org/wiki/CIELUV

*/
#include "/Base.glsl"
#include "Math.glsl"
#include "/util/colors/ColorSpaces.glsl"
#include "/util/colors/TransferFunctions.glsl"

#define COLORS_MATERIAL_COLORSPACE SETTING_MATERIAL_COLOR_SPACE
#define COLORS_MATERIAL_TF SETTING_MATERIAL_TRANSFER_FUNC

#define COLORS_CONSTANTS_COLORSPACE COLORS_COLORSPACES_ACES_AP0
#define COLORS_CONSTANTS_TF COLORS_TF_IDENTITY

#define COLORS_WORKING_COLORSPACE SETTING_WORKING_COLOR_SPACE
#define COLORS_DRT_WORKING_COLORSPACE SETTING_DRT_WORKING_COLOR_SPACE

#define COLORS_GRADING_COLORSPACE SETTING_COLOR_GRADING_COLOR_SPACE
#define COLORS_GRADING_TF SETTING_COLOR_GRADING_TRANSFER_FUNC

#define COLORS_OUTPUT_COLORSPACE SETTING_OUTPUT_COLOR_SPACE
#define COLORS_OUTPUT_TF SETTING_OUTPUT_TRANSFER_FUNC

#define colors_material_toWorkSpace(x) colors_colorspaces_convert(COLORS_MATERIAL_COLORSPACE, COLORS_WORKING_COLORSPACE, colors_eotf(COLORS_MATERIAL_TF, x))

#define colors_constants_toWorkSpace(x) colors_colorspaces_convert(COLORS_CONSTANTS_COLORSPACE, COLORS_WORKING_COLORSPACE, x)

// ------------------------------------------------------- YCoCg -------------------------------------------------------
// [ROS18]
const mat3 _SRGB_TO_YCOCG = mat3(
    0.25, 0.5, -0.25,
    0.5, 0.0, 0.5,
    0.25, -0.5, -0.25
);

// [ROS18]
vec3 colors_RGBToYCoCg(vec3 color) {
    return _SRGB_TO_YCOCG * color;
}

// [ROS18]
const mat3 _YCOCG_TO_SRGB = mat3(
    1.0, 1.0, 1.0,
    1.0, 0.0, -1.0,
    -1.0, 1.0, -1.0
);

// [ROS18]
vec3 colors_YCoCgToRGB(vec3 color) {
    return _YCOCG_TO_SRGB * color;
}

// ---------------------------------------------------- LogLuv32 -------------------------------------------------
// [ERI07]
// M matrix, for encoding
const mat3 _COLORS_LOGLUV32_M = mat3(
    0.2209, 0.3390, 0.4184,
    0.1138, 0.6780, 0.7319,
    0.0102, 0.1130, 0.2969
);

// [ERI07]
vec4 colors_sRGBToLogLuv32(in vec3 vRGB)  {
    if (all(lessThanEqual(vRGB, vec3(0.0)))) {
        return vec4(0.0);
    }
    vec4 vResult;
    vec3 Xp_Y_XYZp = _COLORS_LOGLUV32_M * vRGB;
    Xp_Y_XYZp = max(Xp_Y_XYZp, vec3(1e-6, 1e-6, 1e-6));
    vResult.xy = Xp_Y_XYZp.xy / Xp_Y_XYZp.z;
    float Le = 2 * log2(Xp_Y_XYZp.y) + 127;
    vResult.w = fract(Le);
    vResult.z = (Le - (floor(vResult.w * 255.0f)) / 255.0f) / 255.0f;
    return vResult;
}

// [ERI07]
// Inverse M matrix, for decoding
const mat3 _COLORS_LOGLUV32_INVERSE_M = mat3(
    6.0014, -2.7008, -1.7996,
    -1.3320, 3.1029, -5.7721,
    0.3008, -1.0882, 5.6268
);

// [ERI07]
vec3 colors_LogLuv32ToSRGB(in vec4 vLogLuv) {
    if (all(lessThanEqual(vLogLuv, vec4(0.0)))) {
        return vec3(0.0);
    }
    float Le = vLogLuv.z * 255 + vLogLuv.w;
    vec3 Xp_Y_XYZp;
    Xp_Y_XYZp.y = exp2((Le - 127) / 2);
    Xp_Y_XYZp.z = Xp_Y_XYZp.y / vLogLuv.y;
    Xp_Y_XYZp.x = vLogLuv.x * Xp_Y_XYZp.z;
    vec3 vRGB = _COLORS_LOGLUV32_INVERSE_M * Xp_Y_XYZp;
    return max(vRGB, 0);
}

// [LAR98], [WIK26a]
// vec2(4.0, 9.0) * 410.0 / 255.0, 410 is the magic number to scale uv to fit in 0-255 range, see [LAR98]
const vec2 _COLORS_UV_MUL = vec2(6.4313725490, 14.4705882353);
const vec3 _COLORS_UV_DIV = vec3(1.0, 15.0, 3.0);
const float _COLORS_UV_INV_MUL = 0.6219512195; // 1.0 / 410.0 * 255.0
const vec2 _COLORS_UV_Z_MUL = vec2(-3.0, -20.0);

uint colors_workingColorToFP16Luv(vec3 color) {
    vec3 xyz = colors_colorspaces_convert(COLORS_WORKING_COLORSPACE, COLORS_COLORSPACES_CIE_XYZ, color);
    float uvDiv = safeRcp(dot(xyz, _COLORS_UV_DIV));
    vec2 uv = xyz.xy * uvDiv * _COLORS_UV_MUL;
    uv = clamp(uv, vec2(0.0), vec2(255.0));
    uint result = packHalf2x16(vec2(0.0, xyz.y)) & 0xFFFF0000u;
    result = bitfieldInsert(result, packUnorm4x8(vec4(uv, 0.0, 0.0)), 0, 16);
    return result;
}

vec3 colors_FP16LuvToWorkingColor(uint luv) {
    vec2 uv = unpackUnorm4x8(luv).xy * _COLORS_UV_INV_MUL;
    float Y = unpackHalf2x16(luv).y;
    float rcp4VTimeY = safeRcp(uv.y * 4.0) * Y;
    vec3 xyz;
    xyz.x = 9.0 * uv.x * rcp4VTimeY;
    xyz.y = Y;
    xyz.z = (12.0 + dot(_COLORS_UV_Z_MUL, uv)) * rcp4VTimeY;
    return colors_colorspaces_convert(COLORS_COLORSPACES_CIE_XYZ, COLORS_WORKING_COLORSPACE, xyz);
}

#endif
