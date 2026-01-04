#ifndef INCLUDE_util_Colors_glsl
#define INCLUDE_util_Colors_glsl a
/*
    References:
        [ERI07] Ericson, Christer. "Converting RGB to LogLuv in a fragment shader". 2007.
            https://realtimecollisiondetection.net/blog/?p=15
        [ITU11] ITU. "Recommendation ITU-R BT.601-7". 2011.
            https://www.itu.int/rec/R-REC-BT.601-7-201103-I/en
        [ITU15] ITU. "Recommendation ITU-R BT.709-6". 2015.
            https://www.itu.int/rec/R-REC-BT.709
        [KAR13] Karis, Brian. "Tone mapping". Graphic Rants. 2013.
            https://graphicrants.blogspot.com/2013/12/tone-mapping.html
        [LAR98] Wikipedia. "The LogLuv Encoding for Full Gamut, High Dynamic Range Images". 1998.
            http://www.anyhere.com/gward/papers/jgtpap1.pdf
        [LOT16] Lottes, Timothy. "Optimized Reversible Tonemapper for Resolve". 2016.
            https://gpuopen.com/learn/optimized-reversible-tonemapper-for-resolve/
        [ROS18] Rosseaux, Benjamin. "Matrix-based RGB from/to YCoCg color space conversion". 2018.
            CC0 License (Public Domain).
            https://www.shadertoy.com/view/4dXGzN
        [WIK25a] Wikipedia. "Rec. 601". 2025.
            https://en.wikipedia.org/wiki/Rec._601
        [WIK25b] Wikipedia. "Rec. 709". 2025.
            https://en.wikipedia.org/wiki/Rec._709
        [WIK25c] Wikipedia. "sRGB". 2025.
            https://en.wikipedia.org/wiki/SRGB
        [WIK26a] Wikipedia. "CIELUV". 2026.
            https://en.wikipedia.org/wiki/CIELUV

*/
#include "/Base.glsl"
#include "Math.glsl"
#include "Colors2.glsl"

// ----------------------------------------------------- Rec. 601 -----------------------------------------------------
// [WIK25a]
vec3 colors_Rec601_encodeGamma(vec3 color) {
    vec3 lower = 4.5 * color;
    vec3 higher = pow(color, vec3(0.45)) * 1.099 - 0.099;
    return mix(lower, higher, greaterThanEqual(color, vec3(0.018)));
}

// [WIK25a]
vec3 colors_Rec601_decodeGamma(vec3 color) {
    vec3 lower = color / 4.5;
    vec3 higher = pow((color + 0.099) / 1.099, vec3(1.0 / 0.45));
    return mix(lower, higher, greaterThanEqual(color, vec3(0.081)));
}

// [ITU11]
float colors_Rec601_luma(vec3 color) {
    return dot(color, vec3(0.299, 0.587, 0.114));
}

// ----------------------------------------------------- Rec. 709 -----------------------------------------------------
// [WIK25b]
vec3 colors_Rec709_encodeGamma(vec3 color) {
    return colors_Rec601_encodeGamma(color);
}

// [WIK25b]
vec3 colors_Rec709_decodeGamma(vec3 color) {
    return colors_Rec601_decodeGamma(color);
}

// [ITU15]
float colors_Rec709_luma(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

// ------------------------------------------------------ sRGB --------------------------------------------------------
// [WIK25c]
float colors_sRGB_encodeGamma(float color) {
    float lower = 12.92 * color;
    float higher = pow(color, 1.0 / 2.4) * 1.055 - 0.055;
    return mix(lower, higher, color > 0.0031308);
}

// [WIK25c]
vec2 colors_sRGB_encodeGamma(vec2 color) {
    vec2 lower = 12.92 * color;
    vec2 higher = pow(color, vec2(1.0 / 2.4)) * 1.055 - 0.055;
    return mix(lower, higher, greaterThan(color, vec2(0.0031308)));
}

// [WIK25c]
vec3 colors_sRGB_encodeGamma(vec3 color) {
    vec3 lower = 12.92 * color;
    vec3 higher = pow(color, vec3(1.0 / 2.4)) * 1.055 - 0.055;
    return mix(lower, higher, greaterThan(color, vec3(0.0031308)));
}

// [WIK25c]
vec4 colors_sRGB_encodeGamma(vec4 color) {
    vec4 lower = 12.92 * color;
    vec4 higher = pow(color, vec4(1.0 / 2.4)) * 1.055 - 0.055;
    return mix(lower, higher, greaterThan(color, vec4(0.0031308)));
}

// [WIK25c]
float colors_sRGB_decodeGamma(float color) {
    float lower = color / 12.92;
    float higher = pow((color + 0.055) / 1.055, 2.4);
    return mix(lower, higher, color > 0.04045);
}

// [WIK25c]
vec3 colors_sRGB_decodeGamma(vec3 color) {
    vec3 lower = color / 12.92;
    vec3 higher = pow((color + 0.055) / 1.055, vec3(2.4));
    return mix(lower, higher, greaterThan(color, vec3(0.04045)));
}

// [WIK25c]
float colors_sRGB_luma(vec3 color) {
    return colors_Rec709_luma(color);
}

// ------------------------------------------------------- YCoCg -------------------------------------------------------
// [ROS18]
const mat3 _SRGB_TO_YCOCG = mat3(
    0.25, 0.5, -0.25,
    0.5, 0.0, 0.5,
    0.25, -0.5, -0.25
);

// [ROS18]
vec3 colors_SRGBToYCoCg(vec3 color) {
    return _SRGB_TO_YCOCG * color;
}

// [ROS18]
const mat3 _YCOCG_TO_SRGB = mat3(
    1.0, 1.0, 1.0,
    1.0, 0.0, -1.0,
    -1.0, 1.0, -1.0
);

// [ROS18]
vec3 colors_YCoCgToSRGB(vec3 color) {
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

uint colors_CIEXYZToFP16Luv(vec3 xyz) {
    float uvDiv = safeRcp(dot(xyz, _COLORS_UV_DIV));
    vec2 uv = xyz.xy * uvDiv * _COLORS_UV_MUL;
    uv = clamp(uv, vec2(0.0), vec2(255.0));
    uint result = packHalf2x16(vec2(0.0, xyz.y)) & 0xFFFF0000u;
    result = bitfieldInsert(result, packUnorm4x8(vec4(uv, 0.0, 0.0)), 0, 16);
    return result;
}

vec3 colors_FP16LuvToCIEXYZ(uint luv) {
    vec2 uv = unpackUnorm4x8(luv).xy * _COLORS_UV_INV_MUL;
    float Y = unpackHalf2x16(luv).y;
    float rcp4VTimeY = safeRcp(uv.y * 4.0) * Y;
    vec3 xyz;
    xyz.x = 9.0 * uv.x * rcp4VTimeY;
    xyz.y = Y;
    xyz.z = (12.0 + dot(_COLORS_UV_Z_MUL, uv)) * rcp4VTimeY;
    return xyz;
}

uint colors_workingColorToFP16Luv(vec3 color) {
    vec3 xyz = colors2_colorspaces_convert(COLORS2_WORKING_COLORSPACE, COLORS2_COLORSPACES_CIE_XYZ, color);
    return colors_CIEXYZToFP16Luv(xyz);
}

vec3 colors_FP16LuvToWorkingColor(uint luv) {
    vec3 xyz = colors_FP16LuvToCIEXYZ(luv);
    return colors2_colorspaces_convert(COLORS2_COLORSPACES_CIE_XYZ, COLORS2_WORKING_COLORSPACE, xyz);
}


// -------------------------------------------------- Misc functions --------------------------------------------------
// [KAR13]
float colors_karisWeight(vec3 color) {
    float luma = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, color.rgb);
    return 1.0 / (1.0 + luma);
}

// [LOT16]
vec3 colors_reversibleTonemap(vec3 color) {
    color = color * rcp(mmax3(color) + 1.0);
    return color;
}

vec3 colors_reversibleTonemapWeighted(vec3 color, float weight) {
    return color * (weight * rcp(mmax3(color) + 1.0));
}

vec3 colors_reversibleTonemapInvert(vec3 color) {
    color = color / (1.0 - mmax3(color));
    return color;
}

#endif