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
        [ROS18] Rosseaux, Benjamin. "Matrix-based RGB from/to YCoCg color space conversion". 2018.
            CC0 License (Public Domain).
            https://www.shadertoy.com/view/4dXGzN
        [WIK25a] Wikipedia. "Rec. 601". 2025.
            https://en.wikipedia.org/wiki/Rec._601
        [WIK25b] Wikipedia. "Rec. 709". 2025.
            https://en.wikipedia.org/wiki/Rec._709
        [WIK25c] Wikipedia. "sRGB". 2025.
            https://en.wikipedia.org/wiki/SRGB

*/
#ifndef INCLUDE_util_Colors_glsl
#define INCLUDE_util_Colors_glsl a
#include "/_Base.glsl"
#include "Math.glsl"

// ----------------------------------------------------- Rec. 601 -----------------------------------------------------
// [WIK25a]
vec3 colors_Rec601_encodeGamma(vec3 color) {
    vec3 lower = 4.5 * color;
    vec3 higher = pow(color, vec3(0.45)) * 1.099 - 0.099;
    return mix(lower, higher, vec3(greaterThanEqual(color, vec3(0.018))));
}

// [WIK25a]
vec3 colors_Rec601_decodeGamma(vec3 color) {
    vec3 lower = color / 4.5;
    vec3 higher = pow((color + 0.099) / 1.099, vec3(1.0 / 0.45));
    return mix(lower, higher, vec3(greaterThanEqual(color, vec3(0.081))));
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
vec3 colors_sRGB_encodeGamma(vec3 color) {
    vec3 lower = 12.92 * color;
    vec3 higher = pow(color, vec3(1.0 / 2.4)) * 1.055 - 0.055;
    return mix(lower, higher, vec3(greaterThan(color, vec3(0.0031308))));
}

// [WIK25c]
float color_sRGB_decodeGamma(float color) {
    float lower = color / 12.92;
    float higher = pow((color + 0.055) / 1.055, 2.4);
    return mix(lower, higher, color > 0.04045);
}

// [WIK25c]
vec3 colors_sRGB_decodeGamma(vec3 color) {
    vec3 lower = color / 12.92;
    vec3 higher = pow((color + 0.055) / 1.055, vec3(2.4));
    return mix(lower, higher, vec3(greaterThan(color, vec3(0.04045))));
}

// [WIK25c]
float colors_sRGB_luma(vec3 color) {
    return colors_Rec709_luma(color);
}

// Simple approximation of sRGB to linear conversion
vec3 colors_srgbToLinear(vec3 color) {
    const float a0 = 0.000570846;
    const float a1 = -0.0403863;
    const float a2 = 0.862127;
    const float a3 = 0.178572;
    vec3 x = max(color, 0.0232545);
    vec3 x2 = x * x;
    vec3 x3 = x2 * x;
    return a0 + a1 * x + a2 * x2 + a3 * x3;
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


// -------------------------------------------------- Misc functions --------------------------------------------------
// [KAR13]
float colors_karisWeight(vec3 color) {
    float luma = colors_sRGB_luma(color.rgb);
    return 1.0 / (1.0 + luma);
}

#endif