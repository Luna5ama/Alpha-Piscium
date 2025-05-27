#ifndef INCLUDE_util_Colors_glsl
#define INCLUDE_util_Colors_glsl a
#include "/_Base.glsl"
#include "Math.glsl"

float colors_srgbLuma(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

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

float colors_karisWeight(vec3 color) {
    float luma = colors_srgbLuma(color.rgb);
    return 1.0 / (1.0 + luma);
}

const mat3 _SRGB_TO_YCOCG = mat3(
    0.25, 0.5, -0.25,
    0.5, 0.0, 0.5,
    0.25, -0.5, -0.25
);
vec3 colors_SRGBToYCoCg(vec3 color) {
    return _SRGB_TO_YCOCG * color;
}

const mat3 _YCOCG_TO_SRGB = mat3(
    1.0, 1.0, 1.0,
    1.0, 0.0, -1.0,
    -1.0, 1.0, -1.0
);

vec3 colors_YCoCgToSRGB(vec3 color) {
    return _YCOCG_TO_SRGB * color;
}

// https://graphicrants.blogspot.com/2009/04/rgbm-color-encoding.html
// M matrix, for encoding
const mat3 M = (mat3(
    0.2209, 0.3390, 0.4184,
    0.1138, 0.6780, 0.7319,
    0.0102, 0.1130, 0.2969
));

// Inverse M matrix, for decoding
const mat3 InverseM = (mat3(
    6.0014, -2.7008, -1.7996,
    -1.3320, 3.1029, -5.7721,
    0.3008, -1.0882, 5.6268
));

vec4 colors_SRGBToLogLuv(in vec3 vRGB)  {
    vec4 vResult;
    vec3 Xp_Y_XYZp = M * vRGB;
    Xp_Y_XYZp = max(Xp_Y_XYZp, vec3(1e-6, 1e-6, 1e-6));
    vResult.xy = Xp_Y_XYZp.xy / Xp_Y_XYZp.z;
    float Le = 2 * log2(Xp_Y_XYZp.y) + 127;
    vResult.w = fract(Le);
    vResult.z = (Le - (floor(vResult.w * 255.0f)) / 255.0f) / 255.0f;
    return vResult;
}

vec3 colors_LogLuvToSRGB(in vec4 vLogLuv) {
    float Le = vLogLuv.z * 255 + vLogLuv.w;
    vec3 Xp_Y_XYZp;
    Xp_Y_XYZp.y = exp2((Le - 127) / 2);
    Xp_Y_XYZp.z = Xp_Y_XYZp.y / vLogLuv.y;
    Xp_Y_XYZp.x = vLogLuv.x * Xp_Y_XYZp.z;
    vec3 vRGB = InverseM * Xp_Y_XYZp;
    return max(vRGB, 0);
}

#endif