#include "/util/Colors.glsl"
#include "TargetGamut.glsl"
#include "PrimaryColorCalibration.glsl"
#include "HSLColorMixer.glsl"

#if defined(HDR_MOD_INSTALLED) && defined(HDR_ENABLED)
#define AP_HDR_OUTPUT
uniform float HdrGamePaperWhiteBrightness;
uniform float HdrGamePeakBrightness;
uniform float HdrUIBrightness;

float hdrPeakRatio() {
    return max(HdrGamePeakBrightness / max(HdrGamePaperWhiteBrightness, 1.0), 1.0);
}
#endif

#if SETTING_DRT_TYPE == 0
#include "Skibidi.glsl"
#elif SETTING_DRT_TYPE == 1
#ifdef AP_HDR_OUTPUT
#include "AgXHDR.glsl"
#else
#include "AgXSDR.glsl"
#endif
#elif SETTING_DRT_TYPE == 2
#include "OpenDRT.glsl"
#else
#include "Psychov30.glsl"
#endif

// Linear BT.709/D65. HDR RGB is signed and relative to game paper white.
vec3 displayLinear(vec3 scene) {
    vec3 ap0 = colors_colorspaces_convert(COLORS_WORKING_COLORSPACE, COLORS_COLORSPACES_ACES_AP0, scene);
    #if SETTING_DRT_TYPE == 0
    return drt_skibidi_ap0(ap0);
    #elif SETTING_DRT_TYPE == 1
    vec3 encoded = blenderAgX(vec4(ap0, 1.0)).rgb;
    #ifdef AP_HDR_OUTPUT
    // The Blender formation LUT specifies a 1000-nit peak and 100-nit reference white.
    const float a = 0.17883277, b = 0.28466892, c = 0.55991073;
    vec3 e = mix(encoded * encoded / 3.0, (exp((encoded - c) / a) + b) / 12.0, greaterThan(encoded, vec3(0.5)));
    vec3 rgb = e * pow(max(dot(e, vec3(.2627, .6780, .0593)), 0.0), .2);
    float m = max(rgb.r, max(rgb.g, rgb.b));
    const float referenceWhite = 100.0 / 1000.0;
    float level = m <= referenceWhite ? m / referenceWhite
        : mix(1.0, hdrPeakRatio(), clamp((m - referenceWhite) / (1.0 - referenceWhite), 0.0, 1.0));
    rgb *= level / max(m, 1e-20);
    return colors_colorspaces_convert(COLORS_COLORSPACES_REC2020, COLORS_COLORSPACES_SRGB, rgb);
    #else
    return colors_eotf(COLORS_TF_SRGB, clamp(encoded, 0.0, 1.0));
    #endif
    #elif SETTING_DRT_TYPE == 2
    return drt_openDRT(colors_colorspaces_convert(COLORS_COLORSPACES_ACES_AP0, COLORS_COLORSPACES_CIE_XYZ, ap0));
    #elif defined(AP_HDR_OUTPUT)
    return drt_psychov(ap0, hdrPeakRatio(), uint(!targetIsRec709()));
    #else
    return drt_psychov(ap0, 1.0, uint(samePrimaries(targetGamutToXYZ(), rec2020ToXYZ())));
    #endif
}

vec3 displayLook(vec3 val) {
    #if SETTING_TONE_MAPPING_LOOK == 0
    return val;
    #else
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
    vec3 adjusted = val * slope + offset;
    #ifdef AP_HDR_OUTPUT
    val = sign(adjusted) * pow(abs(adjusted), power);
    #else
    val = pow(max(adjusted, 0.0), power);
    #endif

    #ifdef AP_HDR_OUTPUT
    float luma = colors_colorspaces_luma(COLORS_COLORSPACES_SRGB, val);
    #else
    float luma = colors_colorspaces_luma(COLORS_OUTPUT_COLORSPACE, val);
    #endif

    return luma + sat * (val - luma);
    #endif
}

void _displaytransform_DRT_apply(inout vec4 color) {
    color.rgb = displayLinear(color.rgb);
    #ifdef AP_HDR_OUTPUT
    #if SETTING_TONE_MAPPING_LOOK != 0
    // One maps to the scene paper white. Retain signed wide-gamut channels and highlights above one.
    vec3 encoded = sign(color.rgb) * colors_oetf(COLORS_TF_SRGB, abs(color.rgb));
    encoded = displayLook(encoded);
    color.rgb = sign(encoded) * colors_eotf(COLORS_TF_SRGB, abs(encoded));
    #endif
    vec3 scaled = color.rgb * (HdrGamePaperWhiteBrightness / max(HdrUIBrightness, 1.0));
    color.rgb = sign(scaled) * colors_oetf(COLORS_TF_SRGB, abs(scaled));
    #else
    #if defined(SETTING_PRIMARY_COLOR_CALIBRATION) || defined(SETTING_HCM_COLOR_MIXER)
    {
        color.rgb = colors_colorspaces_convert(COLORS_COLORSPACES_SRGB, COLORS_GRADING_COLORSPACE, color.rgb);

        #ifdef SETTING_PRIMARY_COLOR_CALIBRATION
        color.rgb = displaytransform_primarycolorcalibration_apply(color.rgb);
        #endif

        color.rgb = saturate(color.rgb);
        color.rgb = colors_oetf(COLORS_GRADING_TF, color.rgb);

        #ifdef SETTING_HCM_COLOR_MIXER
        color.rgb = displaytransform_hslcolormixer_apply(color.rgb);
        #endif

        #if COLORS_GRADING_TF != COLORS_OUTPUT_TF || COLORS_GRADING_COLORSPACE != COLORS_OUTPUT_COLORSPACE
        color.rgb = saturate(color.rgb);
        color.rgb = colors_eotf(COLORS_GRADING_TF, color.rgb);
        #if COLORS_GRADING_COLORSPACE != COLORS_OUTPUT_COLORSPACE
        color.rgb = colors_colorspaces_convert(COLORS_GRADING_COLORSPACE, COLORS_OUTPUT_COLORSPACE, color.rgb);
        #endif
        color.rgb = saturate(color.rgb);
        color.rgb = colors_oetf(COLORS_OUTPUT_TF, color.rgb);
        #endif
    }
    #else
    {
        color.rgb = colors_colorspaces_convert(COLORS_COLORSPACES_SRGB, COLORS_OUTPUT_COLORSPACE, color.rgb);
        color.rgb = saturate(color.rgb);
        color.rgb = colors_oetf(COLORS_OUTPUT_TF, color.rgb);
    }
    #endif

    color.rgb = saturate(displayLook(saturate(color.rgb)));
    #endif
}
