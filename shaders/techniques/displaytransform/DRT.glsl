/*
    References:
        [SOB22] Sobotka, Troy. "AgX". 2022.
            https://sobotka.github.io/AgX
        [WRE23] Wrensch, Benjamin. "Minimal AgX Implementation". IOLITE Development Blog. 2023.
            MIT License. Copyright (c) 2024 Missing Deadlines (Benjamin Wrensch)
            https://iolite-engine.com/blog_posts/minimal_agx_implementation
        [LIN24] linlin, "AgX". 2024.
            MIT License. Copyright (c) 2024 linlin
            https://github.com/bWFuanVzYWth/AgX

        You can find full license texts in /licenses

    Other Credits:
        - GeforceLegend - Optimized AgX curve function (https://github.com/GeForceLegend)
*/

#include "/util/Colors2.glsl"
#include "HSLMixer.glsl"

// All values used to derive this implementation are sourced from Troy’s initial AgX implementation/OCIO config file available here:
//   https://github.com/sobotka/AgX
vec3 agxDefaultContrastApprox(vec3 x) {
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

vec3 agx(vec3 val) {
    const mat3 agx_mat = mat3(
        0.842479062253094, 0.0423282422610123, 0.0423756549057051,
        0.0784335999999992, 0.878468636469772, 0.0784336,
        0.0792237451477643, 0.0791661274605434, 0.879142973793104
    );

    // Input transform
    val = agx_mat * val;

    // Log2 space encoding
    const float EV_RANGE = SETTING_TONE_MAPPING_DYNAMIC_RANGE;
    const float EV_RANGE_HALF = EV_RANGE * 0.5;
    val = clamp(log2(val), -EV_RANGE_HALF, EV_RANGE_HALF);
    val = (val + EV_RANGE_HALF) / EV_RANGE;

    // Apply sigmoid function approximation
    val = agxDefaultContrastApprox(val);

    return val;
}

vec3 agxEotf(vec3 val) {
    const mat3 agx_mat_inv = mat3(
        1.19687900512017, -0.0528968517574562, -0.0529716355144438,
        -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
        -0.0990297440797205, -0.0989611768448433, 1.15107367264116
    );

    // Inverse input transform (outset)
    val = agx_mat_inv * val;

    // sRGB IEC 61966-2-1 2.2 Exponent Reference EOTF Display
    // NOTE: We're linearizing the output here. Comment/adjust when
    // *not* using a sRGB render target
    val = colors2_eotf(COLORS2_TF_SRGB, val);

    return val;
}

vec3 agxLook(vec3 val) {
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
    val = pow(max(val * slope + offset, 0.0), power);

    float luma = colors2_colorspaces_luma(COLORS2_DRT_WORKING_COLORSPACE, val);

    return luma + sat * (val - luma);
}

#ifndef COLOR_CALIBRATION_INCLUDED
#define COLOR_CALIBRATION_INCLUDED

// Adobe Camera Raw's Calibration panel rotates each primary's CIE xy
// chromaticity around the working white point (Hue) and scales its distance
// from that white point (Saturation), then rebuilds the RGB->XYZ matrix from
// the shifted primaries. This mirrors the DNG SDK's dng_color_spec model
// (Adobe DNG SDK, dng_color_spec.cpp) rather than a naive RGB hue rotation,
// which cannot reproduce the panel's actual behavior.
//
// Rebuilding the matrix (instead of just projecting each rotated primary
// back to sRGB and blending by the pixel's own r/g/b) is required to keep
// the matrix mapping (1,1,1) to the white point exactly: without that
// constraint, neutrals (grays, whites) drift with the same hue rotation as
// saturated pixels, tinting the whole image instead of isolating the reds,
// greens and blues like ACR does.
vec2 calib_XYZ2xy(vec3 XYZ) {
    float sum = max(XYZ.x + XYZ.y + XYZ.z, 1.0e-6);
    return XYZ.xy / sum;
}

vec3 calib_xy2XYZ(vec2 xy) {
    float y = max(xy.y, 1.0e-6);
    return vec3(xy.x / y, 1.0, (1.0 - xy.x - xy.y) / y);
}

// Done in sRGB rather than the internal (much wider, far more saturated)
// working colorspace: rotating a primary that already sits deep in a huge
// gamut turns a modest-looking slider value into a near-total hue swap,
// since its distance from white is so much larger than a real camera's.
vec2 calib_rotatePrimaryXy(vec2 primaryXy, vec2 whiteXy, float hueDeg, float satMult) {
    vec2 fromWhite = primaryXy - whiteXy;

    float angle = radians(hueDeg);
    float cosA = cos(angle);
    float sinA = sin(angle);
    fromWhite = vec2(fromWhite.x * cosA - fromWhite.y * sinA, fromWhite.x * sinA + fromWhite.y * cosA) * satMult;

    return whiteXy + fromWhite;
}

vec3 applyColorCalibration(vec3 color) {
    vec3 srgbColor = colors2_colorspaces_convert(COLORS2_WORKING_COLORSPACE, COLORS2_COLORSPACES_SRGB, color);

    vec3 whiteXYZ = colors2_colorspaces_convert(COLORS2_COLORSPACES_SRGB, COLORS2_COLORSPACES_CIE_XYZ, vec3(1.0));
    vec2 whiteXy = calib_XYZ2xy(whiteXYZ);
    vec2 redXy = calib_XYZ2xy(colors2_colorspaces_convert(COLORS2_COLORSPACES_SRGB, COLORS2_COLORSPACES_CIE_XYZ, vec3(1.0, 0.0, 0.0)));
    vec2 greenXy = calib_XYZ2xy(colors2_colorspaces_convert(COLORS2_COLORSPACES_SRGB, COLORS2_COLORSPACES_CIE_XYZ, vec3(0.0, 1.0, 0.0)));
    vec2 blueXy = calib_XYZ2xy(colors2_colorspaces_convert(COLORS2_COLORSPACES_SRGB, COLORS2_COLORSPACES_CIE_XYZ, vec3(0.0, 0.0, 1.0)));

    // Sliders use ACR's own -100..100 convention. ACR's own range only ever
    // nudges hue a small amount (its 100 is nowhere near a literal 100 degree
    // spin), so the reading is rescaled down to a realistic rotation.
    // Saturation is a percent offset from the primary's natural distance
    // from white (0 = unchanged, -100 = fully desaturated, 100 = doubled).
    const float CALIB_HUE_RANGE_DEG = 25.0;
    vec3 rXYZ = calib_xy2XYZ(calib_rotatePrimaryXy(redXy, whiteXy, RED_HUE / 100.0 * CALIB_HUE_RANGE_DEG, 1.0 + RED_SAT / 100.0));
    vec3 gXYZ = calib_xy2XYZ(calib_rotatePrimaryXy(greenXy, whiteXy, GREEN_HUE / 100.0 * CALIB_HUE_RANGE_DEG, 1.0 + GREEN_SAT / 100.0));
    vec3 bXYZ = calib_xy2XYZ(calib_rotatePrimaryXy(blueXy, whiteXy, BLUE_HUE / 100.0 * CALIB_HUE_RANGE_DEG, 1.0 + BLUE_SAT / 100.0));

    // Rescale the shifted primaries so the matrix still maps white (1,1,1)
    // to the correct XYZ white point, exactly like the DNG SDK's matrix
    // build (dng_color_spec::NewColorMatrix using the shifted primaries and
    // the same calibration illuminant white).
    mat3 primariesToXYZ = mat3(rXYZ, gXYZ, bXYZ);
    vec3 primaryScale = inverse(primariesToXYZ) * whiteXYZ;
    mat3 calibMatrix = mat3(rXYZ * primaryScale.x, gXYZ * primaryScale.y, bXYZ * primaryScale.z);

    vec3 calibratedXYZ = calibMatrix * srgbColor;
    vec3 calibrated = colors2_colorspaces_convert(COLORS2_COLORSPACES_CIE_XYZ, COLORS2_COLORSPACES_SRGB, calibratedXYZ);
    return colors2_colorspaces_convert(COLORS2_COLORSPACES_SRGB, COLORS2_WORKING_COLORSPACE, calibrated);
}

vec3 _displaytransform_DRT_AgX(vec3 color) {
    color = agx(color);
    color = agxLook(color);
    color = agxEotf(color);
    return color;
}

void _displaytransform_DRT_apply(inout vec4 color) {
    color.rgb = max(color.rgb, 0.0);

    #if SETTING_TONE_MAPPING_LOOK == 3
    // Camera Raw Color Calibration only makes sense paired with the Custom look
    color.rgb = applyColorCalibration(color.rgb);
    color.rgb = max(color.rgb, 0.0); // Ensure no negative values after shift
    #endif

    color.rgb = colors2_colorspaces_convert(COLORS2_WORKING_COLORSPACE, COLORS2_DRT_WORKING_COLORSPACE, color.rgb);
    color.rgb = max(color.rgb, 0.0);
    color.rgb = _displaytransform_DRT_AgX(color.rgb);
    color.rgb = colors2_colorspaces_convert(COLORS2_DRT_WORKING_COLORSPACE, COLORS2_OUTPUT_COLORSPACE, color.rgb);
    color.rgb = saturate(color.rgb);

    #if SETTING_TONE_MAPPING_LOOK == 3
    // Runs on the final display-referred [0,1] color, like a raster editor's HSL panel would
    color.rgb = applyHSLMixer(color.rgb);
    color.rgb = saturate(color.rgb);
    #endif

    color.rgb = colors2_oetf(COLORS2_OUTPUT_TF, color.rgb);
    color.rgb = saturate(color.rgb);
}

#endif // COLOR_CALIBRATION_INCLUDED
