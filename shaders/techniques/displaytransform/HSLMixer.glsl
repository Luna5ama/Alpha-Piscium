/*
    8-band HSL color mixer, modeled after Lightroom/Camera Raw's HSL/Color
    panel: red, orange, yellow, green, aqua, blue, magenta and pink each get
    an independent hue, saturation and luminance control. Unlike the
    primaries Calibration panel (DRT.glsl), which rebuilds a global RGB->XYZ
    matrix, this tool is explicitly selective: it must only touch pixels
    whose hue actually falls in (or near) a given band, leaving every other
    hue alone - including the Hue slider itself, which (like Calibration's
    primary Hue rotation) only rotates the hue of pixels weighted into that
    band instead of spinning the whole image's hue wheel.

    Applied to the final display-referred linear color (after the AgX look
    and the DRT->output colorspace conversion, before the output OETF), the
    same stage a raster editor's HSL panel would operate on, rather than to
    scene-linear HDR data where "hue" and "lightness" are less meaningful.
*/

#ifndef HSL_MIXER_INCLUDED
#define HSL_MIXER_INCLUDED

float hslmixer_hueToRgb(float p, float q, float t) {
    if (t < 0.0) t += 1.0;
    if (t > 1.0) t -= 1.0;
    if (t < 1.0 / 6.0) return p + (q - p) * 6.0 * t;
    if (t < 0.5) return q;
    if (t < 2.0 / 3.0) return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    return p;
}

vec3 hslmixer_hsl2rgb(vec3 hsl) {
    float h = hsl.x / 360.0;
    float s = hsl.y;
    float l = hsl.z;
    if (s < 1.0e-6) return vec3(l);

    float q = l < 0.5 ? l * (1.0 + s) : l + s - l * s;
    float p = 2.0 * l - q;
    return vec3(
        hslmixer_hueToRgb(p, q, h + 1.0 / 3.0),
        hslmixer_hueToRgb(p, q, h),
        hslmixer_hueToRgb(p, q, h - 1.0 / 3.0)
    );
}

vec3 hslmixer_rgb2hsl(vec3 c) {
    float maxC = max(c.r, max(c.g, c.b));
    float minC = min(c.r, min(c.g, c.b));
    float delta = maxC - minC;

    float l = (maxC + minC) * 0.5;
    float s = delta < 1.0e-6 ? 0.0 : delta / (1.0 - abs(2.0 * l - 1.0) + 1.0e-6);

    float h;
    if (delta < 1.0e-6) {
        h = 0.0;
    } else if (maxC == c.r) {
        h = mod((c.g - c.b) / delta, 6.0);
    } else if (maxC == c.g) {
        h = (c.b - c.r) / delta + 2.0;
    } else {
        h = (c.r - c.g) / delta + 4.0;
    }
    h *= 60.0;
    if (h < 0.0) h += 360.0;

    return vec3(h, s, l);
}

// 8 hue anchors spaced around the wheel like Lightroom's HSL panel:
// Red 0, Orange 30, Yellow 60, Green 120, Aqua 180, Blue 240, Magenta 300, Pink 330.
// For a given hue, exactly the two anchors bracketing it get a nonzero,
// linearly-interpolated weight (a partition of unity) - every other band is
// left at exactly 0, so a band's slider can only ever move pixels whose hue
// is in or adjacent to that band.
void hslmixer_bandWeights(
    float hueDeg,
    out float wRed, out float wOrange, out float wYellow, out float wGreen,
    out float wAqua, out float wBlue, out float wMagenta, out float wPink
) {
    wRed = 0.0; wOrange = 0.0; wYellow = 0.0; wGreen = 0.0;
    wAqua = 0.0; wBlue = 0.0; wMagenta = 0.0; wPink = 0.0;

    if (hueDeg < 30.0) {
        float t = hueDeg / 30.0;
        wRed = 1.0 - t; wOrange = t;
    } else if (hueDeg < 60.0) {
        float t = (hueDeg - 30.0) / 30.0;
        wOrange = 1.0 - t; wYellow = t;
    } else if (hueDeg < 120.0) {
        float t = (hueDeg - 60.0) / 60.0;
        wYellow = 1.0 - t; wGreen = t;
    } else if (hueDeg < 180.0) {
        float t = (hueDeg - 120.0) / 60.0;
        wGreen = 1.0 - t; wAqua = t;
    } else if (hueDeg < 240.0) {
        float t = (hueDeg - 180.0) / 60.0;
        wAqua = 1.0 - t; wBlue = t;
    } else if (hueDeg < 300.0) {
        float t = (hueDeg - 240.0) / 60.0;
        wBlue = 1.0 - t; wMagenta = t;
    } else if (hueDeg < 330.0) {
        float t = (hueDeg - 300.0) / 30.0;
        wMagenta = 1.0 - t; wPink = t;
    } else {
        float t = (hueDeg - 330.0) / 30.0;
        wPink = 1.0 - t; wRed = t;
    }
}

vec3 applyHSLMixer(vec3 color) {
    vec3 hsl = hslmixer_rgb2hsl(color);

    float wRed, wOrange, wYellow, wGreen, wAqua, wBlue, wMagenta, wPink;
    hslmixer_bandWeights(hsl.x, wRed, wOrange, wYellow, wGreen, wAqua, wBlue, wMagenta, wPink);

    // Hue: -100..100 -> a rotation in degrees, weighted by the very same
    // band membership used for Saturation/Luminance below, so a band's Hue
    // slider can only ever drag around pixels that already have nonzero
    // weight in that band (and its two immediate neighbors) - it can never
    // touch a hue that band has no weight over.
    const float HSL_HUE_RANGE_DEG = 60.0;
    float hueShift =
        wRed * HSL_RED_HUE + wOrange * HSL_ORANGE_HUE + wYellow * HSL_YELLOW_HUE + wGreen * HSL_GREEN_HUE +
        wAqua * HSL_AQUA_HUE + wBlue * HSL_BLUE_HUE + wMagenta * HSL_MAGENTA_HUE + wPink * HSL_PINK_HUE;
    hueShift *= HSL_HUE_RANGE_DEG / 100.0;

    // Saturation: same -100..100 -> (1 + x/100) multiplier convention as the
    // primaries Calibration panel. Luminance: -100..100 -> -1..1, applied
    // through a curve that fades to 0 at pure black/white so it can't clip
    // otherwise-untouched hues into invalid lightness.
    float satMult =
        wRed * (1.0 + HSL_RED_SAT / 100.0) + wOrange * (1.0 + HSL_ORANGE_SAT / 100.0) +
        wYellow * (1.0 + HSL_YELLOW_SAT / 100.0) + wGreen * (1.0 + HSL_GREEN_SAT / 100.0) +
        wAqua * (1.0 + HSL_AQUA_SAT / 100.0) + wBlue * (1.0 + HSL_BLUE_SAT / 100.0) +
        wMagenta * (1.0 + HSL_MAGENTA_SAT / 100.0) + wPink * (1.0 + HSL_PINK_SAT / 100.0);

    float lumShift =
        wRed * HSL_RED_LUM + wOrange * HSL_ORANGE_LUM + wYellow * HSL_YELLOW_LUM + wGreen * HSL_GREEN_LUM +
        wAqua * HSL_AQUA_LUM + wBlue * HSL_BLUE_LUM + wMagenta * HSL_MAGENTA_LUM + wPink * HSL_PINK_LUM;
    lumShift *= 0.01;

    float lumCurve = hsl.z * (1.0 - hsl.z) * 4.0; // 0 at L=0/1, peaks at L=0.5

    hsl.x = mod(hsl.x + hueShift, 360.0);
    hsl.y = clamp(hsl.y * satMult, 0.0, 1.0);
    hsl.z = clamp(hsl.z + lumShift * lumCurve * 0.5, 0.0, 1.0);

    return hslmixer_hsl2rgb(hsl);
}

#endif // HSL_MIXER_INCLUDED
