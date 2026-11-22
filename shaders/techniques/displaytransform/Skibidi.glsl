#include "SkibidiGeometry.glsl"
vec3 _sk_eo13(vec3 rgb) {
    if (rgb.r == rgb.g && rgb.g == rgb.b)
        return rgb;
    float y = clamp(dot(rgb, _SKM_DISPLAY_Y), 0.0, 1.0);
    vec3 delta = rgb - vec3(y);
    vec2 xy = skPlane(delta);
    float hue = fract(atan(xy.y, xy.x) / (2.0 * _SK_PI) + 1.0);
    float low = _sk_div(y - min(rgb.r, min(rgb.g, rgb.b)), y),
          high = _sk_div(max(rgb.r, max(rgb.g, rgb.b)) - y, 1.0 - y);
    float rho = clamp(max(low, high), 0.0, 1.0);
    float outerRadius = targetIsRec709() ? _sk_atlas(_SKM_OUTER_FIELD.xy + _SKM_OUTER_FIELD.zw * vec2(sqrt(y), hue))
                                           : skGeometry(vec2(hue * 96.0, 33.0 + sqrt(y) * 47.0));
    float scale = outerRadius / (outerRadius + (1.0 - outerRadius) * rho);
    // Every EO13 step rescales the same constant-Y ray. Its hue fractions
    // and its cusp luminance remain fixed throughout the allocation.
    float minimum = min(rgb.r, min(rgb.g, rgb.b));
    float width = max(rgb.r, max(rgb.g, rgb.b)) - minimum;
    float redBlue = max(rgb.r - rgb.b, 0.0), greenBlue = max(rgb.g - rgb.b, 0.0);
    float redFraction = _sk_div(redBlue, redBlue + greenBlue);
    float warm = 4.0 * redFraction * (1.0 - redFraction);
    scale += .85 * warm * (1.0 - scale);
    float occupancy = clamp(low * high * scale * scale, 0.0, 1.0);
    float cuspY = clamp(_sk_div(y - minimum, width), 0.0, 1.0);
    float halfWidth = .65 * min(cuspY, 1.0 - cuspY);
    float t = clamp(_sk_div(y - cuspY + halfWidth, 2.0 * halfWidth), 0.0, 1.0);
    float below = 1.0 - t * t * (3.0 - 2.0 * t);
    float greenRed = _sk_div(max(rgb.g - rgb.r, 0.0), width);
    float greenFraction = _sk_div(rgb.g - minimum, width);
    float greenWeight =
        greenRed * greenRed * greenFraction * greenFraction * (.55 + .45 * _sk_div(greenBlue, width)) * below;
    float roundScale = pow(1.0 + 5.0 * greenWeight * pow(occupancy, 3.0), -1.0 / 6.0);
    float redGreen = max(rgb.r - rgb.g, 0.0);
    float dark = _sk_div(redGreen, width) * pow(1.0 - warm, 4.0) * (1.0 - y) * (1.0 - y);
    scale *= roundScale * inversesqrt(1.0 + .40 * dark * occupancy * roundScale * roundScale);
    vec2 gaps = scale * vec2(redGreen, redBlue);
    vec2 reserve = gaps * gaps / (gaps + .01);
    scale *= 1.0 - .10 * (1.0 - warm) * (1.0 - warm) * reserve.x * reserve.y;
    return vec3(y) + scale * delta;
}
// Shift bright orange toward yellow at constant display Y (the EO18 sunset halo).
vec3 _sk_eo18(vec3 rgb) {
    float highlight = pow(max(2.0 * dot(rgb, _SKM_DISPLAY_Y) - 1.0, 0.0), 2.0);
    vec2 gaps = pow(max(rgb.rg - rgb.b, vec2(0.0)), vec2(4.0));
    vec2 fraction = gaps / (gaps + pow(.22, 4.0));
    float drive = 4.0 * highlight * (1.0 - rgb.g) * fraction.x * fraction.y;
    float removed = rgb.b * drive / (1.0 + drive);
    return rgb + vec3(0.0, (_SKM_DISPLAY_Y.b / _SKM_DISPLAY_Y.g) * removed, -removed);
}
// Source EV offsets in signed Rec.2020. EO17 takes the midpoint of the two red exposures.
float _sk_sourceOffset(vec3 raw, float magnitude) {
    float green = max(raw.g, 0.0), blue = max(raw.b, 0.0), negative = max(-raw.r, 0.0);
    float excess = pow(max(green - raw.r, 0.0), 2.0),
          energy = excess + pow(max(raw.r, 0.0), 2.0) + blue * blue;
    float weight = _sk_div(excess, energy), outside = _sk_div(green * negative, green + negative);
    float x = length(vec2(max(green - blue, 0.0), 3.0 * outside)) * (.25 * magnitude);
    float t = min(x, 1.0 / max(x, 1.0));
    float window = 2.0 * t / (1.0 + t * t);
    float offset = -.8 * weight * window * window;
    vec3 positive = max(raw, 0.0);
    excess = pow(max(positive.r - max(positive.g, positive.b), 0.0), 2.0);
    energy = excess + 144.0 * dot(positive.gb, positive.gb);
    x = sqrt(positive.r * (magnitude / 32.0));
    t = min(x, 1.0 / max(x, 1.0));
    offset -= 1.6 * _sk_div(excess, energy) * t / ((1.0 + t) * (1.0 + t));
    return offset;
}
#ifdef AP_HDR_OUTPUT
// Both arguments use linear Rec.2020/D65. raw is normalized scene RGB;
// luminance is SI5 display Y divided by peak. A common lift keeps the
// differences between signed channels before the radial highlight shoulder.
vec3 _sk_hdrSourceChroma(vec3 raw, vec3 repaired, float magnitude, float luminance) {
    float high = max(raw.r, max(raw.g, raw.b));
    float low = min(raw.r, min(raw.g, raw.b));
    float width = 0.02 * (high - low);
    vec3 aboveLow = raw - vec3(low);
    float floorValue = 0.5 * (length(vec2(low, width)) + low);
    if (low < 0.0) {
        // Source chromaticity loses its physical interpretation as signed Y
        // disappears. Use SI5's positive extension continuously in that limit.
        float positiveY = dot(max(raw, vec3(0.0)), _SKM_DISPLAY_Y);
        float weight = smoothstep(0.0, 1.0, _sk_div(dot(raw, _SKM_DISPLAY_Y), positiveY));
        vec3 source = (1.0 - weight) * repaired + weight * (aboveLow + vec3(floorValue));
        floorValue = min(source.r, min(source.g, source.b));
        aboveLow = source - vec3(floorValue);
    }
    float meanAboveLow = dot(aboveLow, _SKM_DISPLAY_Y);
    float sourceY = meanAboveLow + floorValue;
    // Use one scene-referred chroma budget for all HDR display peaks. The
    // fixed 203/1000 ratio is the reference viewing scale for this shoulder.
    float budget = 1.0 / (1.0 + magnitude * (sourceY * (203.0 / 1000.0)));
    // Reserve the same fraction of available headroom along a given hue.
    // At fixed lifted source Y and display Y, intermediate chromas get
    // their own fraction of that hue's allowable peak excursion.
    float span = max(aboveLow.r, max(aboveLow.g, aboveLow.b));
    float endRatio = _sk_div(max(span - meanAboveLow, 0.0), meanAboveLow);
    float room = max(1.0 - luminance, 0.0);
    float scale = min(1.0, _sk_div(budget * room, length(vec2(room, budget * luminance * endRatio))));
    float gain = _sk_div(luminance * scale, sourceY);
    // The convex form keeps dim signed emitters nonnegative at PQ's input.
    return vec3(luminance * (1.0 - scale)) + gain * (aboveLow + vec3(floorValue));
}
#endif
#ifdef AP_HDR_OUTPUT
float _sk_shoulderScale(float value, float knee) {
    if (value <= knee) return 1.0;
    float excess = value - knee;
    return (knee + (1.0 - knee) * excess / (1.0 - knee + excess)) / value;
}

float _sk_uvRadius(vec3 rgb, float denominator) {
    // Removing gray analytically keeps the two D65 offsets stable near white.
    vec3 delta = rgb - vec3(rgb.g);
    return _sk_div(length(vec2(dot(delta, _SKM_U_OFFSET), dot(delta, _SKM_V_OFFSET))), denominator);
}

vec3 _sk_hdrSdrStyle(vec3 sdr, vec3 raw, vec3 repaired, float magnitude) {
    float s = clamp(dot(sdr, _SKM_DISPLAY_Y), 0.0, 1.0);
    if (s <= 0.0) return vec3(0.0);
    if (s >= 1.0) return vec3(1.0);
    float white = 1.0 / hdrPeakRatio();
    float gray = .18 * white, exponent = .82 / (1.0 - gray);
    float light = pow(s, exponent);
    float dark = (1.0 - gray) / gray * pow((.18 / .82) * (1.0 - s), exponent);
    float y = mix(white * s, light / (light + dark), smoothstep(.18, 1.0, s));
    if (y >= 1.0) return vec3(1.0);

    vec3 anchor = (sdr / s) * y;
    mat3 toTarget = inverse(targetGamutToXYZ()) * rec2020ToXYZ();
    vec3 targetRaw = toTarget * raw;
    vec3 wide = _sk_hdrSourceChroma(targetRaw, max(toTarget * repaired, 0.0), magnitude, y);
    vec3 positive = max(raw, vec3(0.0)), in709 = _SKM_2020_TO_SRGB * positive;
    float outside = max(-min(in709.r, min(in709.g, in709.b)), 0.0);
    float outsideFraction = _sk_div(outside, dot(positive, _SKM_SOURCE_Y) + outside);
    float purity = clamp((s - min(sdr.r, min(sdr.g, sdr.b))) / s, 0.0, 1.0);
    float amount = mix(.25 * (1.0 - white) * purity, 1.0, outsideFraction);
    vec3 target = mix(anchor, wide, amount);

    float ceiling = _sk_uvRadius(positive, dot(positive, _SKM_UV_DEN));
    if (ceiling <= 1e-12) return vec3(y);
    vec3 target2020 = (inverse(rec2020ToXYZ()) * targetGamutToXYZ()) * target;
    float den = dot(target2020, _SKM_UV_DEN);
    float fraction = _sk_shoulderScale(_sk_uvRadius(target2020, den) / ceiling, .95);
    // u'v' is projective. Convert its radial scale to an RGB ray scale at fixed Y.
    float neutralDen = y * dot(vec3(1.0), _SKM_UV_DEN);
    float scale = _sk_div(fraction * neutralDen, mix(den, neutralDen, fraction));
    float excursion = max(max(target.r, target.g), target.b) - y;
    scale *= _sk_shoulderScale(_sk_div(scale * max(excursion, 0.0), 1.0 - y), .8);
    return mix(vec3(y), target, clamp(scale, 0.0, 1.0));
}
#endif

vec3 _sk_runtime(vec3 raw, float magnitude) {
    vec3 positive = max(raw, 0.0);
    float rms = sqrt(dot(positive * positive, _SKM_SOURCE_Y));
    if (rms <= 0.0)
        return vec3(0.0);
    float exposurePower = exp2(.42 * _sk_sourceOffset(raw, magnitude));
    float epsilon = .02 * rms;
    vec3 hyp = sqrt(raw * raw + epsilon * epsilon), repaired;
    for (int c = 0; c < 3; ++c)
        repaired[c] = raw[c] >= 0.0 ? .5 * (raw[c] + hyp[c]) : epsilon * epsilon / (2.0 * (hyp[c] - raw[c]));
    repaired /= 1.0000999900019996;
    if (raw.r == raw.g && raw.g == raw.b && raw.r >= 0.0)
        repaired = raw;
    vec3 rgb = _sk_si5(repaired, exposurePower * pow(_SKM_FL * magnitude, .42));
#ifdef AP_HDR_OUTPUT
    rgb = _sk_hdrSdrStyle(_sk_eo18(_sk_eo13(rgb)), raw, repaired, magnitude);
    return (inverse(rec709ToXYZ()) * targetGamutToXYZ() * rgb) * hdrPeakRatio();
#else
    rgb = _sk_eo18(_sk_eo13(rgb));
    return inverse(rec709ToXYZ()) * targetGamutToXYZ() * clamp(rgb, 0.0, 1.0);
#endif
}
vec3 drt_skibidi_ap0(vec3 ap0) {
    ap0 = mix(ap0, sign(ap0) * 65504.0, isinf(ap0));
    ap0 = mix(ap0, vec3(0.0), isnan(ap0));
    float magnitude = max(abs(ap0.r), max(abs(ap0.g), abs(ap0.b)));
    if (magnitude <= 0.0)
        return vec3(0.0);
    vec3 normalized = ap0 / magnitude;
    vec3 raw = vec3(normalized.g) +
               _SKM_AP0_TO_2020 * vec3(normalized.r - normalized.g, 0.0, normalized.b - normalized.g);
    return _sk_runtime(raw, magnitude);
}
