// One workgroup accumulates one weighted histogram before merging with the frame.
shared uint meterBins[256];
const float METER_MIN_LOG = -24.0;
const float METER_BIN_WIDTH = 40.0 / 256.0;

void meterScene(ivec2 pixel, vec3 scene) {
    float Y = colors_colorspaces_luma(COLORS_WORKING_COLORSPACE, scene);
    if (isnan(Y) || isinf(Y)) return;
    float logY = log2(max(Y, exp2(METER_MIN_LOG)));
    uint bin = uint(clamp((logY - METER_MIN_LOG) / METER_BIN_WIDTH, 0.0, 255.0));

    vec2 weights = transient_exposureWeights_fetch(pixel).xy;
    vec2 ndc = (vec2(pixel) + 0.5) * uval_mainImageSizeRcp * 2.0 - 1.0;
    float center = pow(max(1.0 - length(ndc), 0.0), SETTING_EXPOSURE_CENTER_WEIGHTING_CURVE);
    float w = weights.x * exp2(weights.y * SETTING_EXPOSURE_EMISSIVE_WEIGHTING);
    w *= 1.0 + SETTING_EXPOSURE_CENTER_WEIGHTING * center;

    // Bound even a one-bin full-screen histogram at the configured resolution.
    uint maxUnits = 0xffffffffu / (uint(uval_mainImageSizeI.x) * uint(uval_mainImageSizeI.y));
    float maxWeight = (1.0 + SETTING_EXPOSURE_CENTER_WEIGHTING) * exp2(max(0.0, SETTING_EXPOSURE_EMISSIVE_WEIGHTING));
    uint units = uint(round(w * min(256.0, float(maxUnits - 1u) / maxWeight)));
    atomicAdd(meterBins[bin], units);
}
