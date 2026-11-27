#ifndef INCLUDE_BLOOM
#define INCLUDE_BLOOM

#include "/Base.glsl"

#if defined(BLOOM_DOWNSAMPLE) && BLOOM_PASS == 1 && SETTING_BLOOM_HIGHLIGHT_COMPRESSION != 0
#include "/util/AgxInvertible.glsl"
#if SETTING_BLOOM_HIGHLIGHT_COMPRESSION_MODE == 1
#include "/util/Colors.glsl"
#endif

float bloom_compressHighlight(float value) {
    if (value <= 1.0) return value;
    #if SETTING_BLOOM_HIGHLIGHT_COMPRESSION == 1
    return 1.0 + log(value);
    #elif SETTING_BLOOM_HIGHLIGHT_COMPRESSION == 2
    return 2.0 - exp(-(value - 1.0));
    #elif SETTING_BLOOM_HIGHLIGHT_COMPRESSION == 3
    return 2.0 - 1.0 / value;
    #else
    return 1.0;
    #endif
}

vec3 bloom_compressHighlights(vec3 color) {
    // Apply the original log-space rolloff to Bloom only, after AA.
    vec3 encoded = agxInvertible_forward(color);
    #if SETTING_BLOOM_HIGHLIGHT_COMPRESSION_MODE == 0
    if (all(lessThanEqual(encoded, vec3(1.0)))) return color;
    encoded = vec3(bloom_compressHighlight(encoded.r),
                   bloom_compressHighlight(encoded.g),
                   bloom_compressHighlight(encoded.b));
    #else
    float luma = colors_colorspaces_luma(COLORS_WORKING_COLORSPACE, encoded);
    if (luma <= 1.0) return color;
    encoded *= bloom_compressHighlight(luma) / luma;
    #endif
    return agxInvertible_inverse(encoded);
}
#endif

vec3 bloom_sample(vec2 pos, vec4 tile) {
    // Clamp inside the input mip before converting to atlas coordinates.
    vec2 texel = tile.xy + clamp(pos, vec2(0.5), tile.zw - 0.5);
    vec2 uv = texel * uval_mainImageSizeRcp;
    #if defined(BLOOM_DOWNSAMPLE) || defined(BLOOM_ZERO_EXTEND)
    // Zero-extend the bilinear footprint without sampling an adjacent tile.
    vec2 coverage = clamp(min(pos, tile.zw - pos) + 0.5, 0.0, 1.0);
    #endif
    #if defined(BLOOM_DOWNSAMPLE) && BLOOM_PASS == 1
    vec3 value = max(texture(usam_main, uv).rgb, 0.0);
    #if SETTING_BLOOM_HIGHLIGHT_COMPRESSION != 0
    value = bloom_compressHighlights(value);
    #endif
    #else
    vec3 value = transient_bloom_sample(uv).rgb;
    #endif
    #if defined(BLOOM_DOWNSAMPLE) || defined(BLOOM_ZERO_EXTEND)
    value *= coverage.x * coverage.y;
    #endif
    return value;
}

#include "/techniques/BloomTaps.glsl"

vec3 bloom_filterPixel(ivec2 pos) {
    #ifdef BLOOM_DOWNSAMPLE
    const int inputLevel = BLOOM_PASS - 1;
    vec2 center = vec2(pos) * 2.0 + 1.0;
    #else
    const int inputLevel = BLOOM_PASS;
    vec2 center = (vec2(pos) + 0.5) * 0.5;
    #endif
    return bloom_filter(center, vec4(global_mipmapTileCeilPadded[inputLevel]));
}

#endif
