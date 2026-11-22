#ifndef INCLUDE_BLOOM
#define INCLUDE_BLOOM

#include "/Base.glsl"

#if (defined(BLOOM_DOWNSAMPLE) && BLOOM_PASS == 1) || defined(BLOOM_FINAL_RECONSTRUCTION)
#if SETTING_BLOOM_HIGHLIGHT_COMPRESSION != 0
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

vec3 bloom_sampleInput(vec2 uv) {
    vec3 value = max(texture(usam_main, uv).rgb, 0.0);
    #if SETTING_BLOOM_HIGHLIGHT_COMPRESSION != 0
    value = bloom_compressHighlights(value);
    #endif
    return value;
}
#endif

vec3 bloom_sample(vec2 pos, vec4 tile) {
    // Clamp inside the input mip before converting to atlas coordinates.
    vec2 texel = tile.xy + clamp(pos, vec2(0.5), tile.zw - 0.5);
    vec2 uv = texel * uval_mainImageSizeRcp;
    #ifdef BLOOM_DOWNSAMPLE
    // Zero-extend the bilinear footprint without sampling an adjacent tile.
    vec2 coverage = clamp(min(pos, tile.zw - pos) + 0.5, 0.0, 1.0);
    #endif
    #if defined(BLOOM_DOWNSAMPLE) && BLOOM_PASS == 1
    vec3 value = bloom_sampleInput(uv);
    #else
    #if defined(BLOOM_DOWNSAMPLE) || defined(BLOOM_FIRST_UPSAMPLE)
    vec3 value = transient_bloomRaw_sample(uv).rgb;
    #else
    vec3 value = transient_bloomReconstruction_sample(uv).rgb;
    #endif
    #endif
    #ifdef BLOOM_DOWNSAMPLE
    value *= coverage.x * coverage.y;
    #endif
    return value;
}

#if !defined(BLOOM_DOWNSAMPLE) && (BLOOM_PASS > 1 || defined(BLOOM_FINAL_RECONSTRUCTION))
vec3 bloom_sampleRaw(vec2 pos, vec4 tile) {
    vec2 texel = tile.xy + clamp(pos, vec2(0.5), tile.zw - 0.5);
    vec2 coverage = clamp(min(pos, tile.zw - pos) + 0.5, 0.0, 1.0);
    vec2 uv = texel * uval_mainImageSizeRcp;
    #ifdef BLOOM_FINAL_RECONSTRUCTION
    vec3 value = bloom_sampleInput(uv);
    #else
    vec3 value = transient_bloomRaw_sample(uv).rgb;
    #endif
    return value * (coverage.x * coverage.y);
}
#endif

#define BLOOM_DPF_RUNTIME_D0 1
#define BLOOM_DPF_RUNTIME_CARRIER_LEVELS 1
#define BLOOM_DPF_RUNTIME_MAX_CARRIERS 6
#if !defined(BLOOM_DOWNSAMPLE) && (BLOOM_PASS == 2 || BLOOM_PASS == 3)
vec3 bloom_featureSample(vec2 uv, int feature) {
    if (feature == 0) return transient_gi_blurDiff2_sample(uv).rgb;
    if (feature == 1) return transient_gi_blurSpec2_sample(uv).rgb;
    if (feature == 2) return transient_gi_diffShadingOutput_sample(uv).rgb;
    if (feature == 3) return transient_gi3Reprojected_sample(uv).rgb;
    if (feature == 4) return transient_gi_specShadingOutput_sample(uv).rgb;
    return transient_gi_blurSpec1_sample(uv).rgb;
}

vec3 bloom_featureSampleEdge(vec2 pos, vec2 size, int feature) {
    vec2 texel = clamp(pos, vec2(0.5), size - 0.5);
    vec2 coverage = max(1.0 - abs(pos - texel), 0.0);
    return bloom_featureSample(texel * uval_mainImageSizeRcp, feature) * (coverage.x * coverage.y);
}
#endif

#if SETTING_BLOOM_SHAPE == 10
#include "/techniques/bloom/BloomDpfCustom.glsl"
#else
#include "/techniques/bloom/BloomDpfPresets.glsl"
#endif

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
