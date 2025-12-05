#include "/util/Colors.glsl"
#include "/util/Colors2.glsl"
#include "/util/Math.glsl"

vec3 ffxcas_load(ivec2 texelPos);

vec3 ffxcas_fastPass(ivec2 texelPos) {
    vec3 c10 = ffxcas_load(texelPos + ivec2(0, -1));
    vec3 c01 = ffxcas_load(texelPos + ivec2(-1, 0));
    vec3 c11 = ffxcas_load(texelPos);
    vec3 c21 = ffxcas_load(texelPos + ivec2(1, 0));
    vec3 c12 = ffxcas_load(texelPos + ivec2(0, 1));
    float b10 = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, c10);
    float b01 = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, c01);
    float b11 = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, c11);
    float b21 = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, c21);
    float b12 = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, c12);
    float minBrightness = min(min4(b10, b01, b11, b21), b12);
    float maxBrightness = max(max4(b10, b01, b11, b21), b12);
    float contrast = maxBrightness - minBrightness;
    float sharpnessScale = 1.0 / (1.0 + contrast * 10.0);
    float sharpenFactor = clamp(FFXCAS_SHARPENESS * sharpnessScale, 0.0, 1.0);
    vec3 sharpenedColor = c11 * (1.0 + 4.0 * FFXCAS_SHARPENESS) - (c01 + c10 + c12 + c21) * FFXCAS_SHARPENESS;
    sharpenedColor = clamp(sharpenedColor, min(min(c10, c12), min(c01, c21)), max(max(c10, c12), max(c01, c21)));
    return mix(c11, sharpenedColor, sharpenFactor);
}

vec3 ffxcas_pass(ivec2 texelPos) {
    vec3 c00 = ffxcas_load(texelPos + ivec2(-1, -1));
    vec3 c10 = ffxcas_load(texelPos + ivec2(0, -1));
    vec3 c20 = ffxcas_load(texelPos + ivec2(1, -1));
    vec3 c01 = ffxcas_load(texelPos + ivec2(-1, 0));
    vec3 c11 = ffxcas_load(texelPos);
    vec3 c21 = ffxcas_load(texelPos + ivec2(1, 0));
    vec3 c02 = ffxcas_load(texelPos + ivec2(-1, 1));
    vec3 c12 = ffxcas_load(texelPos + ivec2(0,  1));
    vec3 c22 = ffxcas_load(texelPos + ivec2(1, 1));
    float b00 = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, c00);
    float b10 = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, c10);
    float b20 = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, c20);
    float b01 = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, c01);
    float b11 = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, c11);
    float b21 = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, c21);
    float b02 = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, c02);
    float b12 = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, c12);
    float b22 = colors2_colorspaces_luma(COLORS2_WORKING_COLORSPACE, c22);
    float minBrightness = mmin3(min4(b00, b10, b20, b01), min4(b11, b21, b02, b12), b22);
    float maxBrightness = mmax3(max4(b00, b10, b20, b01), max4(b11, b21, b02, b12), b22);
    float contrast = maxBrightness - minBrightness;
    float sharpnessScale = 1.0 / (1.0 + contrast * 10.0);
    float sharpenFactor = clamp(FFXCAS_SHARPENESS * sharpnessScale, 0.0, 1.0);
    vec3 sharpenedColor = c11 * (1.0 + 8.0 * FFXCAS_SHARPENESS) - (c00 + c01 + c02 + c10 + c12 + c20 + c21 + c22) * FFXCAS_SHARPENESS;
    sharpenedColor = clamp(sharpenedColor, min(min(c10, c12), min(c01, c21)), max(max(c10, c12), max(c01, c21)));
    return mix(c11, sharpenedColor, sharpenFactor);
}