#ifndef INCLUDE_util_SplitSumSpecular_glsl
#define INCLUDE_util_SplitSumSpecular_glsl a

#include "Math.glsl"

vec3 splitSumSpecularLUT(vec3 F0, vec3 F82Tint, float NDotV, float roughness) {
    vec2 lutCoord = vec2(saturate(NDotV), 1.0 - saturate(roughness));
    vec3 specBrdf = texture(usam_specBRDFLUT, lutCoord).rgb;
    return saturate(F0 * specBrdf.x + F82Tint * specBrdf.y + specBrdf.z);
}

vec3 splitSumSpecularLUT(float F0, float NDotV, float roughness) {
    return splitSumSpecularLUT(vec3(F0), vec3(1.0), NDotV, roughness);
}

const float GI_SPEC_DENOISE_MIN_FACTOR = 0.05;

vec3 splitSumSpecularDenoiseFactor(vec3 F0, vec3 F82Tint, float NDotV, float roughness) {
    vec3 physicalAlbedo = splitSumSpecularLUT(F0, F82Tint, NDotV, roughness);
    return max(physicalAlbedo, vec3(GI_SPEC_DENOISE_MIN_FACTOR));
}

vec3 splitSumSpecularDenoiseFactor(float F0, float NDotV, float roughness) {
    return splitSumSpecularDenoiseFactor(vec3(F0), vec3(1.0), NDotV, roughness);
}

#endif
