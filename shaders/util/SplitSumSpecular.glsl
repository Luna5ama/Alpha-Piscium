#ifndef INCLUDE_util_SplitSumSpecular_glsl
#define INCLUDE_util_SplitSumSpecular_glsl a

#include "Math.glsl"

vec3 splitSumSpecularLUT(vec3 F0, vec3 F82Tint, float NDotV, float roughness) {
    vec3 specBrdf = texture(usam_specBRDFLUT, vec2(saturate(NDotV), roughness)).rgb;
    return saturate(F0 * specBrdf.x + F82Tint * specBrdf.y + specBrdf.z);
}

vec3 splitSumSpecularLUT(float F0, float NDotV, float roughness) {
    return splitSumSpecularLUT(vec3(F0), vec3(1.0), NDotV, roughness);
}

#ifndef GI_SPEC_DENOISE_MIN_FACTOR
#define GI_SPEC_DENOISE_MIN_FACTOR 0.02
#endif

#ifndef GI_SPEC_DENOISE_ROUGHNESS_START
#define GI_SPEC_DENOISE_ROUGHNESS_START 0.02
#endif

#ifndef GI_SPEC_DENOISE_ROUGHNESS_END
#define GI_SPEC_DENOISE_ROUGHNESS_END 0.08
#endif

vec3 splitSumSpecularDenoiseFactor(vec3 F0, vec3 F82Tint, float NDotV, float roughness) {
    vec3 physicalAlbedo = splitSumSpecularLUT(F0, F82Tint, NDotV, roughness);
    vec3 liftedFactor = mix(vec3(GI_SPEC_DENOISE_MIN_FACTOR), vec3(1.0), physicalAlbedo);
    float demodulationAmount = smoothstep(
        GI_SPEC_DENOISE_ROUGHNESS_START,
        GI_SPEC_DENOISE_ROUGHNESS_END,
        roughness
    );
    return mix(vec3(1.0), liftedFactor, demodulationAmount);
}

vec3 splitSumSpecularDenoiseFactor(float F0, float NDotV, float roughness) {
    return splitSumSpecularDenoiseFactor(vec3(F0), vec3(1.0), NDotV, roughness);
}

#endif
