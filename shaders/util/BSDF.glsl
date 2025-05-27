/*
    References:
        [HAM17] Hammon, Earl, Jr. "PBR Diffuse Lighting for GGX+Smith Microsurfaces". GDC 2017. 2017.
            https://ubm-twvideo01.s3.amazonaws.com/o1/vault/gdc2017/Presentations/Hammon_Earl_PBR_Diffuse_Lighting.pdf
        [HOF13] Hoffman, Naty. "Crafting Physically Motivated Shading Models for Game Development".
            SIGGRAPH 2010 Course: Physically Based Shading Models in Film and Game Production. 2010.
            https://michael-hofmann.net/2013/04/30/a-practical-model-for-anisotropic-brdfs/
        [KAR10] Karis, Brian. "Specular BRDF Reference". Graphic Rants. 2013.
            https://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html

    Contains code adopted from:
        https://github.com/BelmuTM/Noble
        GPL v3.0 License
        Copyright (c) 2025 Belmu

        You can find full license texts in /licenses)
*/
#ifndef INCLUDE_util_BSDF_glsl
#define INCLUDE_util_BSDF_glsl a
#include "Fresnel.glsl"
#include "Material.glsl"
#include "Math.glsl"

// [HAM17]
vec3 bsdf_diffuseHammon(Material material, float NDotL, float NDotV, float NDotH, float LDotV) {
    if (NDotL <= 0.0) return vec3(0.0);
    float facing = saturate(0.5 * LDotV + 0.5);

    // Jessie modification for energy conservation
    float fresnelNL = fresnel_dielectricDielectric_transmittance(max(NDotL, 1e-2), vec3(AIR_IOR), vec3(fresnel_f0ToIor(material.f0))).x;
    float fresnelNV = fresnel_dielectricDielectric_transmittance(max(NDotV, 1e-2), vec3(AIR_IOR), vec3(fresnel_f0ToIor(material.f0))).x;
    float energyConservationFactor = 1.0 - (4.0 * sqrt(material.f0) + 5.0 * material.f0 * material.f0) * rcp(9.0);
    float singleSmooth = (fresnelNL * fresnelNV) / energyConservationFactor;

    float singleRough = facing * (0.9 - 0.4 * facing) * (0.5 + NDotH) / NDotH;
    float single = mix(singleSmooth, singleRough, material.roughness) * RCP_PI;
    float multi = 0.1159 * material.roughness;
    return NDotL * (singleRough + material.albedo * multi);
}

// [KAR13]
float _bsdf_g_Smith_Schlick_denom(float cosTheta, float k) {
    return cosTheta * (1.0 - k) + k;
}

float bsdf_ggx(Material material, float NDotL, float NDotV, float NDotH) {
    if (NDotL <= 0.0) return 0.0;
    float NDotH2 = pow2(NDotH);
    float a2 = pow2(material.roughness);

    // Normal Distribution Function Term [KAR13]
    // GGX (Trowbridge-Reitz)
    float d = a2 / (PI * pow2(NDotH2 * (a2 - 1.0) + 1.0));

    // Visibility Function Term [HOF10]
    float k = material.roughness * 0.5;
    float v = rcp(_bsdf_g_Smith_Schlick_denom(NDotL, k) * _bsdf_g_Smith_Schlick_denom(saturate(NDotV), k));

    return NDotL * d * v;
}

#endif