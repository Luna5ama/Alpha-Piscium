#ifndef INCLUDE_util_BSDF.glsl
#define INCLUDE_util_BSDF.glsl

#include "Material.glsl"
#include "Math.glsl"

vec3 bsdf_f0ToIor(vec3 f0) {
    vec3 f0Sqrt = sqrt(f0) * 0.9999;
    return (1.0 + f0Sqrt) / (1.0 - f0Sqrt);
}

float bsdf_f0ToIor(float f0) {
    float f0Sqrt = sqrt(f0) * 0.9999;
    return (1.0 + f0Sqrt) / (1.0 - f0Sqrt);
}

vec3 bsdf_frenel_cookTorrance_ior(float cosTheta, float ior) {
    float c = float(cosTheta);
    float g = sqrt(ior * ior + c * c - float(1.0));
    return vec3(0.5 * pow2((g - c) / (g + c)) * (1.0 + pow2(((g + c) * c - 1.0) / ((g - c) * c + 1.0))));
}

vec3 bsdf_frenel_cookTorrance_f0(float cosTheta, float f0) {
    float ior = bsdf_f0ToIor(f0);
    return bsdf_frenel_cookTorrance_ior(cosTheta, ior);
}

vec3 bsdf_frenel_schlick_f0(float cosTheta, vec3 f0) {
    return f0 + (1.0 - f0) * pow5(1.0 - cosTheta);
}

float bsdf_frenel_schlick_f90(float cosTheta, float f90) {
    return 1.0 + (f90 - 1.0) * pow5(1.0 - cosTheta);
}

vec3 bsdf_frenel_schlick_ior(float cosTheta, vec3 ior) {
    vec3 f0 = pow2((ior - 1.0) / (ior + 1.0));
    return bsdf_frenel_schlick_f0(cosTheta, f0);
}

// Fresnel Term Approximations for Metals by Lazanyi
vec3 bsdf_fresnel_lazanyi(float cosTheta, vec3 ior, vec3 k) {
    vec3 k2 = pow2(k);
    return (pow2(ior - 1.0) + 4 * ior * pow5(1.0 - cosTheta) + k2) / (pow2(ior + 1.0) + k2);
}

// PPBR Diffuse Lighting for GGX+Smith Microsurfaces by Earl Hammon Jr.
// (https://www.gdcvault.com/play/1024478/PBR-Diffuse-Lighting-for-GGX)
vec3 bsdf_diffuseHammon(Material material, float NDotL, float NDotV, float NDotH, float LDotV) {
    if (NDotL <= 0.0) return vec3(0.0);
    float facing = saturate(0.5 * LDotV + 0.5);

    // Jessie modification for energy conservation
    float fresnelNL = bsdf_frenel_cookTorrance_f0(max(NDotL, 1e-2), material.f0).x;
    float fresnelNV = bsdf_frenel_cookTorrance_f0(max(NDotV, 1e-2), material.f0).x;
    float energyConservationFactor = 1.0 - (4.0 * sqrt(material.f0) + 5.0 * material.f0 * material.f0) * (1.0 / 9.0);
    float singleSmooth = (1.0 - fresnelNL) * (1.0 - fresnelNV) / energyConservationFactor;

    float singleRough = facing * (0.9 - 0.4 * facing) * (0.5 + NDotH) / NDotH;
    float single = mix(singleSmooth, singleRough, material.roughness) * RCP_PI;
    float multi = 0.1159 * material.roughness;
    return material.albedo * (singleRough + material.albedo * multi);
}

vec3 bsdf_disneyDiffuse(Material material, float NDotL, float NDotV, float LDotH) {
    if (NDotL <= 0.0) return vec3(0.0);

    float f90 = 2.0 * material.roughness * LDotH * LDotH + 0.5;
    float fresnelNL90 = bsdf_frenel_schlick_f90(max(NDotL, 0.01), f90);
    float fresnelNV90 = bsdf_frenel_schlick_f90(max(NDotV, 0.01), f90);

    return (RCP_PI * f90 * fresnelNL90 * fresnelNV90) * material.albedo;
}

float _gs_Smith_Schlick(float NDotV, float k) {
    return NDotV / (NDotV * (1.0 - k) + k);
}

float _gs_Smith_Schlick_GGX(float NDotV, float a) {
    float k = a / 2.0;
    return _gs_Smith_Schlick(NDotV, k);
}

vec3 bsdf_ggx(Material material, vec3 F, float NDotL, float NDotV, float NDotH) {
    if (NDotL <= 0.0) return vec3(0.0);
    float NDotH2 = NDotH * NDotH;
    float a2 = material.roughness * material.roughness;

    // Normal Distribution Function Term
    float d;
    // GGX (Trowbridge-Reitz)
    {
        float sqTerm = NDotH2 * (a2 - 1.0) + 1.0;
        d = a2 / max(PI * sqTerm * sqTerm, 0.00001);
    }

    // Geometric Shadowing Term
    float g;
    // Smith
    {
        float gl, gv;
        // Schlick GGX
        {
            gl = _gs_Smith_Schlick_GGX(max(NDotL, 0.0), material.roughness);
            gv = _gs_Smith_Schlick_GGX(max(NDotV, 0.0), material.roughness);
        }
        g = gl * gv;
    }

    return material.albedo * F * ((d * g) / (4 * NDotL * NDotV));
}

#endif