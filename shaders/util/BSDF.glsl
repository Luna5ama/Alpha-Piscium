#ifndef INCLUDE_util_BSDF.glsl
#define INCLUDE_util_BSDF.glsl

#include "Math.glsl"

vec3 bsdf_f0ToIor(vec3 f0) {
    vec3 f0Sqrt = sqrt(f0) * 0.9999;
    return (1.0 + f0Sqrt) / (1.0 - f0Sqrt);
}

float bsdf_f0ToIor(float f0) {
    float f0Sqrt = sqrt(f0) * 0.9999;
    return (1.0 + f0Sqrt) / (1.0 - f0Sqrt);
}

vec3 bsdf_frenel_cook_torrance(float lDotH, float f0) {
    float ior = bsdf_f0ToIor(f0);
    float c = float(lDotH);
    float g = sqrt(ior * ior + c * c - float(1.0));
    return vec3(0.5 * pow2((g - c) / (g + c)) * (1.0 + pow2(((g + c) * c - 1.0) / ((g - c) * c + 1.0))));
}

vec3 bsdf_frenel_schlick_f0(float lDotH, vec3 f0) {
    return f0 + (1.0 - f0) * pow5(1.0 - lDotH);
}

vec3 bsdf_frenel_schlick(float lDotH, vec3 ior) {
    vec3 f0 = pow2((ior - 1.0) / (ior + 1.0));
    return f0 + (1.0 - f0) * pow5(1.0 - lDotH);
}

// Fresnel Term Approximations for Metals by Lazanyi
vec3 bsdf_fresnel_lazanyi(float lDotH, vec3 ior, vec3 k) {
    vec3 k2 = pow2(k);
    return (pow2(ior - 1.0) + 4 * ior * pow5(1.0 - lDotH) + k2) / (pow2(ior + 1.0) + k2);
}

// PPBR Diffuse Lighting for GGX+Smith Microsurfaces by Earl Hammon Jr.
// (https://www.gdcvault.com/play/1024478/PBR-Diffuse-Lighting-for-GGX)
vec3 bsdf_diffuseHammon(float NDotL, float NDotV, float NDotH, float LDotV, vec3 albedo, float a) {
    if (NDotL <= 0.0) return vec3(0.0);
    float facing = 0.5 + 0.5 * LDotV;
    float singleRough = facing * (0.9 - 0.4 * facing) * (0.5 + NDotH) / NDotH;
    float singleSmooth = 1.05 * (1.0 - pow5(1.0 - NDotL)) * (1.0 - pow5(1.0 - NDotV));
    float single = mix(singleSmooth, singleRough, a) * RCP_PI_CONST;
    float multi = 0.1159 * a;
    return albedo * (single + albedo * multi);
}

float _gs_Smith_Schlick(float NDotV, float k) {
    return NDotV / (NDotV * (1.0 - k) + k);
}

float _gs_Smith_Schlick_GGX(float NDotV, float a) {
    float k = a / 2.0;
    return _gs_Smith_Schlick(NDotV, k);
}

vec3 bsdf_ggx(float NDotL, float NDotV, float NDotH, vec3 f, vec3 albedo, float a) {
    if (NDotL <= 0.0) return vec3(0.0);
    float NDotH2 = NDotH * NDotH;
    float a2 = a * a;

    // Normal Distribution Function Term
    float d;
    // GGX (Trowbridge-Reitz)
    {
        float sqTerm = NDotH2 * (a2 - 1.0) + 1.0;
        d = a2 / max(PI_CONST * sqTerm * sqTerm, 0.00001);
    }

    // Geometric Shadowing Term
    float g;
    // Smith
    {
        float gl, gv;
        // Schlick GGX
        {
            gl = _gs_Smith_Schlick_GGX(max(NDotL, 0.0), a);
            gv = _gs_Smith_Schlick_GGX(max(NDotV, 0.0), a);
        }
        g = gl * gv;
    }

    return albedo * f * ((d * g) / (4 * NDotL * NDotV));
}

#endif