#ifndef INCLUDE_util_BSDF_glsl
#define INCLUDE_util_BSDF_glsl a
/*
    References:
        [GIL23] Gilcher, Pascal. "Better GGX VNDF Sampler". Shadertoy. 2023.
            https://www.shadertoy.com/view/MX3XDf
            MIT License. Copyright (c) 2023 Pascal Gilcher.
        [HAM17] Hammon, Earl, Jr. "PBR Diffuse Lighting for GGX+Smith Microsurfaces". GDC 2017. 2017.
            https://ubm-twvideo01.s3.amazonaws.com/o1/vault/gdc2017/Presentations/Hammon_Earl_PBR_Diffuse_Lighting.pdf
        [HOF10] Hoffman, Naty. "Background: Physics and Math of Shading".
            SIGGRAPH 2012 Course: Practical Physically Based Shading in Film and Game Production. 2010.
            https://blog.selfshadow.com/publications/s2012-shading-course/hoffman/s2012_pbs_physics_math_notes.pdf
        [KAR10] Karis, Brian. "Specular BRDF Reference". Graphic Rants. 2013.
            https://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html

        You can find full license texts in /licenses)
*/
#include "Fresnel.glsl"
#include "Material.glsl"
#include "Math.glsl"

// [HAM17]
vec3 bsdf_diffuseHammon(Material material, float NDotL, float NDotV, float NDotH, float LDotV) {
    vec3 result = vec3(0.0);
    if (NDotL > 0.0) {
        float facing = saturate(0.5 * LDotV + 0.5);

        // Jessie's modification for energy conservation
        float fresnelNL = fresnel_dielectricDielectric_transmittance(max(NDotL, 1e-2), AIR_IOR, fresnel_f0ToIor(material.f0));
        float fresnelNV = fresnel_dielectricDielectric_transmittance(max(NDotV, 1e-2), AIR_IOR, fresnel_f0ToIor(material.f0));
        float energyConservationFactor = 1.0 - (4.0 * sqrt(material.f0) + 5.0 * material.f0 * material.f0) * rcp(9.0);
        float singleSmooth = (fresnelNL * fresnelNV) / energyConservationFactor;

        float singleRough = facing * (0.9 - 0.4 * facing) * (0.5 + NDotH) / NDotH;
        float single = mix(singleSmooth, singleRough, material.roughness) * RCP_PI;
        float multi = 0.1159 * material.roughness;
        result = NDotL * (single + material.albedo * multi);
    }
    return result;
}

// [KAR13]
float _bsdf_g_Smith_Schlick_denom(float cosTheta, float k) {
    return cosTheta * (1.0 - k) + k;
}

float bsdf_ggx(Material material, float NDotL, float NDotV, float NDotH) {
    float result = 0.0;
    if (NDotL > 0.0) {
        float NDotH2 = pow2(NDotH);
        float a2 = pow2(material.roughness);

        // Normal Distribution Function Term [KAR13]
        // GGX (Trowbridge-Reitz)
        float d = a2 / max(PI * pow2(NDotH2 * (a2 - 1.0) + 1.0), 1e-16);

        // Visibility Function Term [HOF10]
        float k = material.roughness * 0.5;
        float v = rcp(_bsdf_g_Smith_Schlick_denom(NDotL, k) * _bsdf_g_Smith_Schlick_denom(saturate(NDotV), k));

        result = NDotL * d * v * 0.25;
    }
    return result;
}

// [GIL23]
float _bsdf_lambdaSmith(float ndotx, float alpha) {
    float alpha_sqr = alpha * alpha;
    float ndotx_sqr = ndotx * ndotx;
    return (-1.0 + sqrt(alpha_sqr * (1.0 - ndotx_sqr) / ndotx_sqr + 1.0)) * 0.5;
}

float bsdf_smithG1(float ndotv, float alpha) {
    // float lambda_v = _bsdf_lambdaSmith(ndotv, alpha);
    // return 1.0 / (1.0 + lambda_v);
    // Simplied version of the above
    float alphaSq = alpha * alpha;
    float ndotvSq = ndotv * ndotv;
    float denom = ndotv + sqrt(ndotvSq - alphaSq * ndotvSq + alphaSq);
    return 2.0 * ndotv / denom;
}

// [GIL23]
float bsdf_smithG2(float ndotl, float ndotv, float alpha) {
    //height correlated
    float lambda_v = _bsdf_lambdaSmith(ndotv, alpha);
    float lambda_l = _bsdf_lambdaSmith(ndotl, alpha);
    return 1.0 / (1.0 + lambda_v + lambda_l);
}

// Zombye
vec3 bsdf_VNDFSphericalCap(
    vec3 viewerDirection, // Direction pointing towards the viewer, oriented such that +Z corresponds to the surface normal
    vec2 alpha, // Roughness parameter along X and Y of the distribution
    vec2 xy // Pair of uniformly distributed numbers in [0, 1)
) {
    // Transform viewer direction to the hemisphere configuration
    viewerDirection = normalize(vec3(alpha * viewerDirection.xy, viewerDirection.z));

    // Sample a reflection direction off the hemisphere
    const float tau = 6.2831853; // 2 * pi
    float phi = tau * xy.x;
    float cosTheta = fma(1.0 - xy.y, 1.0 + viewerDirection.z, -viewerDirection.z);
    float sinTheta = sqrt(clamp(1.0 - cosTheta * cosTheta, 0.0, 1.0));
    vec3 reflected = vec3(vec2(cos(phi), sin(phi)) * sinTheta, cosTheta);

    // Evaluate halfway direction
    // This gives the normal on the hemisphere
    vec3 halfway = reflected + viewerDirection;

    // Transform the halfway direction back to hemiellispoid configuation
    // This gives the final sampled normal
    return normalize(vec3(alpha * halfway.xy, halfway.z));
}

// Return:
// xyz: visible normal vector
// w: pdf
vec3 bsdf_VNDFSphericalCapTrimmed(
    vec3 V, // Direction pointing towards the viewer, oriented such that +Z corresponds to the surface normal
    float alpha, // Roughness parameter along X and Y of the distribution
    vec2 xy, // Pair of uniformly distributed numbers in [0, 1)
    float trimFactor
) {
    const float EPS = 1e-5;
    // Transform viewer direction to the hemisphere configuration
    vec3 stretchedV = normalize(vec3(alpha * V.xy, V.z));

    float yMax = clamp(1.0 - trimFactor / (1.0 + stretchedV.z), 0.0, 1.0);
    xy.y *= yMax;

    // Sample a reflection direction off the hemisphere
    const float tau = 6.2831853; // 2 * pi
    float phi = tau * xy.x;
    float cosTheta = fma(1.0 - xy.y, 1.0 + stretchedV.z, -stretchedV.z);
    float sinTheta = sqrt(clamp(1.0 - cosTheta * cosTheta, 0.0, 1.0));
    vec3 reflected = vec3(vec2(cos(phi), sin(phi)) * sinTheta, cosTheta);

    // Evaluate halfway direction
    // This gives the normal on the hemisphere
    vec3 Hstretched = reflected + stretchedV;

    // Transform the halfway direction back to hemiellispoid configuation
    // This gives the final sampled normal
    return normalize(vec3(alpha * Hstretched.xy, Hstretched.z));
}

#endif