#ifndef INCLUDE_util_BSDF_glsl
#define INCLUDE_util_BSDF_glsl a
/*
    References:
        [GIL23] Gilcher, Pascal. "Better GGX VNDF Sampler". Shadertoy. 2023.
            https://www.shadertoy.com/view/MX3XDf
            MIT License. Copyright (c) 2023 Pascal Gilcher.
        [HAM17] Hammon, Earl, Jr. "PBR Diffuse Lighting for GGX+Smith Microsurfaces". GDC 2017. 2017.
            https://ubm-twvideo01.s3.amazonaws.com/o1/vault/gdc2017/Presentations/Hammon_Earl_PBR_Diffuse_Lighting.pdf
        [HOF13] Hoffman, Naty. "Crafting Physically Motivated Shading Models for Game Development".
            SIGGRAPH 2010 Course: Physically Based Shading Models in Film and Game Production. 2010.
            https://michael-hofmann.net/2013/04/30/a-practical-model-for-anisotropic-brdfs/
        [KAR10] Karis, Brian. "Specular BRDF Reference". Graphic Rants. 2013.
            https://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html

        You can find full license texts in /licenses)
*/
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

// [GIL23]
float _bsdf_lambdaSmith(float ndotx, float alpha) {
    float alpha_sqr = alpha * alpha;
    float ndotx_sqr = ndotx * ndotx;
    return (-1.0 + sqrt(alpha_sqr * (1.0 - ndotx_sqr) / ndotx_sqr + 1.0)) * 0.5;
}

// [GIL23]
float bsdf_smithG1(float ndotv, float alpha) {
    float lambda_v = _bsdf_lambdaSmith(ndotv, alpha);
    return 1.0 / (1.0 + lambda_v);
}

// [GIL23]
float bsdf_smithG2(float ndotl, float ndotv, float alpha) {
    //height correlated
    float lambda_v = _bsdf_lambdaSmith(ndotv, alpha);
    float lambda_l = _bsdf_lambdaSmith(ndotl, alpha);
    return 1.0 / (1.0 + lambda_v + lambda_l);
}

// [GIL23]
vec3 bsdf_SphericalCapBoundedWithPDFRatio(vec2 u, vec3 wi, vec2 alpha, out float pdf_ratio) {
    #define mad(a,b,c) ((a)*(b)+(c))
    // warp to the hemisphere configuration

    //PGilcher: save the length t here for pdf ratio
    vec3 wiStd = vec3(wi.xy * alpha, wi.z);
    float t = length(wiStd);
    wiStd /= t;

    // sample a spherical cap in (-wi.z, 1]
    float phi = (2.0f * u.x - 1.0f) * PI;

    float a = saturate(min( alpha.x, alpha.y)); // Eq. 6
    float s = 1.0f + length(wi.xy); // Omit sgn for a <=1
    float a2 = a * a;
    float s2 = s * s;
    float k = (1.0 - a2) * s2 / (s2 + a2 * wi.z * wi.z);

    float b = wiStd.z;
    b = wi.z > 0.0 ? k * b : b;

    //PGilcher: compute ratio of unchanged pdf to actual pdf (ndf/2 cancels out)
    //Dupuy's method is identical to this except that "k" is always 1, so
    //we extract the differences of the PDFs (Listing 2 in the paper)
    pdf_ratio = (k * wi.z + t) / (wi.z + t);

    float z = mad((1.0f - u.y), (1.0f + b), -b);
    float sinTheta = sqrt(clamp(1.0f - z * z, 0.0f, 1.0f));
    float x = sinTheta * cos(phi);
    float y = sinTheta * sin(phi);
    vec3 c = vec3(x, y, z);
    // compute halfway direction as standard normal
    vec3 wmStd = c + wiStd;
    // warp back to the ellipsoid configuration
    vec3 wm = normalize(vec3(wmStd.xy * alpha, wmStd.z));
    // return final normal
    return wm;
    #undef mad
}

#endif