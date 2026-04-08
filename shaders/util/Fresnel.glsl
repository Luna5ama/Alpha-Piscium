#ifndef INCLUDE_util_Fresnel_glsl
#define INCLUDE_util_Fresnel_glsl a
/*
    References:
        [ARN08] Arnott, W.P. "Fresnel equations". 2008.
            https://www.patarnott.com/atms749/pdf/FresnelEquations.pdf
        [KUT21] Kutz, Peter, et al. "Novel aspects of the Adobe Standard Material". 2021.
            https://helpx.adobe.com/content/dam/substance-3d/general-knowledge/asm/Adobe%20Standard%20Material%20-%20Technical%20Documentation%20-%20May2023.pdf
        [LAB21] shaderLABS. "LabPBR Material Standard". 2021.
            https://shaderlabs.org/wiki/LabPBR_Material_Standard
        [LAG13] Lagarde, Sébastien. "Memo on Fresnel equations". 2013.
            https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
        [WIK23] Wikipedia. "Snell's law". 2023.
            https://en.wikipedia.org/wiki/Snell%27s_law
        [BEL25a] Belmu. "fresnel.glsl". Noble Shaders.
            GPL v3.0 License. Copyright (c) 2025 Belmu
            https://github.com/BelmuTM/Noble/blob/4d3544b078e9c71debc0c6ac9936b9e9847b442d/shaders/include/fragment/fresnel.glsl
        [BEL25b] Belmu. "material.glsl". Noble Shaders.
            GPL v3.0 License. Copyright (c) 2025 Belmu
            https://github.com/BelmuTM/Noble/blob/4d3544b078e9c71debc0c6ac9936b9e9847b442d/shaders/include/utility/material.glsl

        You can find full license texts in /licenses

    Other Credits:
        Jessie - providing beam ratio equation for transmission (https://github.com/Jessie-LC)
*/

#include "Material.glsl"

const float AIR_IOR = 1.00029;
const float WATER_IOR = 1.333;

// [BEL25b]
vec3 fresnel_f0ToIor(vec3 f0) {
    vec3 f0Sqrt = sqrt(f0) * 0.99999;
    return AIR_IOR * ((1.0 + f0Sqrt) / (1.0 - f0Sqrt));
}

// [BEL25b]
float fresnel_f0ToIor(float f0) {
    float f0Sqrt = sqrt(f0) * 0.99999;
    return AIR_IOR * ((1.0 + f0Sqrt) / (1.0 - f0Sqrt));
}

// [BEL25b]
float fresnel_iorToF0(float ior) {
    return pow2((ior - AIR_IOR) / (ior + AIR_IOR));
}

// [BEL25a]
vec3 fresnel_dielectricDielectric_reflection(float cosThetaI, vec3 n1, vec3 n2) {
    float sinThetaI = sqrt(saturate(1.0 - pow2(cosThetaI)));
    vec3 sinThetaT = (n1 / n2) * sinThetaI;
    vec3 cosThetaT = sqrt(saturate(1.0 - pow2(sinThetaT)));

    vec3 Rs = (n1 * cosThetaI - n2 * cosThetaT) / (n1 * cosThetaI + n2 * cosThetaT);
    vec3 Rp = (n1 * cosThetaT - n2 * cosThetaI) / (n1 * cosThetaT + n2 * cosThetaI);

    return saturate((Rs * Rs + Rp * Rp) * 0.5);
}

float fresnel_dielectricDielectric_reflection(float cosThetaI, float n1, float n2) {
    float sinThetaI = sqrt(saturate(1.0 - pow2(cosThetaI)));
    float sinThetaT = (n1 / n2) * sinThetaI;
    float cosThetaT = sqrt(saturate(1.0 - pow2(sinThetaT)));

    float Rs = (n1 * cosThetaI - n2 * cosThetaT) / (n1 * cosThetaI + n2 * cosThetaT);
    float Rp = (n1 * cosThetaT - n2 * cosThetaI) / (n1 * cosThetaT + n2 * cosThetaI);

    return saturate((Rs * Rs + Rp * Rp) * 0.5);
}

// [BEL25a]
vec3 fresnel_dielectricDielectric_transmittance(float cosThetaI, vec3 n1, vec3 n2) {
    float sinThetaI = sqrt(saturate(1.0 - pow2(cosThetaI)));
    vec3 sinThetaT = (n1 / n2) * sinThetaI;
    vec3 cosThetaT = sqrt(saturate(1.0 - pow2(sinThetaT)));


    vec3 result = vec3(1.0);
    if (any(lessThan(sinThetaT, vec3(1.0)))) {
        vec3 numerator = 2.0 * n1 * cosThetaI;

        vec3 Ts = abs(numerator / (n1 * cosThetaI + n2 * cosThetaT));
        vec3 Tp = abs(numerator / (n1 * cosThetaT + n2 * cosThetaI));

        vec3 beamRatio = abs((n2 * cosThetaT) / (n1 * cosThetaI));

        result = saturate(beamRatio * (Ts * Ts + Tp * Tp) * 0.5);
    }

    return result;
}

// [BEL25a]
float fresnel_dielectricDielectric_transmittance(float cosThetaI, float n1, float n2) {
    float sinThetaI = sqrt(saturate(1.0 - pow2(cosThetaI)));
    float sinThetaT = (n1 / n2) * sinThetaI;
    float cosThetaT = sqrt(saturate(1.0 - pow2(sinThetaT)));

    float result = 1.0;
    if (sinThetaT < 1.0) {
        float numerator = 2.0 * n1 * cosThetaI;

        float Ts = abs(numerator / (n1 * cosThetaI + n2 * cosThetaT));
        float Tp = abs(numerator / (n1 * cosThetaT + n2 * cosThetaI));

        float beamRatio = abs((n2 * cosThetaT) / (n1 * cosThetaI));

        result = saturate(beamRatio * (Ts * Ts + Tp * Tp) * 0.5);
    }

    return result;
}

// [BEL25a]
vec3 fresnel_dielectricConductor(float cosTheta, vec3 eta, vec3 etaK) {
    float cosThetaSq = cosTheta * cosTheta, sinThetaSq = 1.0 - cosThetaSq;
    vec3 etaSq = eta * eta, etaKSq = etaK * etaK;

    vec3 t0 = etaSq - etaKSq - sinThetaSq;
    vec3 a2b2 = sqrt(t0 * t0 + 4.0 * etaSq * etaKSq);
    vec3 t1 = a2b2 + cosThetaSq;
    vec3 t2 = 2.0 * sqrt(0.5 * (a2b2 + t0)) * cosTheta;
    vec3 Rs = (t1 - t2) / (t1 + t2);

    vec3 t3 = cosThetaSq * a2b2 + sinThetaSq * sinThetaSq;
    vec3 t4 = t2 * sinThetaSq;
    vec3 Rp = Rs * (t3 - t4) / (t3 + t4);

    return saturate((Rp + Rs) * 0.5);
}

// [LAG13]
vec3 frenel_schlick(float cosTheta, vec3 f0) {
    return f0 + (1.0 - f0) * pow5(1.0 - cosTheta);
}

float frenel_schlick(float cosTheta, float f0) {
    return f0 + (1.0 - f0) * pow5(1.0 - cosTheta);
}

// [KUT21]
// f82Tint isn't the absolute fresnel value at 82 degrees like the 2019 Lazanyi
// But rather a multiplier over the schlick fresnel at 82 degrees
float fresnel_adobe(float cosTheta, float f0, float f82Tint) {
    float oneMinusF0 = 1.0 - f0;
    float b = fma(oneMinusF0, 0.462664878484, f0) * fma(f82Tint, -17.6513843536, 17.6513843536);
    return saturate(fma(fma(fma(cosTheta, cosTheta, -cosTheta), b, oneMinusF0), pow5(1.0 - cosTheta), f0));
}

// [KUT21]
// f82Tint isn't the absolute fresnel value at 82 degrees like the 2019 Lazanyi
// But rather a multiplier over the schlick fresnel at 82 degrees
vec3 fresnel_adobe(float cosTheta, vec3 f0, vec3 f82Tint) {
    vec3 oneMinusF0 = vec3(1.0) - f0;
    vec3 b = fma(oneMinusF0, vec3(0.462664878484), f0) * fma(f82Tint, vec3(-17.6513843536), vec3(17.6513843536));
    return saturate(fma(fma(vec3(fma(cosTheta, cosTheta, -cosTheta)), b, oneMinusF0), vec3(pow5(1.0 - cosTheta)), f0));
}

vec3 fresnel_evalMaterial(Material material, float cosTheta) {
    return fresnel_adobe(cosTheta, material.f0RGB, material.f82TintRGB);
}

#endif