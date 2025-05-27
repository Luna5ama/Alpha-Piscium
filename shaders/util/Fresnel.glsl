/*
    References:
        [ARN08] Arnott, W.P. "Fresnel equations". 2008.
            https://www.patarnott.com/atms749/pdf/FresnelEquations.pdf
        [LAG13] Lagarde, Sébastien . "Memo on Fresnel equations". 2013.
            https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
        [WIK23] Wikipedia. "Snell's law". 2023.
            https://en.wikipedia.org/wiki/Snell%27s_law

    Contains code adopted from:
        https://github.com/BelmuTM/Noble
        GPL v3.0 License
        Copyright (c) 2025 Belmu

        You can find full license texts in /licenses

    Credits:
        Jessie - providing beam ratio equation for transmission (https://github.com/Jessie-LC)
*/
#ifndef INCLUDE_util_Fresnel.glsl
#define INCLUDE_util_Fresnel.glsl

#include "Material.glsl"

const float AIR_IOR = 1.00029;
const float WATER_IOR = 1.333;

vec3 _fresnel_calculateRefractedAngle(vec3 thetaI, vec3 n1, vec3 n2) {
    return asin((n1 / n2) * sin(thetaI));
}

vec3 fresnel_f0ToIor(vec3 f0) {
    vec3 f0Sqrt = sqrt(f0) * 0.99999;
    return AIR_IOR * ((1.0 + f0Sqrt) / (1.0 - f0Sqrt));
}

float fresnel_f0ToIor(float f0) {
    float f0Sqrt = sqrt(f0) * 0.99999;
    return AIR_IOR * ((1.0 + f0Sqrt) / (1.0 - f0Sqrt));
}

float fresnel_iorToF0(float ior) {
    return pow2((ior - AIR_IOR) / (ior + AIR_IOR));
}

vec3 fresnel_dielectricDielectric_reflection(float cosThetaI, vec3 n1, vec3 n2) {
    float sinThetaI = sqrt(1.0 - pow2(cosThetaI));
    vec3 sinThetaT = (n1 / n2) * sinThetaI;
    vec3 cosThetaT = sqrt(1.0 - pow2(sinThetaT));

    vec3 Rs = (n1 * cosThetaI - n2 * cosThetaT) / (n1 * cosThetaI + n2 * cosThetaT);
    vec3 Rp = (n1 * cosThetaT - n2 * cosThetaI) / (n1 * cosThetaT + n2 * cosThetaI);

    return saturate((Rs * Rs + Rp * Rp) * 0.5);
}

vec3 fresnel_dielectricDielectric_transmittance(float cosThetaI, vec3 n1, vec3 n2) {
    float sinThetaI = sqrt(1.0 - pow2(cosThetaI));
    vec3 sinThetaT = (n1 / n2) * sinThetaI;
    vec3 cosThetaT = sqrt(1.0 - pow2(sinThetaT));

    if (any(greaterThan(abs(sinThetaT), vec3(1.0)))) return vec3(1.0);

    vec3 numerator = 2.0 * n1 * cosThetaI;

    vec3 Ts = abs(numerator / (n1 * cosThetaI + n2 * cosThetaT));
    vec3 Tp = abs(numerator / (n1 * cosThetaT + n2 * cosThetaI));

    vec3 beamRatio = abs((n2 * cosThetaT) / (n1 * cosThetaI));

    return saturate(beamRatio * (Ts * Ts + Tp * Tp) * 0.5);
}

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

vec3 frenel_schlick(float cosTheta, vec3 f0) {
    return f0 + (1.0 - f0) * pow5(1.0 - cosTheta);
}

vec3 fresnel_evalMaterial(Material material, float cosTheta) {
    /*
        Hardcoded metals
        https://shaderlabs.org/wiki/LabPBR_Material_Standard
        Metal	    Bit Value	N (R, G, B)	                K (R, G, B)
        Iron	    230	        2.9114,  2.9497,  2.5845	3.0893, 2.9318, 2.7670
        Gold	    231	        0.18299, 0.42108, 1.3734	3.4242, 2.3459, 1.7704
        Aluminum	232	        1.3456,  0.96521, 0.61722	7.4746, 6.3995, 5.3031
        Chrome	    233	        3.1071,  3.1812,  2.3230	3.3314, 3.3291, 3.1350
        Copper	    234	        0.27105, 0.67693, 1.3164	3.6092, 2.6248, 2.2921
        Lead	    235	        1.9100,  1.8300,  1.4400	3.5100, 3.4000, 3.1800
        Platinum	236	        2.3757,  2.0847,  1.8453	4.2655, 3.7153, 3.1365
        Silver	    237	        0.15943, 0.14512, 0.13547	3.9291, 3.1900, 2.3808
    */
    const vec3[] METAL_IOR = vec3[](
        vec3(2.9114, 2.9497, 2.5845),
        vec3(0.18299, 0.42108, 1.3734),
        vec3(1.3456, 0.96521, 0.61722),
        vec3(3.1071, 3.1812, 2.3230),
        vec3(0.27105, 0.67693, 1.3164),
        vec3(1.9100, 1.8300, 1.4400),
        vec3(2.3757, 2.0847, 1.8453),
        vec3(0.15943, 0.14512, 0.13547)
    );

    const vec3[] METAL_K = vec3[](
        vec3(3.0893, 2.9318, 2.7670),
        vec3(3.4242, 2.3459, 1.7704),
        vec3(7.4746, 6.3995, 5.3031),
        vec3(3.3314, 3.3291, 3.1350),
        vec3(3.6092, 2.6248, 2.2921),
        vec3(3.5100, 3.4000, 3.1800),
        vec3(4.2655, 3.7153, 3.1365),
        vec3(3.9291, 3.1900, 2.3808)
    );

    vec3 f = vec3(0.0);
    if (material.f0 < 229.5 / 255.0) {
        f = fresnel_dielectricDielectric_reflection(cosTheta, vec3(AIR_IOR), vec3(fresnel_f0ToIor(material.f0)));
//        f = frenel_schlick(cosTheta, vec3(material.f0));
    } else if (material.f0 < 237.5 / 255.0) {
        uint metalIdx = clamp(uint(material.f0 * 255.0) - 230u, 0u, 7u);
        vec3 ior = METAL_IOR[metalIdx];
        vec3 k = METAL_K[metalIdx];
        f = fresnel_dielectricConductor(cosTheta, ior, k);
    } else {
        f = frenel_schlick(cosTheta, material.albedo.rgb * 0.9 + 0.1);
    }

    return saturate(f);
}

#endif