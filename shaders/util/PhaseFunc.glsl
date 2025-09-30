/*
    References:
        [DRA03] Draine, B.T. "Scattering by Interstellar Dust Grains. I. Optical and Ultraviolet"
            The Astrophysical Journal. 2003.
            https://doi.org/10.1086/379118
        [JEN23] Jendersie, Johannes. d'Eon, Eugene. "An Approximate Mie Scattering Function for Fog and Cloud Rendering"
            SIGGRAPH 2023. 2023.
            https://research.nvidia.com/labs/rtr/approximate-mie/publications/approximate-mie.pdf

     Credits:
        Jessie - Bi-Lambertian plate and Klein-Nishina phase functions

        You can find full license texts in /licenses
*/
#ifndef INCLUDE_util_PhaseFunc_glsl
#define INCLUDE_util_PhaseFunc_glsl a

#include "Math.glsl"

const float UNIFORM_PHASE = 1.0 / (4.0 * PI);

float phasefunc_Rayleigh(float cosTheta) {
    float k = 3.0 / (16.0 * PI);
    return k * (1.0 + pow2(cosTheta));
}

float phasefunc_HenyeyGreenstein(float cosTheta, float g) {
    float g2 = pow2(g);
    float numerator = 1.0 - g2;
    float denominator = pow(1.0 + g2 - 2.0 * g * cosTheta, 3.0 / 2.0);
    float term0 = UNIFORM_PHASE;
    float term1 = numerator / denominator;
    return term0 * term1;
}

vec3 phasefunc_HenyeyGreenstein(float cosTheta, vec3 g) {
    vec3 g2 = pow2(g);
    vec3 numerator = 1.0 - g2;
    vec3 denominator = pow(1.0 + g2 - 2.0 * g * cosTheta, vec3(3.0 / 2.0));
    float term0 = UNIFORM_PHASE;
    vec3 term1 = numerator / denominator;
    return term0 * term1;
}

float phasefunc_CornetteShanks(float cosTheta, float g) {
    float k = 3.0 / (8.0 * PI) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + pow2(cosTheta)) / pow(1.0 + g * g - 2.0 * g * cosTheta, 1.5);
}

vec3 phasefunc_CornetteShanks(float cosTheta, vec3 g) {
    vec3 k = 3.0 / (8.0 * PI) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + pow2(cosTheta)) / pow(1.0 + g * g - 2.0 * g * cosTheta, vec3(1.5));
}

// [DRA03]
float phasefunc_Draine(float cosTheta, float g, float alpha) {
    float g2 = pow2(g);
    float numerator1 = 1.0 - g2;
    float denominator1 = pow(1.0 + g2 - 2.0 * g * cosTheta, 3.0 / 2.0);
    float numerator2 = 1.0 + alpha * pow2(cosTheta);
    float denominator2 = 1.0 + alpha * (1.0 + 2.0 * g2) / 3.0;
    float term0 = UNIFORM_PHASE;
    float term1 = numerator1 / denominator1;
    float term2 = numerator2 / denominator2;
    return term0 * term1 * term2;
}

// [JEN23]
// d: droplet diameter in µm (micrometers)
float phasefunc_HenyeyGreensteinDraine(float cosTheta, float d) {
    float gHG = exp(-0.0990567 / (d - 1.67154));
    float gD = exp(-2.20679 / (d + 3.91029) - 0.428934);
    float a = exp(3.62489 - 8.29288 / (d + 5.52825));
    float wD = exp(-0.599085 / (d - 0.641583) - 0.665888);
    return mix(phasefunc_HenyeyGreenstein(cosTheta, gHG), phasefunc_Draine(cosTheta, gD, a), wD);
}

float phasefunc_BiLambertianPlate(float cosTheta, float g) {
    float phase = 2.0 * (-PI * g * cosTheta + sqrt(saturate(1.0 - pow2(cosTheta))) + cosTheta * acos(-cosTheta));
    return phase / (3.0 * pow2(PI));
}

float phasefunc_KleinNishinaE(float cosTheta, float e) {
    return e / (2.0 * PI * (e * (1.0 - cosTheta) + 1.0) * log(2.0 * e + 1.0));
}

vec3 phasefunc_KleinNishinaE(float cosTheta, vec3 e) {
    return e / (2.0 * PI * (e * (1.0 - cosTheta) + 1.0) * log(2.0 * e + 1.0));
}

float phasefunc_KleinNishina(float cosTheta, float g) {
    float e = 1.0;
    for (int i = 0; i < 8; ++i) {
        float gFromE = 1.0 / e - 2.0 / log(2.0 * e + 1.0) + 1.0;
        float deriv = 4.0 / ((2.0 * e + 1.0) * pow2(log(2.0 * e + 1.0))) - 1.0 / pow2(e);
        e = e - (gFromE - g) / deriv;
    }

    return phasefunc_KleinNishinaE(cosTheta, e);
}

vec3 phasefunc_KleinNishina(float cosTheta, vec3 g) {
    vec3 e = vec3(1.0);
    for (int i = 0; i < 8; ++i) {
        vec3 gFromE = 1.0 / e - 2.0 / log(2.0 * e + 1.0) + 1.0;
        vec3 deriv = 4.0 / ((2.0 * e + 1.0) * pow2(log(2.0 * e + 1.0))) - 1.0 / pow2(e);
        e = e - (gFromE - g) / deriv;
    }

    return phasefunc_KleinNishinaE(cosTheta, e);
}

#endif