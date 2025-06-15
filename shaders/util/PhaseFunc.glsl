/*
    References:
        [DRA03] Draine, B.T. "Scattering by Interstellar Dust Grains. I. Optical and Ultraviolet"
            The Astrophysical Journal. 2003.
            https://doi.org/10.1086/379118
        [JEN23] Jendersie, Johannes. d'Eon, Eugene. "An Approximate Mie Scattering Function for Fog and Cloud Rendering"
            SIGGRAPH 2023. 2023.
            https://research.nvidia.com/labs/rtr/approximate-mie/publications/approximate-mie.pdf

        You can find full license texts in /licenses
*/
#ifndef INCLUDE_util_PhaseFunc_glsl
#define INCLUDE_util_PhaseFunc_glsl a

#include "Math.glsl"

const float UNIFORM_PHASE = 1.0 / (4.0 * PI);

float rayleighPhase(float cosTheta) {
    float k = 3.0 / (16.0 * PI);
    return k * (1.0 + pow2(cosTheta));
}

float henyeyGreensteinPhase(float cosTheta, float g) {
    float g2 = pow2(g);
    float numerator = 1.0 - g2;
    float denominator = pow(1.0 + g2 - 2.0 * g * cosTheta, 3.0 / 2.0);
    float term0 = UNIFORM_PHASE;
    float term1 = numerator / denominator;
    return term0 * term1;
}

// Cornette-Shanks phase function for Mie scattering
float cornetteShanksPhase(float cosTheta, float g) {
    float k = 3.0 / (8.0 * PI) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + cosTheta * cosTheta) / pow(1.0 + g * g - 2.0 * g * -cosTheta, 1.5);
}

// [DRA03]
float drainePhase(float cosTheta, float g, float alpha) {
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
float hgDrainePhase(float cosTheta, float d) {
    float gHG = exp(-0.0990567 / (d - 1.67154));
    float gD = exp(-2.20679 / (d + 3.91029) - 0.428934);
    float a = exp(3.62489 - 8.29288 / (d + 5.52825));
    float wD = exp(-0.599085 / (d - 0.641583) - 0.665888);
    return mix(henyeyGreensteinPhase(cosTheta, gHG), drainePhase(cosTheta, gD, a), wD);
}

#endif