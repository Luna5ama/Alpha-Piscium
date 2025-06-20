#include "../Common.glsl"
#include "/util/PhaseFunc.glsl"

#ifndef AMBLUT_DATA_MODIFIER
#define AMBLUT_DATA_MODIFIER \

#endif

#define AMBIENT_IRRADIANCE_LUT_SIZE 16
#define SAMPLE_COUNT 256

layout(std430, binding = 2) AMBLUT_DATA_MODIFIER buffer AmbLUTWorkingBuffer {
    vec2 rayDir[SAMPLE_COUNT];
    vec3 inSctr[SAMPLE_COUNT];
} ssbo_ambLUTWorkingBuffer;

int clouds_amblut_currLayerIndex() {
    return frameCounter % 6;
}

const float _CLOUDS_AMBLUT_HEIGHTS[] = float[](
    1.0, // Stratus
    SETTING_CLOUDS_CU_HEIGHT, // Cumulus
    4.0, // Altocumulus
    SETTING_CLOUDS_CI_HEIGHT, // Cirrus/Cirrocumulus
    10.0, // Cirrostratus
    80.0 // Noctilucent
);

float clouds_amblut_height(int layerIndex) {
    return _CLOUDS_AMBLUT_HEIGHTS[layerIndex];
}

vec3 clouds_amblut_phase(float cosTheta, int layerIndex) {
    vec3 phase = vec3(UNIFORM_PHASE);
    if (layerIndex == 0) {
        phase = cornetteShanksPhase(cosTheta, CLOUDS_ST_ASYM);
    }
    if (layerIndex == 1) {
        phase = clouds_phase_cu(cosTheta, SETTING_CLOUDS_CU_PHASE_RATIO * 0.5);
    }
    if (layerIndex == 2) {
        phase = mix(cornetteShanksPhase(cosTheta, CLOUDS_CI_ASYM), cornetteShanksPhase(cosTheta, CLOUDS_CU_ASYM), 0.8);
    }
    if (layerIndex == 3) {
        phase = clouds_phase_ci(cosTheta, SETTING_CLOUDS_CI_PHASE_RATIO * 0.5);
    }
    if (layerIndex == 4) {
        phase = mix(cornetteShanksPhase(cosTheta, CLOUDS_ST_ASYM), cornetteShanksPhase(cosTheta, CLOUDS_CI_ASYM), 0.8);
    }
    if (layerIndex == 5) {
        phase = cornetteShanksPhase(cosTheta, CLOUDS_CI_ASYM);
    }
    return phase;
}