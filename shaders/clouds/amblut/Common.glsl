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
    SETTING_CLOUDS_CU_HEIGHT, // Stratus
    SETTING_CLOUDS_CU_HEIGHT, // Cumulus
    SETTING_CLOUDS_CU_HEIGHT, // Altocumulus
    SETTING_CLOUDS_CI_HEIGHT, // Cirrus/Cirrocumulus
    SETTING_CLOUDS_CI_HEIGHT, // Cirrostratus
    SETTING_CLOUDS_CI_HEIGHT // Noctilucent
);

float clouds_amblut_height(int layerIndex) {
    return _CLOUDS_AMBLUT_HEIGHTS[layerIndex];
}

vec3 clouds_amblut_phase(float cosTheta, int layerIndex) {
    vec3 phase = vec3(UNIFORM_PHASE);
    if (layerIndex == 1) {
        phase = cornetteShanksPhase(cosTheta, CLOUDS_CU_ASYM);
    }
    if (layerIndex == 3) {
        phase = cornetteShanksPhase(cosTheta, CLOUDS_CI_ASYM);
    }
    return phase;
}