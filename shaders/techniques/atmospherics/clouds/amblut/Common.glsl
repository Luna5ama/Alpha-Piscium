#include "../Common.glsl"
#include "/util/PhaseFunc.glsl"

#ifndef AMBLUT_DATA_MODIFIER
#define AMBLUT_DATA_MODIFIER buffer
#endif

#define AMBIENT_IRRADIANCE_LUT_SIZE 32
#define SAMPLE_COUNT 1024
#define SAMPLE_COUNT_D256 4

layout(std430, binding = 2) AMBLUT_DATA_MODIFIER AmbLUTWorkingBuffer {
    vec2 rayDir[SAMPLE_COUNT];
    vec3 inSctr[SAMPLE_COUNT];
    vec3 groundDirect[SAMPLE_COUNT];
} ssbo_ambLUTWorkingBuffer;

bool clouds_amblut_isCumulus() {
    return (frameCounter & 1) == 0;
}
