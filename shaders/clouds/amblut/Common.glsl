#ifndef AMBLUT_DATA_MODIFIER
#define AMBLUT_DATA_MODIFIER \

#endif

#define AMBIENT_IRRADIANCE_LUT_SIZE 16
#define SAMPLE_COUNT 256

layout(std430, binding = 2) AMBLUT_DATA_MODIFIER buffer AmbLUTWorkingBuffer {
    vec2 rayDir[SAMPLE_COUNT];
    vec3 inSctr[SAMPLE_COUNT];
} ssbo_ambLUTWorkingBuffer;